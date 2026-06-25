# mymodel/model_pmr.py
# PMR (Prototypical Modal Rebalance, Fan et al. CVPR 2023) adaptation of FlexCare.
# Loss-based variant: the per-modality "confidence" signal used for gradient
# modulation is exp(-L_m), where L_m is the task loss of an auxiliary head that
# only sees modality m's fused CLS token.
#
# Main prediction path is unchanged from FlexCare (so eval scripts work identically).
# Training-time forward additionally returns per-modality auxiliary logits.

from __future__ import annotations
import torch
import torch.nn as nn

from mymodel.model_wo import FlexCare, calculate_ortho_loss
from mymodel.module import generate_cross_modal_mask
from utils import length_to_mask


_TASK_KEY = {0: "mortality", 1: "decomp", 2: "ph", 3: "los",
             4: "readm", 5: "diag", 6: "drg"}


def _make_task_heads(hidden_dim: int) -> nn.ModuleDict:
    return nn.ModuleDict({
        "mortality": nn.Linear(hidden_dim, 1),
        "decomp":    nn.Linear(hidden_dim, 1),
        "ph":        nn.Linear(hidden_dim, 25),
        "los":       nn.Linear(hidden_dim, 10),
        "readm":     nn.Linear(hidden_dim, 1),
        "diag":      nn.Linear(hidden_dim, 14),
        "drg":       nn.Linear(hidden_dim, 769),
    })


class FlexCarePMR(FlexCare):
    """
    FlexCare with per-modality auxiliary heads for PMR's confidence signal.
    Main prediction path is the standard FlexCare head (unchanged), so checkpoints
    are interchangeable with FlexCare for eval purposes (auxiliary heads are
    simply unused at eval).
    """

    def __init__(self, ehr_dim=76, hidden_dim=128, layers=4,
                 device=torch.device("cpu"), **kw):
        super().__init__(ehr_dim=ehr_dim, hidden_dim=hidden_dim, layers=layers,
                         device=device, **kw)
        self.aux_ehr  = _make_task_heads(hidden_dim)
        self.aux_cxr  = _make_task_heads(hidden_dim)
        self.aux_note = _make_task_heads(hidden_dim)

    def _aux_logits(self, modality: str, task_idx: int, x: torch.Tensor) -> torch.Tensor:
        if modality == "ehr":
            heads = self.aux_ehr
        elif modality == "cxr":
            heads = self.aux_cxr
        else:
            heads = self.aux_note
        return heads[_TASK_KEY[task_idx]](x)

    def forward(self, ehr, ehr_lengths, use_ehr, img, use_img, note, use_note, task_index):
        device = self.device
        B = ehr.size(0)

        task_embed = self.task_embedding(task_index).unsqueeze(1)

        # ----- EHR -----
        ehr_embed = self.ehr_projection(ehr)
        ehr_cls_tokens = self.ehr_cls_token.repeat(B, 1, 1)
        ehr_embed = ehr_embed + self.ehr_pos_embed[:, : ehr_embed.shape[1], :]
        ehr_embed = torch.cat((ehr_cls_tokens, ehr_embed), dim=1)
        ehr_embed = self._feature_knockout(ehr_embed, branch="ehr")
        ehr_lengths_t = torch.tensor(ehr_lengths, device=device)
        ehr_lengths_with_cls = ehr_lengths_t + use_ehr
        ehr_pad_mask = length_to_mask(ehr_lengths_with_cls, max_len=ehr_embed.shape[1])

        # ----- CXR -----
        cxr_embed = self.patch_projection(img)
        cxr_cls_tokens = self.cxr_cls_token.repeat(B, 1, 1)
        cxr_embed = cxr_embed + self.cxr_pos_embed[:, : cxr_embed.shape[1], :]
        cxr_embed = torch.cat((cxr_cls_tokens, cxr_embed), dim=1)
        cxr_embed = self._feature_knockout(cxr_embed, branch="cxr")
        cxr_pad_mask = length_to_mask(use_img, max_len=1).repeat(1, cxr_embed.shape[1])

        # ----- Note -----
        with torch.no_grad():
            enc = self.tokenizer(note, padding=True, truncation=True,
                                 max_length=512, add_special_tokens=False,
                                 return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            if attention_mask.sum() != 0:
                outs = self.note_projection(input_ids, attention_mask=attention_mask)
                note_embed = outs.last_hidden_state
            else:
                note_embed = torch.zeros((B, 1, self.note_fc.in_features), device=device)
                attention_mask = torch.zeros((B, 1), device=device).int()

        note_embed = self.note_fc(note_embed)
        note_cls_tokens = self.note_cls_token.repeat(B, 1, 1)
        note_embed = note_embed + self.note_pos_embed[:, : note_embed.shape[1], :]
        if attention_mask.sum() != 0:
            note_embed = torch.cat((note_cls_tokens, note_embed), dim=1)
            note_lengths = attention_mask.sum(dim=1) + use_note
        else:
            note_embed = note_cls_tokens
            note_lengths = torch.zeros_like(use_note)
        note_embed = self._feature_knockout(note_embed, branch="note")
        note_pad_mask = length_to_mask(note_lengths, max_len=note_embed.shape[1])

        # ----- Fusion -----
        mm_cls = self.mm_cls_token
        for i in range(3):
            mm_cls = torch.cat((mm_cls, self.cross_cls_tokens[i].unsqueeze(0)), dim=1)
        mm_cls = mm_cls.repeat(B, 1, 1)

        mm_embed = torch.cat((task_embed, mm_cls, ehr_embed, cxr_embed, note_embed), dim=1)
        cls_pad_mask  = length_to_mask(4 * torch.ones(use_img.shape, device=device), max_len=4)
        task_pad_mask = length_to_mask(torch.ones(use_img.shape, device=device), max_len=1)
        mm_pad_mask = torch.cat(
            (task_pad_mask, cls_pad_mask, ehr_pad_mask, cxr_pad_mask, note_pad_mask), dim=1
        )

        ehr_cls_index = 5
        cxr_cls_index = ehr_cls_index + ehr_embed.shape[1]
        note_cls_index = cxr_cls_index + cxr_embed.shape[1]
        cross_cls_mask = generate_cross_modal_mask(
            ehr_cls_index, cxr_cls_index, note_cls_index, mm_embed.shape[1]
        ).to(device)

        mm_embed = mm_embed.transpose(0, 1)
        fusion_embed = self.transformer_fusion(
            mm_embed, mask=cross_cls_mask, src_key_padding_mask=mm_pad_mask
        )
        fusion_embed = fusion_embed.transpose(0, 1)

        task_mm_embed = fusion_embed[:, 0]
        ehr_cls = fusion_embed[:, ehr_cls_index]
        cxr_cls = fusion_embed[:, cxr_cls_index]
        note_cls = fusion_embed[:, note_cls_index]
        modality_avg = torch.stack([ehr_cls, cxr_cls, note_cls], dim=1).mean(dim=1)
        final_mm_embed = torch.cat([task_mm_embed, modality_avg], dim=1)

        # Main task head (unchanged from FlexCare)
        ti0 = int(task_index[0].item())
        if ti0 == 0:
            scores = torch.sigmoid(self.dense_layer_mortality(final_mm_embed))
        elif ti0 == 1:
            scores = torch.sigmoid(self.dense_layer_decomp(final_mm_embed))
        elif ti0 == 3:
            scores = self.dense_layer_los(final_mm_embed)
        elif ti0 == 4:
            scores = torch.sigmoid(self.dense_layer_readm(final_mm_embed))
        elif ti0 == 5:
            scores = torch.sigmoid(self.dense_layer_diag(final_mm_embed))
        elif ti0 == 6:
            scores = self.dense_layer_drg(final_mm_embed)
        else:
            scores = torch.sigmoid(self.dense_layer_ph(final_mm_embed))

        ortho_loss = calculate_ortho_loss(fusion_embed)

        if self.training:
            # Per-modality auxiliary logits for PMR confidence signal
            aux_e = self._aux_logits("ehr",  ti0, ehr_cls)
            aux_c = self._aux_logits("cxr",  ti0, cxr_cls)
            aux_n = self._aux_logits("note", ti0, note_cls)
            # Apply same activation as main path so loss helpers match
            if ti0 in (3, 6):  # CE tasks return raw logits
                aux_logits = {"ehr": aux_e, "cxr": aux_c, "note": aux_n}
            else:
                aux_logits = {"ehr": torch.sigmoid(aux_e),
                              "cxr": torch.sigmoid(aux_c),
                              "note": torch.sigmoid(aux_n)}
            moe_loss = torch.tensor(0.0, device=device)
            return scores, ortho_loss, moe_loss, aux_logits
        return scores

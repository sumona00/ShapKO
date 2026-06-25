# mymodel/model_simmlm.py
# SimMLM-style adaptation of FlexCare (Li et al., 2025).
#
# Backbone: FlexCare encoders + cross-modal transformer (unchanged).
# Output: per-modality "expert" task heads on top of each modality's fused CLS,
#         combined by a learnable gating network at the LOGIT level (DMoME).
# Missing modalities: gating logits set to -inf -> weight = 0 after softmax.
#
# The model returns the FlexCare-compatible interface so existing eval scripts
# (Test_eval_base.py) work without modification:
#   training: (scores, ortho_loss, moe_loss)
#   eval:     scores
#
# MoFe ranking loss is computed in the training script by running this model on
# pairs of modality subsets (x+, x-).

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from mymodel.model_wo import FlexCare, calculate_ortho_loss
from mymodel.module import generate_cross_modal_mask
from utils import length_to_mask


# task_index -> head key
_TASK_KEY = {0: "mortality", 1: "decomp", 2: "ph", 3: "los",
             4: "readm", 5: "diag", 6: "drg"}
_CE_TASK_INDICES = {3, 6}  # length-of-stay, drg


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


class FlexCareSimMLM(FlexCare):
    """
    SimMLM-style FlexCare:
      - 3 modality experts: each is a per-task linear head taking that modality's fused CLS token.
      - Gating network: maps concatenated [ehr_cls, cxr_cls, note_cls] -> 3 gating logits.
        Missing-modality entries get -inf so their softmax weight is 0.
      - Logit-level mixture: y = sum_m w_m * o_m (then sigmoid for binary tasks).
    """

    def __init__(self, ehr_dim=76, hidden_dim=128, layers=4,
                 device=torch.device("cpu"), **kw):
        super().__init__(ehr_dim=ehr_dim, hidden_dim=hidden_dim, layers=layers,
                         device=device, **kw)
        D = hidden_dim

        # Per-modality expert task heads
        self.expert_ehr  = _make_task_heads(D)
        self.expert_cxr  = _make_task_heads(D)
        self.expert_note = _make_task_heads(D)

        # Gating network: takes [ehr_cls || cxr_cls || note_cls] -> 3 gating logits
        self.gating_net = nn.Sequential(
            nn.Linear(3 * D, D),
            nn.GELU(),
            nn.Linear(D, 3),
        )

    # ------------------------------------------------------------------
    # Backbone forward up through fused per-modality CLS tokens.
    # ------------------------------------------------------------------
    def _encode(self, ehr, ehr_lengths, use_ehr, img, use_img, note, use_note, task_index):
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
            (task_pad_mask, cls_pad_mask, ehr_pad_mask, cxr_pad_mask, note_pad_mask),
            dim=1,
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
        fusion_embed = fusion_embed.transpose(0, 1)  # (B, T, D)

        ehr_cls = fusion_embed[:, ehr_cls_index]
        cxr_cls = fusion_embed[:, cxr_cls_index]
        note_cls = fusion_embed[:, note_cls_index]
        return fusion_embed, ehr_cls, cxr_cls, note_cls

    # ------------------------------------------------------------------
    # Per-modality expert head selection
    # ------------------------------------------------------------------
    def _expert_logits(self, modality: str, task_idx: int, x: torch.Tensor) -> torch.Tensor:
        if modality == "ehr":
            heads = self.expert_ehr
        elif modality == "cxr":
            heads = self.expert_cxr
        elif modality == "note":
            heads = self.expert_note
        else:
            raise ValueError(modality)
        return heads[_TASK_KEY[task_idx]](x)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, ehr, ehr_lengths, use_ehr, img, use_img, note, use_note, task_index):
        device = self.device
        fusion_embed, ehr_cls, cxr_cls, note_cls = self._encode(
            ehr, ehr_lengths, use_ehr, img, use_img, note, use_note, task_index,
        )

        task_idx = int(task_index[0].item())

        # Per-modality expert logits
        o_e = self._expert_logits("ehr",  task_idx, ehr_cls)
        o_c = self._expert_logits("cxr",  task_idx, cxr_cls)
        o_n = self._expert_logits("note", task_idx, note_cls)

        # Gating: zero-out per-modality features when modality is missing so the
        # gating network doesn't latch onto noise from absent modalities.
        ue = use_ehr.float().unsqueeze(1)
        uc = use_img.float().unsqueeze(1)
        un = use_note.float().unsqueeze(1)
        gate_in = torch.cat([ehr_cls * ue, cxr_cls * uc, note_cls * un], dim=1)
        gate_logits = self.gating_net(gate_in)               # (B, 3)
        avail = torch.cat([ue, uc, un], dim=1)               # (B, 3)
        any_present = (avail.sum(dim=1, keepdim=True) > 0).float()
        masked = gate_logits.masked_fill(avail < 0.5, float("-inf"))
        # Edge case: all-missing rows -> fall back to raw logits (avoids softmax-of-all-(-inf) NaN)
        masked = torch.where(any_present.bool().expand_as(masked), masked, gate_logits)
        weights = F.softmax(masked, dim=1)
        # Force exact zero on missing rows (numerical cleanup) and renormalize
        weights = weights * avail
        s = weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        weights = weights / s

        we, wc, wn = weights[:, 0:1], weights[:, 1:2], weights[:, 2:3]
        mixed_logits = we * o_e + wc * o_c + wn * o_n

        if task_idx in _CE_TASK_INDICES:
            scores = mixed_logits  # CrossEntropyLoss consumes raw logits
        else:
            scores = torch.sigmoid(mixed_logits)  # BCE on probabilities

        ortho_loss = calculate_ortho_loss(fusion_embed)

        if self.training:
            moe_loss = torch.tensor(0.0, device=device)
            return scores, ortho_loss, moe_loss
        return scores

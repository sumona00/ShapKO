# mymodel/model_moddroppp.py
# ModDrop++ adaptation of FlexCare (Liu et al., 2022).
#
# - ModDrop training (random modality dropout) is handled in the training script.
# - Dynamic head: a small MLP that takes a 3-bit modality availability code and
#   produces FiLM-style scaling vectors for each modality's first projection
#   (ehr_projection, patch_projection, note_fc). This is a lightweight adaptation
#   of the paper's filter scaling (M ∈ R^{u×v}) — we scale only along the output
#   channel dim per-sample, which is equivalent to scaling all kernel weights
#   producing each output channel. This keeps training fast and works uniformly
#   for the EHR linear, CXR conv, and note FC layers.
# - Intra-subject co-training is handled in the training script: the model is
#   forwarded twice (full-modality, missing-modality) and the post-projection
#   feature maps are compared with cosine similarity.
#
# Returns the FlexCare-compatible interface so existing eval scripts work.

from __future__ import annotations
import torch
import torch.nn as nn

from mymodel.model_wo import FlexCare, calculate_ortho_loss
from mymodel.module import generate_cross_modal_mask
from utils import length_to_mask


class DynamicHead(nn.Module):
    """
    Maps a 3-bit modality code (B, 3) -> per-modality output-channel scales of
    shape (B, hidden_dim), bounded to [0, 2] via sigmoid * 2 (so identity = 1.0
    is reachable).
    """

    def __init__(self, hidden_dim: int, modality_code_dim: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.code_embed = nn.Sequential(
            nn.Linear(modality_code_dim, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.GELU(),
        )
        # 3 modalities × hidden_dim
        self.head = nn.Linear(128, 3 * hidden_dim)
        # Initialize so initial scale is ~1.0 (sigmoid(0)*2 = 1.0)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, code: torch.Tensor) -> dict:
        h = self.code_embed(code)
        s = torch.sigmoid(self.head(h)) * 2.0  # (B, 3*D)
        s = s.view(-1, 3, self.hidden_dim)
        return {"ehr": s[:, 0], "cxr": s[:, 1], "note": s[:, 2]}


class FlexCareModDropPP(FlexCare):
    """
    FlexCare + Dynamic Head (per-sample FiLM scaling on each modality's first projection).
    The forward also caches post-projection feature maps in `self.last_proj` so the
    training script can compute the intra-subject similarity loss.
    """

    def __init__(self, ehr_dim=76, hidden_dim=128, layers=4,
                 device=torch.device("cpu"), **kw):
        super().__init__(ehr_dim=ehr_dim, hidden_dim=hidden_dim, layers=layers,
                         device=device, **kw)
        self.dynamic_head = DynamicHead(hidden_dim=hidden_dim)
        self.last_proj = {"ehr": None, "cxr": None, "note": None}

    @staticmethod
    def _modality_code(use_ehr, use_img, use_note) -> torch.Tensor:
        return torch.stack([use_ehr.float(), use_img.float(), use_note.float()], dim=1)

    def forward(self, ehr, ehr_lengths, use_ehr, img, use_img, note, use_note, task_index):
        device = self.device
        B = ehr.size(0)

        # Per-sample dynamic scales conditioned on the modality availability code
        code = self._modality_code(use_ehr, use_img, use_note)
        scales = self.dynamic_head(code)
        s_ehr  = scales["ehr"].unsqueeze(1)   # (B, 1, D) — broadcast across T
        s_cxr  = scales["cxr"].unsqueeze(1)   # (B, 1, D) — broadcast across patches
        s_note = scales["note"].unsqueeze(1)  # (B, 1, D) — broadcast across L

        task_embed = self.task_embedding(task_index).unsqueeze(1)

        # ----- EHR -----
        ehr_proj = self.ehr_projection(ehr) * s_ehr   # FiLM scaling
        self.last_proj["ehr"] = ehr_proj
        ehr_cls_tokens = self.ehr_cls_token.repeat(B, 1, 1)
        ehr_proj = ehr_proj + self.ehr_pos_embed[:, : ehr_proj.shape[1], :]
        ehr_embed = torch.cat((ehr_cls_tokens, ehr_proj), dim=1)
        ehr_embed = self._feature_knockout(ehr_embed, branch="ehr")

        ehr_lengths_t = torch.tensor(ehr_lengths, device=device)
        ehr_lengths_with_cls = ehr_lengths_t + use_ehr
        ehr_pad_mask = length_to_mask(ehr_lengths_with_cls, max_len=ehr_embed.shape[1])

        # ----- CXR -----
        cxr_proj = self.patch_projection(img) * s_cxr
        self.last_proj["cxr"] = cxr_proj
        cxr_cls_tokens = self.cxr_cls_token.repeat(B, 1, 1)
        cxr_proj = cxr_proj + self.cxr_pos_embed[:, : cxr_proj.shape[1], :]
        cxr_embed = torch.cat((cxr_cls_tokens, cxr_proj), dim=1)
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
                note_raw = outs.last_hidden_state
            else:
                note_raw = torch.zeros((B, 1, self.note_fc.in_features), device=device)
                attention_mask = torch.zeros((B, 1), device=device).int()

        note_proj = self.note_fc(note_raw) * s_note
        self.last_proj["note"] = note_proj
        note_cls_tokens = self.note_cls_token.repeat(B, 1, 1)
        note_proj = note_proj + self.note_pos_embed[:, : note_proj.shape[1], :]
        if attention_mask.sum() != 0:
            note_embed = torch.cat((note_cls_tokens, note_proj), dim=1)
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
        fusion_embed = fusion_embed.transpose(0, 1)

        task_mm_embed = fusion_embed[:, 0]
        ehr_cls = fusion_embed[:, ehr_cls_index]
        cxr_cls = fusion_embed[:, cxr_cls_index]
        note_cls = fusion_embed[:, note_cls_index]
        modality_avg = torch.stack([ehr_cls, cxr_cls, note_cls], dim=1).mean(dim=1)
        final_mm_embed = torch.cat([task_mm_embed, modality_avg], dim=1)

        # Task-specific head (same as base FlexCare)
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
            moe_loss = torch.tensor(0.0, device=device)
            return scores, ortho_loss, moe_loss
        return scores

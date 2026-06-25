# mymodel/model_wo.py  (PATCHED: MISS/KO tokens + modality-level knockout)
from __future__ import annotations
import os, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import length_to_mask
from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM
from mymodel.module import PatchEmbed, generate_cross_modal_mask

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

os.environ["TOKENIZERS_PARALLELISM"] = "false"
HF_ID = "dmis-lab/biobert-base-cased-v1.2"
CACHE_DIR = "/midtier/sablab/scratch/gay9002/FlexCare/mymodel/pretrained/biobert-base-cased-v1.2"
tokenizer_global = AutoTokenizer.from_pretrained(HF_ID, cache_dir=CACHE_DIR, use_fast=False)
maskedLM_global = AutoModelForMaskedLM.from_pretrained(HF_ID, cache_dir=CACHE_DIR)


def calculate_ortho_loss(input_vec: torch.Tensor) -> torch.Tensor:
    x = input_vec - torch.mean(input_vec, axis=2, keepdim=True)
    cov_matrix = torch.matmul(x, x.transpose(1, 2)) / (x.shape[2] - 1)
    loss = (
        torch.sum(cov_matrix ** 2)
        - torch.sum(torch.diagonal(cov_matrix, dim1=1, dim2=2) ** 2)
    ) / (cov_matrix.shape[0] * (cov_matrix.shape[1] - 1) * (cov_matrix.shape[2] - 1))
    return loss


class FlexCare(nn.Module):
    """
    Supports Shapley-based adaptive modality knockout by:
      - keep_mask per modality: 1=use, 0=knockout
      - avail mask per modality: 1=present in data, 0=missing in data
      - MISS token vs KO token (learned) to distinguish the two
      - stable masking: if KO/MISS -> keep only CLS token for that modality (no leakage)
      - optional embedding L2 normalization before placeholder injection
    """
    def __init__(
        self,
        ehr_dim: int = 76,
        num_classes: int = 1,
        hidden_dim: int = 128,
        batch_first: bool = True,
        dropout: float = 0.0,
        layers: int = 4,
        expert_k: int = 2,
        expert_total: int = 10,
        device: torch.device = torch.device("cpu"),
        normalize_before_placeholder: bool = True,
    ):
        super().__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.normalize_before_placeholder = normalize_before_placeholder

        self.task_embedding = nn.Embedding(40, hidden_dim)

        # ---- EHR ----
        self.ehr_projection = nn.Linear(ehr_dim, hidden_dim)
        self.ehr_cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.ehr_pos_embed = nn.Parameter(torch.zeros(1, 600, hidden_dim))

        # ---- CXR ----
        self.patch_projection = PatchEmbed(patch_size=16, embed_dim=hidden_dim)
        num_patches = (224 // 16) * (224 // 16)
        self.cxr_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, hidden_dim))
        self.cxr_cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        # ---- NOTE ----
        self.note_projection = AutoModel.from_pretrained(HF_ID, cache_dir=CACHE_DIR).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(HF_ID, cache_dir=CACHE_DIR, use_fast=False)
        self.note_fc = nn.Linear(768, hidden_dim)
        self.note_cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.note_pos_embed = nn.Parameter(torch.zeros(1, 600, hidden_dim))

        # ---- Fusion tokens ----
        self.cross_cls_tokens = nn.Parameter(torch.zeros(3, 1, hidden_dim))
        self.mm_cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        # ---- NEW: learned placeholder tokens (MISS vs KO) per modality ----
        # Shape: (1,1,D) so we can broadcast per-batch
        self.miss_token = nn.ParameterDict({
            "ehr":  nn.Parameter(torch.zeros(1, 1, hidden_dim)),
            "cxr":  nn.Parameter(torch.zeros(1, 1, hidden_dim)),
            "note": nn.Parameter(torch.zeros(1, 1, hidden_dim)),
        })
        self.ko_token = nn.ParameterDict({
            "ehr":  nn.Parameter(torch.zeros(1, 1, hidden_dim)),
            "cxr":  nn.Parameter(torch.zeros(1, 1, hidden_dim)),
            "note": nn.Parameter(torch.zeros(1, 1, hidden_dim)),
        })
        # small init helps stability
        for k in self.miss_token:
            nn.init.normal_(self.miss_token[k], mean=0.0, std=0.02)
            nn.init.normal_(self.ko_token[k],   mean=0.0, std=0.02)

        # ---- Transformer ----
        self.encoder_layer_fusion = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=2,
            dim_feedforward=hidden_dim * 4,
            batch_first=False,
        )
        self.transformer_fusion = nn.TransformerEncoder(self.encoder_layer_fusion, num_layers=layers)

        # ---- Heads ----
        self.dense_layer_mortality = nn.Linear(hidden_dim * 2, 1)
        self.dense_layer_decomp = nn.Linear(hidden_dim * 2, 1)
        self.dense_layer_ph = nn.Linear(hidden_dim * 2, 25)
        self.dense_layer_los = nn.Linear(hidden_dim * 2, 10)
        self.dense_layer_readm = nn.Linear(hidden_dim * 2, 1)
        self.dense_layer_diag = nn.Linear(hidden_dim * 2, 14)
        self.dense_layer_drg = nn.Linear(hidden_dim * 2, 769)

    def _l2norm(self, x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        # normalize last dim
        return x / (x.norm(dim=-1, keepdim=True) + eps)

    def _apply_modality_state(
        self,
        x_embed: torch.Tensor,        # (B, L, D) includes CLS at position 0
        pad_mask: torch.Tensor,       # (B, L) True=PAD? (depends on your length_to_mask; yours seems 1=valid)
        avail: torch.Tensor,          # (B,) 1=present, 0=missing
        keep: torch.Tensor,           # (B,) 1=keep, 0=knockout
        branch: str,                  # "ehr"|"cxr"|"note"
    ):
        """
        Stable modality KO/MISS:
          - If missing (avail=0): keep only CLS token but set it to MISS token
          - If knocked out (avail=1, keep=0): keep only CLS token but set it to KO token
          - If kept (avail=1, keep=1): keep sequence as-is
        Also masks out non-CLS tokens when MISS/KO.
        """
        device = x_embed.device
        B, L, D = x_embed.shape

        # optional normalization before placeholder injection
        if self.normalize_before_placeholder:
            x_embed = self._l2norm(x_embed)

        # indices
        miss = (avail <= 0).bool()
        ko   = (avail > 0).bool() & (keep <= 0).bool()

        # When MISS/KO: force only CLS token to be valid, rest padded
        # Your length_to_mask seems to return a mask of "valid positions" (not pad) based on lengths.
        # In your code, you pass it as src_key_padding_mask, which expects True=PAD.
        # But you already use it that way; so we keep consistent with your existing semantics:
        # If your length_to_mask returns True for VALID, you must invert before feeding transformer.
        # In your current code you pass multimodal_pad_mask directly as src_key_padding_mask,
        # so length_to_mask must be returning PAD mask (True=PAD). We'll assume that.
        #
        # We'll construct a pad mask that pads everything except CLS for MISS/KO.
        if miss.any() or ko.any():
            forced_pad = torch.ones((B, L), dtype=torch.bool, device=device)
            forced_pad[:, 0] = False  # keep CLS as not padded
            pad_mask = torch.where((miss | ko).unsqueeze(1), forced_pad, pad_mask)

        # Replace CLS embedding
        if miss.any():
            x_embed[miss, 0:1, :] = self.miss_token[branch].to(device)
        if ko.any():
            x_embed[ko, 0:1, :] = self.ko_token[branch].to(device)

        return x_embed, pad_mask

    def forward(
        self,
        ehr,
        ehr_lengths,
        use_ehr,          # (B,) 1/0 availability from dataloader
        img,
        use_img,          # (B,) 1/0 availability
        note,
        use_note,         # (B,) 1/0 availability (1 if non-empty)
        task_index,
        keep_masks=None,  # dict: {"ehr":(B,), "cxr":(B,), "note":(B,)}; if None => keep all
    ):
        device = self.device
        B = ehr.size(0)

        if keep_masks is None:
            keep_masks = {
                "ehr":  torch.ones_like(use_img, device=device),
                "cxr":  torch.ones_like(use_img, device=device),
                "note": torch.ones_like(use_img, device=device),
            }
        # ensure tensors on device
        avail_ehr  = use_ehr.to(device).long()
        avail_cxr  = use_img.to(device).long()
        avail_note = use_note.to(device).long()
        keep_ehr   = keep_masks["ehr"].to(device).long()
        keep_cxr   = keep_masks["cxr"].to(device).long()
        keep_note  = keep_masks["note"].to(device).long()

        # -------- Task embedding --------
        task_embed = self.task_embedding(task_index).unsqueeze(1)  # (B,1,D)

        # -------- EHR --------
        ehr_embed = self.ehr_projection(ehr)  # (B,T,D)
        ehr_cls_tokens = self.ehr_cls_token.repeat(B, 1, 1)
        ehr_embed = ehr_embed + self.ehr_pos_embed[:, : ehr_embed.shape[1], :]
        ehr_embed = torch.cat((ehr_cls_tokens, ehr_embed), dim=1)  # (B,T+1,D)

        ehr_lengths = torch.tensor(ehr_lengths, device=device)
        ehr_lengths_with_cls = ehr_lengths + avail_ehr  # if missing => +0
        ehr_pad_mask = length_to_mask(ehr_lengths_with_cls, max_len=ehr_embed.shape[1])  # (B,T+1) PAD mask

        ehr_embed, ehr_pad_mask = self._apply_modality_state(
            ehr_embed, ehr_pad_mask, avail=avail_ehr, keep=keep_ehr, branch="ehr"
        )

        # -------- CXR --------
        cxr_embed = self.patch_projection(img)  # (B,P,D)
        cxr_cls_tokens = self.cxr_cls_token.repeat(B, 1, 1)
        cxr_embed = cxr_embed + self.cxr_pos_embed[:, : cxr_embed.shape[1], :]
        cxr_embed = torch.cat((cxr_cls_tokens, cxr_embed), dim=1)  # (B,P+1,D)

        # pad mask for cxr: if present => length = P+1; if missing => length=0
        # easiest: build lengths as avail*(P+1)
        cxr_len = avail_cxr * cxr_embed.shape[1]
        cxr_pad_mask = length_to_mask(cxr_len, max_len=cxr_embed.shape[1])

        cxr_embed, cxr_pad_mask = self._apply_modality_state(
            cxr_embed, cxr_pad_mask, avail=avail_cxr, keep=keep_cxr, branch="cxr"
        )

        # -------- NOTE --------
        with torch.no_grad():
            encoding = self.tokenizer(
                note,
                padding=True,
                truncation=True,
                max_length=512,
                add_special_tokens=False,
                return_tensors="pt",
            )
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)

            if attention_mask.sum() != 0:
                outputs = self.note_projection(input_ids, attention_mask=attention_mask)
                note_embed = outputs.last_hidden_state  # (B,L,768)
            else:
                note_embed = torch.zeros((B, 1, self.note_fc.in_features), device=device)
                attention_mask = torch.zeros((B, 1), device=device).int()

        note_embed = self.note_fc(note_embed)  # (B,L,D)
        note_cls_tokens = self.note_cls_token.repeat(B, 1, 1)
        note_embed = note_embed + self.note_pos_embed[:, : note_embed.shape[1], :]

        if attention_mask.sum() != 0:
            note_embed = torch.cat((note_cls_tokens, note_embed), dim=1)  # (B,L+1,D)
            note_lengths = attention_mask.sum(dim=1) + avail_note
        else:
            note_embed = note_cls_tokens  # (B,1,D)
            note_lengths = torch.zeros_like(avail_note)

        note_pad_mask = length_to_mask(note_lengths, max_len=note_embed.shape[1])

        note_embed, note_pad_mask = self._apply_modality_state(
            note_embed, note_pad_mask, avail=avail_note, keep=keep_note, branch="note"
        )

        # -------- Multimodal fusion --------
        multimodal_cls_tokens = self.mm_cls_token
        for i in range(3):
            multimodal_cls_tokens = torch.cat((multimodal_cls_tokens, self.cross_cls_tokens[i].unsqueeze(0)), dim=1)
        multimodal_cls_tokens = multimodal_cls_tokens.repeat(B, 1, 1)  # (B,4,D)

        multimodal_embed = torch.cat((task_embed, multimodal_cls_tokens, ehr_embed, cxr_embed, note_embed), dim=1)

        # Build pad masks for task+cls tokens (not padded)
        task_pad_mask = torch.zeros((B, 1), dtype=torch.bool, device=device)
        cls_pad_mask  = torch.zeros((B, 4), dtype=torch.bool, device=device)

        multimodal_pad_mask = torch.cat((task_pad_mask, cls_pad_mask, ehr_pad_mask, cxr_pad_mask, note_pad_mask), dim=1)

        ehr_cls_index  = 5
        cxr_cls_index  = ehr_cls_index + ehr_embed.shape[1]
        note_cls_index = cxr_cls_index + cxr_embed.shape[1]

        cross_cls_mask = generate_cross_modal_mask(
            ehr_cls_index, cxr_cls_index, note_cls_index, multimodal_embed.shape[1]
        ).to(device)

        multimodal_embed = multimodal_embed.transpose(0, 1)  # (T,B,D)
        fusion_embed = self.transformer_fusion(
            multimodal_embed,
            mask=cross_cls_mask,
            src_key_padding_mask=multimodal_pad_mask,
        )
        fusion_embed = fusion_embed.transpose(0, 1)  # (B,T,D)

        # -------- Final representation --------
        task_mm_embed = fusion_embed[:, 0]  # (B,D)

        ehr_cls  = fusion_embed[:, ehr_cls_index]
        cxr_cls  = fusion_embed[:, cxr_cls_index]
        note_cls = fusion_embed[:, note_cls_index]
        modality_avg = torch.stack([ehr_cls, cxr_cls, note_cls], dim=1).mean(dim=1)
        final_mm_embed = torch.cat([task_mm_embed, modality_avg], dim=1)  # (B,2D)

        # -------- Heads --------
        if task_index[0] == 0:
            scores = torch.sigmoid(self.dense_layer_mortality(final_mm_embed))
        elif task_index[0] == 1:
            scores = torch.sigmoid(self.dense_layer_decomp(final_mm_embed))
        elif task_index[0] == 3:
            scores = self.dense_layer_los(final_mm_embed)
        elif task_index[0] == 4:
            scores = torch.sigmoid(self.dense_layer_readm(final_mm_embed))
        elif task_index[0] == 5:
            scores = torch.sigmoid(self.dense_layer_diag(final_mm_embed))
        elif task_index[0] == 6:
            scores = self.dense_layer_drg(final_mm_embed)
        else:
            scores = torch.sigmoid(self.dense_layer_ph(final_mm_embed))

        ortho_loss = calculate_ortho_loss(fusion_embed)

        if self.training:
            moe_loss = torch.tensor(0.0, device=device)
            return scores, ortho_loss, moe_loss
        else:
            return scores

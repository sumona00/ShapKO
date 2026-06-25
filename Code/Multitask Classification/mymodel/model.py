# mymodel/model.py
from __future__ import annotations
import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import length_to_mask
from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM
from mymodel.module import PatchEmbed, MoE, generate_cross_modal_mask

# -------------------------
# Global seed & HF setup
# -------------------------
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


# -------------------------
# Tokens decorrelation loss
# -------------------------
def calculate_ortho_loss(input_vec: torch.Tensor) -> torch.Tensor:
    """
    input_vec: (B, T, D)
    """
    x = input_vec - torch.mean(input_vec, axis=2, keepdim=True)
    cov_matrix = torch.matmul(x, x.transpose(1, 2)) / (x.shape[2] - 1)
    loss = (
        torch.sum(cov_matrix ** 2)
        - torch.sum(torch.diagonal(cov_matrix, dim1=1, dim2=2) ** 2)
    ) / (cov_matrix.shape[0] * (cov_matrix.shape[1] - 1) * (cov_matrix.shape[2] - 1))
    return loss


def temperature_scaled_softmax(logits: torch.Tensor, temperature: float = 1.0, dim: int = 0):
    logits = logits / temperature
    return torch.softmax(logits, dim=dim)


class FlexCare(nn.Module):
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
    ):
        super(FlexCare, self).__init__()

        self.device = device
        self.hidden_dim = hidden_dim
        self.task_embedding = nn.Embedding(40, hidden_dim)

        # -------------------------
        # Feature-space knockout
        # -------------------------
        # This is set externally from training script:
        #   model.feature_knockout_rate = r
        # where r = 1 - 0.5**(1/d)
        self.feature_knockout_rate: float = 0.0

        # Time series (EHR)
        self.ehr_projection = nn.Linear(ehr_dim, hidden_dim)
        self.ehr_cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.ehr_pos_embed = nn.Parameter(torch.zeros(1, 600, hidden_dim))

        # Image (CXR)
        self.patch_projection = PatchEmbed(patch_size=16, embed_dim=hidden_dim)
        num_patches = (224 // 16) * (224 // 16)
        self.cxr_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, hidden_dim))
        self.cxr_cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        # Text (Notes)
        self.note_projection = AutoModel.from_pretrained(HF_ID, cache_dir=CACHE_DIR).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(HF_ID, cache_dir=CACHE_DIR, use_fast=False)
        self.note_fc = nn.Linear(768, hidden_dim)
        self.note_cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.note_pos_embed = nn.Parameter(torch.zeros(1, 600, hidden_dim))

        # Modality fusion tokens
        self.cross_cls_tokens = nn.Parameter(torch.zeros(3, 1, hidden_dim))

        # Multimodal Transformer
        self.encoder_layer_fusion = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=2, dim_feedforward=hidden_dim * 4
        )
        self.transformer_fusion = nn.TransformerEncoder(
            self.encoder_layer_fusion, num_layers=layers
        )
        self.mm_cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        # Mixture of Experts
        self.moe = MoE(
            hidden_dim,
            hidden_dim,
            hidden_dim,
            expert_total,
            hidden_dim,
            noisy_gating=True,
            k=expert_k,
        )

        self.mm_choose = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)
        self.mm_choose2 = nn.Linear(hidden_dim, 1, bias=False)
        self.mm_layernorm = nn.LayerNorm(hidden_dim)

        # Task-specific heads
        self.dense_layer_mortality = nn.Linear(hidden_dim * 2, 1)
        self.dense_layer_decomp = nn.Linear(hidden_dim * 2, 1)
        self.dense_layer_ph = nn.Linear(hidden_dim * 2, 25)
        self.dense_layer_los = nn.Linear(hidden_dim * 2, 10)
        self.dense_layer_readm = nn.Linear(hidden_dim * 2, 1)
        self.dense_layer_diag = nn.Linear(hidden_dim * 2, 14)
        self.dense_layer_drg = nn.Linear(hidden_dim, 769)

    # -------------------------
    # Feature-space knockout
    # -------------------------
    def _feature_knockout(self, x: torch.Tensor) -> torch.Tensor:
        """
        Element-wise Bernoulli mask with probability p = feature_knockout_rate.
        Applied only during training. Like explicit dropout.

        x: (..., D) – works for (B,T,D) or (B,D).
        """
        p = float(getattr(self, "feature_knockout_rate", 0.0))
        if (not self.training) or p <= 0.0:
            return x
        if p >= 1.0:
            return torch.zeros_like(x)

        mask = (torch.rand_like(x) < p).float()
        # Rescale surviving features to keep expectation of activations
        return (x * (1.0 - mask)) / (1.0 - p)

    # -------------------------
    # Forward
    # -------------------------
    def forward(
        self,
        ehr,
        ehr_lengths,
        use_ehr,
        img,
        use_img,
        note,
        use_note,
        task_index,
    ):
        device = self.device
        B = ehr.size(0)

        # -------- Task embedding --------
        task_embed = self.task_embedding(task_index).unsqueeze(1)  # (B,1,D)

        # -------- EHR branch --------
        # ehr: (B, T, ehr_dim)
        ehr_embed = self.ehr_projection(ehr)  # (B,T,D)
        ehr_cls_tokens = self.ehr_cls_token.repeat(B, 1, 1)
        ehr_embed = ehr_embed + self.ehr_pos_embed[:, : ehr_embed.shape[1], :]
        ehr_embed = torch.cat((ehr_cls_tokens, ehr_embed), dim=1)  # (B, T+1, D)

        # Knockout in feature space
        ehr_embed = self._feature_knockout(ehr_embed)

        ehr_lengths = torch.tensor(ehr_lengths, device=device)
        # include CLS by adding use_ehr (keeps original semantics)
        ehr_lengths_with_cls = ehr_lengths + use_ehr
        ehr_pad_mask = length_to_mask(
            ehr_lengths_with_cls,
            max_len=ehr_embed.shape[1],  # match ehr_embed sequence length
        )  # (B, T_ehr+1)

        # -------- CXR branch --------
        cxr_embed = self.patch_projection(img)  # (B, P, D)
        cxr_cls_tokens = self.cxr_cls_token.repeat(B, 1, 1)
        cxr_embed = cxr_embed + self.cxr_pos_embed[:, : cxr_embed.shape[1], :]
        cxr_embed = torch.cat((cxr_cls_tokens, cxr_embed), dim=1)  # (B, P+1, D)

        # Knockout
        cxr_embed = self._feature_knockout(cxr_embed)

        # use_img: (B,)
        cxr_pad_mask = length_to_mask(use_img, max_len=1).repeat(1, cxr_embed.shape[1])
        # cxr_pad_mask: (B, P+1)

        # -------- Text (Notes) branch --------
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
                note_embed = outputs.last_hidden_state  # (B, L, 768)
            else:
                note_embed = torch.zeros(
                    (B, 1, self.note_fc.in_features), device=device
                )
                attention_mask = torch.zeros((B, 1), device=device).int()

        # Project to hidden_dim and add positions
        note_embed = self.note_fc(note_embed)  # (B, L, D)
        note_cls_tokens = self.note_cls_token.repeat(B, 1, 1)
        note_embed = note_embed + self.note_pos_embed[:, : note_embed.shape[1], :]

        if attention_mask.sum() != 0:
            # prepend CLS token
            note_embed = torch.cat((note_cls_tokens, note_embed), dim=1)  # (B, L+1, D)
            # length = #tokens + 1 CLS when note present
            note_lengths = attention_mask.sum(dim=1) + use_note
        else:
            # no text anywhere in the batch: just CLS
            note_embed = note_cls_tokens  # (B, 1, D)
            # treat as length 0 (CLS considered padding) to keep semantics
            note_lengths = torch.zeros_like(use_note)

        # Knockout
        note_embed = self._feature_knockout(note_embed)

        note_seq_len = note_embed.shape[1]
        note_pad_mask = length_to_mask(
            note_lengths,
            max_len=note_seq_len,  # match note_embed sequence length
        )  # (B, note_seq_len)

        # -------- Multimodal fusion --------
        multimodal_cls_tokens = self.mm_cls_token
        for i in range(3):
            multimodal_cls_tokens = torch.cat(
                (multimodal_cls_tokens, self.cross_cls_tokens[i].unsqueeze(0)), dim=1
            )
        multimodal_cls_tokens = multimodal_cls_tokens.repeat(B, 1, 1)  # (B,4,D)

        multimodal_embed = torch.cat(
            (task_embed, multimodal_cls_tokens, ehr_embed, cxr_embed, note_embed), dim=1
        )  # (B, T_total, D)

        # Build padding masks for task+cls tokens
        cls_pad_mask = length_to_mask(
            4 * torch.ones(use_img.shape, device=device), max_len=4
        )  # (B,4)
        task_pad_mask = length_to_mask(
            torch.ones(use_img.shape, device=device), max_len=1
        )  # (B,1)

        multimodal_pad_mask = torch.cat(
            (task_pad_mask, cls_pad_mask, ehr_pad_mask, cxr_pad_mask, note_pad_mask),
            dim=1,
        )  # (B, T_total)

        # ---- sanity check ----
        assert multimodal_embed.shape[1] == multimodal_pad_mask.shape[1], \
            f"mask width {multimodal_pad_mask.shape[1]} != seq_len {multimodal_embed.shape[1]}"

        ehr_cls_index = 5
        cxr_cls_index = ehr_cls_index + ehr_embed.shape[1]
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

        task_mm_embed = fusion_embed[:, 0]  # (B,D)
        mm_embed = torch.cat(
            (
                fusion_embed[:, 1:ehr_cls_index],
                fusion_embed[:, ehr_cls_index].unsqueeze(1),
                fusion_embed[:, cxr_cls_index].unsqueeze(1),
                fusion_embed[:, note_cls_index].unsqueeze(1),
            ),
            dim=1,
        )  # (B,7,D)

        # Mask for modality-combo tokens
        mm_mask = torch.ones(mm_embed.shape[0], mm_embed.shape[1], device=device)
        mm_mask[:, 0] = ehr_pad_mask[:, 0] | cxr_pad_mask[:, 0] | note_pad_mask[:, 0]
        mm_mask[:, 1] = ehr_pad_mask[:, 0] | cxr_pad_mask[:, 0]
        mm_mask[:, 2] = ehr_pad_mask[:, 0] | note_pad_mask[:, 0]
        mm_mask[:, 3] = cxr_pad_mask[:, 0] | note_pad_mask[:, 0]
        mm_mask[:, 4] = ehr_pad_mask[:, 0]
        mm_mask[:, 5] = cxr_pad_mask[:, 0]
        mm_mask[:, 6] = note_pad_mask[:, 0]

        mm_moe = torch.zeros(
            mm_embed.shape[0] * mm_embed.shape[1],
            mm_embed.shape[2],
            device=device,
        )
        cat_task_mm = task_mm_embed.unsqueeze(1).repeat(1, 7, 1)

        tmp_moe, moe_loss = self.moe(
            cat_task_mm.reshape(-1, cat_task_mm.shape[2])[mm_mask.reshape(-1) == 0],
            mm_embed.reshape(-1, mm_embed.shape[2])[mm_mask.reshape(-1) == 0],
        )
        mm_moe[mm_mask.reshape(-1) == 0] = tmp_moe
        mm_moe = mm_moe.reshape(mm_embed.shape[0], mm_embed.shape[1], mm_embed.shape[2])

        cat_task_mm = torch.cat(
            (task_mm_embed.unsqueeze(1).repeat(1, 7, 1), mm_moe), dim=2
        )
        weight = temperature_scaled_softmax(
            self.mm_choose2(torch.tanh(self.mm_choose(cat_task_mm))).squeeze(2)
            + (-mm_mask * 1e7),
            temperature=0.2,
            dim=1,
        )

        weighted_mm = (mm_moe * weight.unsqueeze(2)).sum(dim=1)
        final_mm_embed = torch.cat(
            (task_mm_embed, self.mm_layernorm(weighted_mm)), dim=1
        )

        # -------- Task-specific output --------
        if task_index[0] == 0:
            out = self.dense_layer_mortality(final_mm_embed)
            scores = torch.sigmoid(out)
        elif task_index[0] == 1:
            out = self.dense_layer_decomp(final_mm_embed)
            scores = torch.sigmoid(out)
        elif task_index[0] == 3:
            out = self.dense_layer_los(final_mm_embed)
            scores = out
        elif task_index[0] == 4:
            out = self.dense_layer_readm(final_mm_embed)
            scores = torch.sigmoid(out)
        elif task_index[0] == 5:
            out = self.dense_layer_diag(final_mm_embed)
            scores = torch.sigmoid(out)
        elif task_index[0] == 6:
            out = self.dense_layer_drg(final_mm_embed)
            scores = out
        else:
            out = self.dense_layer_ph(final_mm_embed)
            scores = torch.sigmoid(out)

        ortho_loss = calculate_ortho_loss(mm_embed)

        if self.training:
            return scores, ortho_loss, moe_loss
        else:
            return scores

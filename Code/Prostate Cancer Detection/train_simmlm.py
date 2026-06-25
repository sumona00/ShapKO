"""
SimMLM: Dynamic Mixture of Modality Experts (DMoME) + More-vs-Fewer (MoFe)
ranking loss for 3-modality binary classification.

Architecture:
  - Three modality experts E_m, each: ModalityNet -> linear -> 1 logit (o^m).
  - A small gating network G: takes the concatenation of per-modality features
    (from a separate lightweight CNN trunk per modality) and produces 3 gating
    values g^m. For missing modalities g^m is set to -inf so the corresponding
    softmax weight is 0. Final logit: o = sum_m w^m * o^m   (logit-level mix).

Two-stage training:
  1) Stage 1 (independent): each expert is trained independently using the
     other modalities zeroed (no gating, only that expert's logit is used).
  2) Stage 2 (cooperative): both experts and gating jointly trained with the
     full DMoME forward pass; for each sample we sample two views x+ ⊃ x-
     and apply the MoFe ranking loss in addition to the BCE on both views.

Final stage-2 loss for a (x+, x-) pair:
   L = BCE(o^+, y) + BCE(o^-, y) + lambda * max(0, BCE(o^+, y) - BCE(o^-, y))
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import argparse
import json
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from my_nnunet_cls.dataset import build_case_list_classification, Case3DClassificationDataset
from my_nnunet_cls.model import ModalityNet
from my_nnunet_cls.utils import safe_auc, safe_ap


# ============================================================
# DMoME architecture
# ============================================================
class GatingTrunk(nn.Module):
    """Tiny 3D CNN to produce a per-modality gating feature vector."""
    def __init__(self, in_channels: int = 1, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, 16, 3, stride=2, padding=1, bias=False),
            nn.InstanceNorm3d(16, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(16, 32, 3, stride=2, padding=1, bias=False),
            nn.InstanceNorm3d(32, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(32, out_dim, 3, stride=2, padding=1, bias=False),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.net(x)  # (B, out_dim)


class GatingNetwork(nn.Module):
    def __init__(self, n_modalities: int = 3, gate_feat: int = 64):
        super().__init__()
        self.trunks = nn.ModuleList([GatingTrunk(1, gate_feat) for _ in range(n_modalities)])
        self.head = nn.Linear(n_modalities * gate_feat, n_modalities)
        with torch.no_grad():
            # init to produce ~uniform gating
            self.head.weight.mul_(0.01)
            self.head.bias.zero_()

    def forward(self, xs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        feats = [trunk(x) for trunk, x in zip(self.trunks, xs)]
        f = torch.cat(feats, dim=1)
        return self.head(f)  # (B, n_modalities)


class SimMLMDMoME(nn.Module):
    """
    Modality experts: ModalityNet -> linear -> 1 logit per modality.
    Gating network: produces per-modality logit weights.
    """
    def __init__(self, base: int = 32, feat_dim: int = 256, dropout: float = 0.1, gate_feat: int = 64):
        super().__init__()
        self.expert_nets = nn.ModuleList([
            ModalityNet(in_channels=1, base=base, feat_dim=feat_dim, dropout=dropout)
            for _ in range(3)
        ])
        self.expert_heads = nn.ModuleList([
            nn.Linear(feat_dim, 1) for _ in range(3)
        ])
        self.gate = GatingNetwork(n_modalities=3, gate_feat=gate_feat)

    def expert_logit(self, x: torch.Tensor, m: int) -> torch.Tensor:
        z = self.expert_nets[m](x)
        return self.expert_heads[m](z)  # (B,1)

    def forward(self, xs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                presence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        xs:       3 modality tensors (B, 1, Z, Y, X)
        presence: (B, 3) float in {0,1} -- which modalities are present (used by gating)
        returns:  (combined_logit (B,1), expert_logits (B,3), weights (B,3))
        """
        # per-expert logits
        ologits = torch.cat([self.expert_logit(xs[m], m) for m in range(3)], dim=1)  # (B,3)
        # gating
        g = self.gate(xs)  # (B, 3)
        # mask missing -> -inf so softmax weights zero
        neg_inf = torch.full_like(g, float("-inf"))
        g = torch.where(presence > 0.5, g, neg_inf)
        w = F.softmax(g, dim=1)
        # if a sample has no modality, softmax of -inf is nan; guard
        w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        combined = (w * ologits).sum(dim=1, keepdim=True)  # (B,1)
        return combined, ologits, w


# ============================================================
# Modality presence + view sampling
# ============================================================
@torch.no_grad()
def presence_from_inputs(xs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
    flags = []
    for x in xs:
        dims = tuple(range(1, x.ndim))
        flags.append((x.abs().sum(dim=dims) > 0))
    return torch.stack(flags, dim=1)  # (B, M)


def sample_more_fewer_views(presence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    For each sample produce two binary masks (B,3): plus ⊇ minus, plus ⊆ presence.
    plus  = random non-empty subset of presence (often equals presence)
    minus = random non-empty proper subset of plus, or equals plus if |plus|==1.
    """
    B, M = presence.shape
    plus = torch.zeros_like(presence, dtype=torch.bool)
    minus = torch.zeros_like(presence, dtype=torch.bool)
    for b in range(B):
        present_idx = [i for i in range(M) if presence[b, i].item()]
        n_p = len(present_idx)
        if n_p == 0:
            continue
        # plus: random non-empty subset (bias toward larger sets)
        # 50% chance use full presence; else random non-empty subset
        if torch.rand(1).item() < 0.5 or n_p == 1:
            plus_set = list(present_idx)
        else:
            n_keep = int(torch.randint(1, n_p + 1, (1,)).item())
            perm = torch.randperm(n_p)[:n_keep].tolist()
            plus_set = [present_idx[j] for j in perm]
        for i in plus_set:
            plus[b, i] = True
        # minus: strict non-empty proper subset of plus, or equal if |plus|==1
        if len(plus_set) <= 1:
            for i in plus_set:
                minus[b, i] = True
        else:
            n_keep = int(torch.randint(1, len(plus_set), (1,)).item())
            perm = torch.randperm(len(plus_set))[:n_keep].tolist()
            for j in perm:
                minus[b, plus_set[j]] = True
    return plus, minus


def apply_mask_to_inputs(
    xs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    out = []
    for m, x in enumerate(xs):
        mm = mask[:, m].to(dtype=x.dtype, device=x.device)
        view = (x.shape[0],) + (1,) * (x.ndim - 1)
        out.append(x * mm.view(view))
    return tuple(out)  # type: ignore


# ============================================================
# Helpers
# ============================================================
def compute_pos_weight(cases) -> float:
    ys = np.array([c["y"] for c in cases], dtype=np.int64)
    pos = float((ys == 1).sum())
    neg = float((ys == 0).sum())
    return 1.0 if pos < 1 else neg / pos


def _load_fold_split(splits_json: Path, fold: int) -> Dict[str, List[str]]:
    splits = json.load(open(splits_json, "r"))
    fold_split = splits[fold] if isinstance(splits, list) else splits[str(fold) if str(fold) in splits else fold]
    return {"train": [str(x).strip() for x in fold_split["train"]],
            "val":   [str(x).strip() for x in fold_split["val"]]}


# ============================================================
# Eval (uses the full DMoME forward, with presence as gating mask)
# ============================================================
@torch.no_grad()
def eval_one_epoch(model: SimMLMDMoME, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    bce = nn.BCEWithLogitsLoss(reduction="mean")
    y_true: List[int] = []
    y_score: List[float] = []
    loss_sum = 0.0; n = 0
    pbar = tqdm(loader, desc="[VAL]", leave=False)
    for (x0, x1, x2), y, _sid in pbar:
        x0 = x0.to(device); x1 = x1.to(device); x2 = x2.to(device); y = y.to(device)
        pres = presence_from_inputs((x0, x1, x2)).float().to(device)
        logit, _, _ = model((x0, x1, x2), pres)
        loss = bce(logit, y)
        y_score.extend(torch.sigmoid(logit).view(-1).cpu().numpy().tolist())
        y_true.extend(y.view(-1).cpu().numpy().astype(int).tolist())
        loss_sum += float(loss.item()); n += 1
        pbar.set_postfix(loss=f"{loss_sum/max(1,n):.4f}")
    return {"loss": loss_sum/max(1,n), "auc": safe_auc(y_true, y_score), "ap": safe_ap(y_true, y_score)}


# ============================================================
# Stage 1: independent expert pretraining
# ============================================================
def stage1_pretrain_experts(
    model: SimMLMDMoME,
    train_ld: DataLoader,
    val_ld: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    pos_w: torch.Tensor,
) -> Dict[str, Any]:
    """Trains each expert (modality net + head) independently using only that modality."""
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_w, reduction="mean")
    history: List[Dict[str, Any]] = []
    print(f"[SimMLM][stage1] pretraining 3 experts independently for {epochs} epochs each")

    for m in range(3):
        # freeze all params, then unfreeze only expert m (net + head)
        for p in model.parameters():
            p.requires_grad_(False)
        for p in model.expert_nets[m].parameters():
            p.requires_grad_(True)
        for p in model.expert_heads[m].parameters():
            p.requires_grad_(True)

        params = list(model.expert_nets[m].parameters()) + list(model.expert_heads[m].parameters())
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)

        for ep in range(1, epochs + 1):
            model.train()
            loss_sum = 0.0; n = 0
            pbar = tqdm(train_ld, desc=f"[stage1][m{m}] ep{ep}/{epochs}", leave=False)
            for (x0, x1, x2), y, _sid in pbar:
                x0 = x0.to(device); x1 = x1.to(device); x2 = x2.to(device); y = y.to(device)
                xs = [x0, x1, x2]
                # presence check for that modality only
                xm = xs[m]
                dims = tuple(range(1, xm.ndim))
                pres_m = (xm.abs().sum(dim=dims) > 0)
                if pres_m.sum().item() == 0:
                    continue
                logit_m = model.expert_logit(xm, m)
                # only count loss on samples where modality m is actually present
                lp = bce(logit_m[pres_m], y[pres_m])
                opt.zero_grad(set_to_none=True)
                lp.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                loss_sum += float(lp.item()); n += 1
                pbar.set_postfix(loss=f"{loss_sum/max(1,n):.4f}")
            print(f"[stage1][m{m}] ep{ep} loss={loss_sum/max(1,n):.4f}")
            history.append({"stage": 1, "modality": m, "epoch": ep, "loss": loss_sum/max(1,n)})

    # unfreeze everything
    for p in model.parameters():
        p.requires_grad_(True)
    return {"history": history}


# ============================================================
# Stage 2: cooperative training with MoFe loss
# ============================================================
def stage2_cooperative_train(
    model: SimMLMDMoME,
    train_ld: DataLoader,
    val_ld: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    pos_w: torch.Tensor,
    lam_mofe: float,
    save_best_path: Path,
    save_last_path: Path,
    history_path: Path,
    fold: int,
    prev_history: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[float, List[Dict[str, Any]]]:
    bce_mean = nn.BCEWithLogitsLoss(pos_weight=pos_w, reduction="mean")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    history = list(prev_history or [])
    best_auc = -1.0
    print(f"[SimMLM][stage2] cooperative training (MoFe lambda={lam_mofe}) for {epochs} epochs")

    for ep in range(1, epochs + 1):
        model.train()
        loss_sum = bcep_sum = bcem_sum = mofe_sum = 0.0
        n = 0
        pbar = tqdm(train_ld, desc=f"[stage2] f{fold} ep{ep}/{epochs}", leave=True)
        for (x0, x1, x2), y, _sid in pbar:
            x0 = x0.to(device); x1 = x1.to(device); x2 = x2.to(device); y = y.to(device)
            pres = presence_from_inputs((x0, x1, x2))
            plus_b, minus_b = sample_more_fewer_views(pres)
            plus = plus_b.float().to(device)
            minus = minus_b.float().to(device)

            xs_plus = apply_mask_to_inputs((x0, x1, x2), plus)
            xs_minus = apply_mask_to_inputs((x0, x1, x2), minus)

            logit_plus, _, _ = model(xs_plus, plus)
            logit_minus, _, _ = model(xs_minus, minus)

            l_plus = bce_mean(logit_plus, y)
            l_minus = bce_mean(logit_minus, y)
            l_mofe = torch.clamp(l_plus - l_minus, min=0.0)

            loss = l_plus + l_minus + lam_mofe * l_mofe

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            loss_sum += float(loss.item())
            bcep_sum += float(l_plus.item())
            bcem_sum += float(l_minus.item())
            mofe_sum += float(l_mofe.item())
            n += 1
            pbar.set_postfix(loss=f"{loss_sum/n:.4f}",
                             pl=f"{bcep_sum/n:.3f}", mn=f"{bcem_sum/n:.3f}",
                             mofe=f"{mofe_sum/n:.4f}")

        va = eval_one_epoch(model, val_ld, device)
        row = {"stage": 2, "epoch": ep, "train_loss": loss_sum/max(1,n),
               "train_lplus": bcep_sum/max(1,n), "train_lminus": bcem_sum/max(1,n),
               "train_lmofe": mofe_sum/max(1,n), "val": va}
        history.append(row)
        print(f"[EP{ep:03d}] loss={row['train_loss']:.4f} | "
              f"val auc={va['auc']:.4f} ap={va['ap']:.4f}")

        torch.save({"model": model.state_dict(), "epoch": ep, "fold": fold,
                    "val": va, "method": "simmlm"}, save_last_path)
        if not math.isnan(va["auc"]) and va["auc"] > best_auc:
            best_auc = va["auc"]
            torch.save({"model": model.state_dict(), "epoch": ep, "fold": fold,
                        "best_auc": best_auc, "val": va, "method": "simmlm"},
                       save_best_path)
            print(f"  ✅ New best: {save_best_path} (auc={best_auc:.4f})")
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

    return best_auc, history


# ============================================================
# Train fold (full pipeline)
# ============================================================
def train_fold(
    images_root: Path,
    labels_csv: Path,
    splits_json: Path,
    outdir: Path,
    fold: int = 0,
    epochs_stage1: int = 15,
    epochs_stage2: int = 50,
    batch_size: int = 2,
    lr: float = 3e-4,
    num_workers: int = 4,
    target_shape=(128, 192, 192),
    base: int = 32,
    feat_dim: int = 256,
    dropout: float = 0.1,
    gate_feat: int = 64,
    lam_mofe: float = 0.1,
) -> Dict[str, Any]:
    outdir.mkdir(parents=True, exist_ok=True)
    fold_dir = outdir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = fold_dir / "best.pt"
    last_ckpt_path = fold_dir / "last.pt"
    stage1_ckpt_path = fold_dir / "after_stage1.pt"
    history_path = fold_dir / "history.json"

    all_cases = build_case_list_classification(
        images_root=images_root, labels_csv=labels_csv, strict=True, verbose=True,
    )
    id_to_case = {str(c["sid"]).strip(): c for c in all_cases}
    sp = _load_fold_split(splits_json, fold)
    train_cases = [id_to_case[s] for s in sp["train"] if s in id_to_case]
    val_cases = [id_to_case[s] for s in sp["val"] if s in id_to_case]

    print(f"\n========== SimMLM Fold {fold} ==========")
    print(f"All={len(all_cases)} | train={len(train_cases)} val={len(val_cases)}")
    if len(train_cases) == 0 or len(val_cases) == 0:
        raise RuntimeError("Empty split after matching.")

    train_ds = Case3DClassificationDataset(
        train_cases, target_shape=target_shape, normalize=True, align_to_ref=True, return_sid=True
    )
    val_ds = Case3DClassificationDataset(
        val_cases, target_shape=target_shape, normalize=True, align_to_ref=True, return_sid=True
    )
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True, drop_last=False)
    val_ld = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimMLMDMoME(base=base, feat_dim=feat_dim, dropout=dropout, gate_feat=gate_feat).to(device)

    pos_w = torch.tensor([compute_pos_weight(train_cases)], dtype=torch.float32, device=device)

    # Stage 1
    s1 = stage1_pretrain_experts(model, train_ld, val_ld, device,
                                 epochs=epochs_stage1, lr=lr, pos_w=pos_w)
    torch.save({"model": model.state_dict(), "fold": fold, "method": "simmlm",
                "stage": 1}, stage1_ckpt_path)
    print(f"  saved after-stage1 ckpt: {stage1_ckpt_path}")

    # Stage 2
    best_auc, history = stage2_cooperative_train(
        model, train_ld, val_ld, device,
        epochs=epochs_stage2, lr=lr, pos_w=pos_w, lam_mofe=lam_mofe,
        save_best_path=best_ckpt_path, save_last_path=last_ckpt_path,
        history_path=history_path, fold=fold,
        prev_history=s1["history"],
    )

    return {"fold": fold, "best_auc": best_auc, "best_ckpt": str(best_ckpt_path),
            "history": str(history_path)}


# ============================================================
# CLI
# ============================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True)
    p.add_argument("--labels_csv", required=True)
    p.add_argument("--splits", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--epochs_stage1", type=int, default=15)
    p.add_argument("--epochs_stage2", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--tZ", type=int, default=128)
    p.add_argument("--tY", type=int, default=192)
    p.add_argument("--tX", type=int, default=192)
    p.add_argument("--base", type=int, default=32)
    p.add_argument("--feat_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--gate_feat", type=int, default=64)
    p.add_argument("--lam_mofe", type=float, default=0.1)
    return p.parse_args()


def main():
    a = parse_args()
    train_fold(
        images_root=Path(a.images), labels_csv=Path(a.labels_csv), splits_json=Path(a.splits),
        outdir=Path(a.outdir), fold=int(a.fold),
        epochs_stage1=int(a.epochs_stage1), epochs_stage2=int(a.epochs_stage2),
        batch_size=int(a.batch_size), lr=float(a.lr), num_workers=int(a.num_workers),
        target_shape=(a.tZ, a.tY, a.tX),
        base=int(a.base), feat_dim=int(a.feat_dim), dropout=float(a.dropout),
        gate_feat=int(a.gate_feat), lam_mofe=float(a.lam_mofe),
    )


if __name__ == "__main__":
    main()

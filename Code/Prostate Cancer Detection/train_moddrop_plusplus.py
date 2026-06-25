"""
ModDrop++: Dynamic Filter Network with Intra-subject Co-training.

Implementation for 3-modality binary classification (T2W/ADC/HBV).

Key components:
  - DynamicHead: MLP that maps a binary modality code m in {0,1}^3 to per-encoder
    filter scaling vectors (one scalar per output channel of the first conv layer).
  - ModDropPPModalityNet: ModalityNet whose first conv is externally scaled.
  - ModDropPPClassifier: 3 modality nets + dynamic head + fusion -> logit.
  - Intra-subject co-training: each step does 2 forward passes
        (1) "full":     m_full   = observed presence
        (2) "missing":  m_miss   = strict random subset of m_full
    Loss = alpha*BCE(logit_full,y) + beta*BCE(logit_miss,y) + gamma*MSE(fm_full, fm_miss)
    (MSE used in place of SSIM as a feature-similarity loss for 3D feature maps.)

Observed-missing modalities (encoded as all-zero tensors by the dataset) are NOT
randomly dropped by this method; only present modalities can be dropped to form
the "missing" view.
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
from my_nnunet_cls.model import conv_block
from my_nnunet_cls.utils import safe_auc, safe_ap


# ============================================================
# ModDrop++ model
# ============================================================
class DynamicHead(nn.Module):
    """Maps modality code (B,3) -> per-encoder per-channel scaling (B, 3, base)."""
    def __init__(self, n_modalities: int = 3, base: int = 32, hidden: int = 64):
        super().__init__()
        self.n = n_modalities
        self.base = base
        self.mlp = nn.Sequential(
            nn.Linear(n_modalities, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_modalities * base),
        )
        # init last layer near zero so initial scale ~= 1
        with torch.no_grad():
            self.mlp[-1].weight.mul_(0.01)
            self.mlp[-1].bias.zero_()

    def forward(self, mcode: torch.Tensor) -> torch.Tensor:
        B = mcode.shape[0]
        delta = self.mlp(mcode.float()).view(B, self.n, self.base)
        # final scale = 1 + delta (so default behavior = identity)
        return 1.0 + delta


class ModDropPPModalityNet(nn.Module):
    """
    Encoder where the first conv's output channels are scaled by an externally
    supplied per-channel scale vector. Same trunk as ModalityNet otherwise.
    """
    def __init__(self, in_channels: int = 1, base: int = 32, feat_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.base = base
        # First conv extracted (so scaling can be applied to its output)
        self.conv1 = nn.Conv3d(in_channels, base, 3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(base, affine=True)
        self.act1 = nn.LeakyReLU(0.01, inplace=True)
        self.conv1b = nn.Conv3d(base, base, 3, padding=1, bias=False)
        self.norm1b = nn.InstanceNorm3d(base, affine=True)
        self.act1b = nn.LeakyReLU(0.01, inplace=True)

        self.p1 = nn.MaxPool3d(2)
        self.e2 = conv_block(base, base * 2)
        self.p2 = nn.MaxPool3d(2)
        self.e3 = conv_block(base * 2, base * 4)
        self.p3 = nn.MaxPool3d(2)
        self.b = conv_block(base * 4, base * 8)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base * 8, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def first_conv_block(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        # x: (B,1,Z,Y,X); scale: (B,base)
        out = self.conv1(x)
        out = out * scale.view(scale.shape[0], scale.shape[1], 1, 1, 1)
        out = self.act1(self.norm1(out))
        out = self.act1b(self.norm1b(self.conv1b(out)))
        return out

    def forward(self, x: torch.Tensor, scale: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        fm = self.first_conv_block(x, scale)
        h = self.e2(self.p1(fm))
        h = self.e3(self.p2(h))
        h = self.b(self.p3(h))
        z = self.head(self.pool(h))
        return z, fm


class ModDropPPClassifier(nn.Module):
    def __init__(self, base: int = 32, feat_dim: int = 256, dropout: float = 0.1, fusion_hidden: int = 256):
        super().__init__()
        self.net0 = ModDropPPModalityNet(1, base, feat_dim, dropout)
        self.net1 = ModDropPPModalityNet(1, base, feat_dim, dropout)
        self.net2 = ModDropPPModalityNet(1, base, feat_dim, dropout)
        self.dyn_head = DynamicHead(n_modalities=3, base=base, hidden=64)
        self.fusion = nn.Sequential(
            nn.Linear(3 * feat_dim, fusion_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, 1),
        )

    def forward(self, x_tuple: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                mcode: torch.Tensor):
        x0, x1, x2 = x_tuple
        scales = self.dyn_head(mcode)  # (B, 3, base)
        z0, fm0 = self.net0(x0, scales[:, 0])
        z1, fm1 = self.net1(x1, scales[:, 1])
        z2, fm2 = self.net2(x2, scales[:, 2])
        logit = self.fusion(torch.cat([z0, z1, z2], dim=1))
        return logit, (fm0, fm1, fm2)


# ============================================================
# Modality presence + missing-config sampling
# ============================================================
@torch.no_grad()
def presence_from_inputs(xs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
    """(B,3) presence mask: True if non-zero anywhere."""
    flags = []
    for x in xs:
        dims = tuple(range(1, x.ndim))
        flags.append((x.abs().sum(dim=dims) > 0))
    return torch.stack(flags, dim=1)  # (B, M)


def sample_missing_mcode(presence: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """
    For each sample, sample a strict non-empty proper subset of present modalities.
    If only 1 modality is present, returns presence unchanged for that sample.
    """
    B, M = presence.shape
    out = presence.clone()
    for b in range(B):
        present_idx = [i for i in range(M) if presence[b, i].item()]
        if len(present_idx) <= 1:
            continue
        # number of modalities to drop in [1, len(present_idx)-1]
        n_drop = int(torch.randint(1, len(present_idx), (1,), generator=generator).item())
        perm = torch.randperm(len(present_idx), generator=generator)[:n_drop].tolist()
        for j in perm:
            out[b, present_idx[j]] = False
    return out


def apply_mcode_to_inputs(
    xs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    mcode: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    out = []
    for m, x in enumerate(xs):
        mask = mcode[:, m].to(dtype=x.dtype, device=x.device)
        view = (x.shape[0],) + (1,) * (x.ndim - 1)
        out.append(x * mask.view(view))
    return tuple(out)  # type: ignore


# ============================================================
# Helpers (split, pos weight)
# ============================================================
def compute_pos_weight(cases) -> float:
    ys = np.array([c["y"] for c in cases], dtype=np.int64)
    pos = float((ys == 1).sum())
    neg = float((ys == 0).sum())
    return 1.0 if pos < 1 else neg / pos


def _load_fold_split(splits_json: Path, fold: int) -> Dict[str, List[str]]:
    splits = json.load(open(splits_json, "r"))
    fold_split = splits[fold] if isinstance(splits, list) else splits[str(fold) if str(fold) in splits else fold]
    train_ids = [str(x).strip() for x in fold_split["train"]]
    val_ids = [str(x).strip() for x in fold_split["val"]]
    return {"train": train_ids, "val": val_ids}


# ============================================================
# Eval (single forward, mcode = presence, no second branch)
# ============================================================
@torch.no_grad()
def eval_one_epoch(model: ModDropPPClassifier, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    bce = nn.BCEWithLogitsLoss(reduction="mean")
    y_true: List[int] = []
    y_score: List[float] = []
    loss_sum = 0.0
    n = 0

    pbar = tqdm(loader, desc="[VAL]", leave=False)
    for (x0, x1, x2), y, _sid in pbar:
        x0 = x0.to(device, non_blocking=True)
        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        pres = presence_from_inputs((x0, x1, x2)).float().to(device)
        logit, _ = model((x0, x1, x2), pres)

        loss = bce(logit, y)
        prob = torch.sigmoid(logit).view(-1).detach().cpu().numpy().tolist()
        yt = y.view(-1).detach().cpu().numpy().astype(int).tolist()
        y_score.extend(prob); y_true.extend(yt)
        loss_sum += float(loss.item()); n += 1
        pbar.set_postfix(loss=f"{loss_sum/max(1,n):.4f}")

    return {"loss": loss_sum / max(1, n), "auc": safe_auc(y_true, y_score), "ap": safe_ap(y_true, y_score)}


# ============================================================
# Train
# ============================================================
def train_fold(
    images_root: Path,
    labels_csv: Path,
    splits_json: Path,
    outdir: Path,
    fold: int = 0,
    epochs: int = 50,
    batch_size: int = 2,
    lr: float = 3e-4,
    num_workers: int = 4,
    target_shape=(128, 192, 192),
    base: int = 32,
    feat_dim: int = 256,
    dropout: float = 0.1,
    fusion_hidden: int = 256,
    alpha: float = 1.0,        # weight on BCE(full)
    beta: float = 1.0,         # weight on BCE(missing)
    gamma: float = 0.05,       # weight on feature-similarity loss
) -> Dict[str, Any]:
    outdir.mkdir(parents=True, exist_ok=True)
    fold_dir = outdir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    best_ckpt_path = fold_dir / "best.pt"
    last_ckpt_path = fold_dir / "last.pt"
    history_path = fold_dir / "history.json"

    all_cases = build_case_list_classification(
        images_root=images_root, labels_csv=labels_csv, strict=True, verbose=True,
    )
    id_to_case = {str(c["sid"]).strip(): c for c in all_cases}
    sp = _load_fold_split(splits_json, fold)
    train_cases = [id_to_case[s] for s in sp["train"] if s in id_to_case]
    val_cases = [id_to_case[s] for s in sp["val"] if s in id_to_case]

    print(f"\n========== ModDrop++ Fold {fold} ==========")
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
    model = ModDropPPClassifier(
        base=base, feat_dim=feat_dim, dropout=dropout, fusion_hidden=fusion_hidden,
    ).to(device)

    pos_w = torch.tensor([compute_pos_weight(train_cases)], dtype=torch.float32, device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_w, reduction="mean")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_auc = -1.0
    history: List[Dict[str, Any]] = []

    print(f"[ModDrop++] alpha={alpha} beta={beta} gamma={gamma}")
    for ep in range(1, epochs + 1):
        model.train()
        loss_sum = lt_full_sum = lt_miss_sum = lsim_sum = 0.0
        n = 0
        pbar = tqdm(train_ld, desc=f"[TRAIN][MD++] f{fold} ep{ep}/{epochs}", leave=True)
        for (x0, x1, x2), y, _sid in pbar:
            x0 = x0.to(device, non_blocking=True)
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            pres = presence_from_inputs((x0, x1, x2))           # (B,3) bool
            mcode_full = pres.float().to(device)                # full = observed presence
            # missing view: strictly drop a subset of present modalities
            mcode_miss_bool = sample_missing_mcode(pres)
            mcode_miss = mcode_miss_bool.float().to(device)

            xs_full = apply_mcode_to_inputs((x0, x1, x2), mcode_full)
            xs_miss = apply_mcode_to_inputs((x0, x1, x2), mcode_miss)

            logit_full, fms_full = model(xs_full, mcode_full)
            logit_miss, fms_miss = model(xs_miss, mcode_miss)

            l_task_full = bce(logit_full, y)
            l_task_miss = bce(logit_miss, y)

            # feature-similarity loss: MSE between first-conv feature maps,
            # only on positions where the modality is present in the missing view
            # (otherwise the missing-view fm is the conv of zeros and shouldn't
            # be forced to match the full one).
            l_sim = x0.new_tensor(0.0)
            n_terms = 0
            for m in range(3):
                # only compare for samples where mcode_miss[:,m] == 1 (modality kept in miss)
                keep_mask = mcode_miss[:, m].view(-1, 1, 1, 1, 1)
                diff = (fms_full[m] - fms_miss[m]) ** 2
                # weighted mean: sum over kept samples; element-wise mean over (C,Z,Y,X)
                num = (diff * keep_mask).flatten(1).sum(dim=1)
                denom = (keep_mask.flatten(1).sum(dim=1) * float(diff.shape[1] * diff.shape[2] * diff.shape[3] * diff.shape[4]) + 1e-8)
                l_sim_m = (num / denom).mean()
                l_sim = l_sim + l_sim_m
                n_terms += 1
            if n_terms > 0:
                l_sim = l_sim / n_terms

            loss = alpha * l_task_full + beta * l_task_miss + gamma * l_sim

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            loss_sum += float(loss.item())
            lt_full_sum += float(l_task_full.item())
            lt_miss_sum += float(l_task_miss.item())
            lsim_sum += float(l_sim.item())
            n += 1
            pbar.set_postfix(loss=f"{loss_sum/n:.4f}",
                             tf=f"{lt_full_sum/n:.3f}", tm=f"{lt_miss_sum/n:.3f}",
                             sim=f"{lsim_sum/n:.4f}")

        va = eval_one_epoch(model, val_ld, device)
        row = {"epoch": ep, "train_loss": loss_sum / max(1, n),
               "train_lfull": lt_full_sum / max(1, n),
               "train_lmiss": lt_miss_sum / max(1, n),
               "train_lsim": lsim_sum / max(1, n),
               "val": va}
        history.append(row)
        print(f"[EP{ep:03d}] loss={row['train_loss']:.4f} | "
              f"val auc={va['auc']:.4f} ap={va['ap']:.4f}")

        torch.save({"model": model.state_dict(), "epoch": ep, "fold": fold,
                    "val": va, "method": "moddrop++"}, last_ckpt_path)

        if not math.isnan(va["auc"]) and va["auc"] > best_auc:
            best_auc = va["auc"]
            torch.save({"model": model.state_dict(), "epoch": ep, "fold": fold,
                        "best_auc": best_auc, "val": va, "method": "moddrop++"},
                       best_ckpt_path)
            print(f"  ✅ New best: {best_ckpt_path} (auc={best_auc:.4f})")

        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

    return {"fold": fold, "best_auc": best_auc, "best_ckpt": str(best_ckpt_path), "history": str(history_path)}


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
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--tZ", type=int, default=128)
    p.add_argument("--tY", type=int, default=192)
    p.add_argument("--tX", type=int, default=192)
    p.add_argument("--base", type=int, default=32)
    p.add_argument("--feat_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--fusion_hidden", type=int, default=256)
    p.add_argument("--alpha", type=float, default=1.0, help="Weight on BCE(full).")
    p.add_argument("--beta", type=float, default=1.0, help="Weight on BCE(missing).")
    p.add_argument("--gamma", type=float, default=0.05, help="Weight on feature-similarity loss.")
    return p.parse_args()


def main():
    a = parse_args()
    train_fold(
        images_root=Path(a.images), labels_csv=Path(a.labels_csv), splits_json=Path(a.splits),
        outdir=Path(a.outdir), fold=int(a.fold), epochs=int(a.epochs),
        batch_size=int(a.batch_size), lr=float(a.lr), num_workers=int(a.num_workers),
        target_shape=(a.tZ, a.tY, a.tX),
        base=int(a.base), feat_dim=int(a.feat_dim), dropout=float(a.dropout),
        fusion_hidden=int(a.fusion_hidden),
        alpha=float(a.alpha), beta=float(a.beta), gamma=float(a.gamma),
    )


if __name__ == "__main__":
    main()

"""
PMR: Prototypical Modal Rebalance for Multimodal Learning.

Adapted to 3-modality binary classification (T2W/ADC/HBV).

This method does NOT drop modalities at training time; instead, it rebalances
each modality's contribution by:
  1) Maintaining per-modality, per-class prototypes c_k^m in feature space
     (computed each epoch on a subset of training samples).
  2) Computing a per-modality "task-correctness score" s_m on the fly using
     softmax over negative distances to prototypes.
  3) Adding a Prototypical CE (PCE) loss for slow-learning modalities, scaled
     by a per-modality coefficient beta_m derived from gap to the dominant
     modality.
  4) Adding a Prototypical Entropy Regularization (PER) on the dominant
     modality during early epochs, to prevent its premature convergence.

Final loss:
   L = L_CE(logit_fused, y)
       + alpha * sum_m beta_m * L_PCE^m
       - mu * gamma_dom * H(p^dom)               (only first Er epochs)

Note: prototypes are computed only from samples with all 3 modalities present;
samples with observed-missing modalities still contribute to the BCE loss but
NOT to prototype/PCE/PER terms (their per-modality features for missing
modalities would be encoded zeros and meaningless).
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
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from my_nnunet_cls.dataset import build_case_list_classification, Case3DClassificationDataset
from my_nnunet_cls.model import ThreeModalityClassifier
from my_nnunet_cls.utils import safe_auc, safe_ap


# ============================================================
# Wrapper around ThreeModalityClassifier exposing per-modality features
# ============================================================
class PMRClassifier(nn.Module):
    """Same as ThreeModalityClassifier, but forward returns (logit, [z0,z1,z2])."""
    def __init__(self, base: int = 32, feat_dim: int = 256, dropout: float = 0.1, fusion_hidden: int = 256):
        super().__init__()
        self.base_model = ThreeModalityClassifier(
            base=base, feat_dim=feat_dim, dropout=dropout, fusion_hidden=fusion_hidden,
        )
        self.feat_dim = feat_dim

    def encode(self, x_tuple):
        x0, x1, x2 = x_tuple
        z0 = self.base_model.net0(x0)
        z1 = self.base_model.net1(x1)
        z2 = self.base_model.net2(x2)
        return z0, z1, z2

    def fuse(self, z0, z1, z2):
        return self.base_model.fusion(torch.cat([z0, z1, z2], dim=1))

    def forward(self, x_tuple):
        z0, z1, z2 = self.encode(x_tuple)
        logit = self.fuse(z0, z1, z2)
        return logit, (z0, z1, z2)


# ============================================================
# Prototype computation
# ============================================================
@torch.no_grad()
def compute_prototypes(
    model: PMRClassifier,
    proto_loader: DataLoader,
    device: torch.device,
    n_classes: int = 2,
    feat_dim: int = 256,
    n_modalities: int = 3,
    prev_protos: Optional[List[torch.Tensor]] = None,
    momentum: float = 0.5,
) -> List[torch.Tensor]:
    """
    Returns list of n_modalities tensors, each (n_classes, feat_dim).
    Uses ONLY fully-present samples (all 3 modalities present).
    """
    model.eval()
    sums = [torch.zeros(n_classes, feat_dim, device=device) for _ in range(n_modalities)]
    counts = [torch.zeros(n_classes, device=device) for _ in range(n_modalities)]

    for (x0, x1, x2), y, _sid in proto_loader:
        x0 = x0.to(device, non_blocking=True)
        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).view(-1).long()

        # presence per sample
        def pres(x):
            dims = tuple(range(1, x.ndim))
            return (x.abs().sum(dim=dims) > 0)
        keep = pres(x0) & pres(x1) & pres(x2)
        if keep.sum().item() == 0:
            continue

        z0, z1, z2 = model.encode((x0, x1, x2))
        zs = [z0, z1, z2]
        yk = y[keep]
        for m in range(n_modalities):
            zk = zs[m][keep]  # (Nk, feat_dim)
            for c in range(n_classes):
                mask = (yk == c)
                if mask.any():
                    sums[m][c] += zk[mask].sum(dim=0)
                    counts[m][c] += mask.sum().float()

    new_protos = []
    for m in range(n_modalities):
        proto = torch.zeros(n_classes, feat_dim, device=device)
        for c in range(n_classes):
            cnt = counts[m][c].clamp(min=1.0)
            proto[c] = sums[m][c] / cnt
        new_protos.append(proto)

    if prev_protos is not None:
        new_protos = [momentum * p_old + (1.0 - momentum) * p_new
                      for p_old, p_new in zip(prev_protos, new_protos)]
    return new_protos


def prototype_logits(z: torch.Tensor, proto: torch.Tensor) -> torch.Tensor:
    """
    z:     (B, D)
    proto: (K, D)
    Returns logits = -dist(z, c_k) (B, K), where dist = squared Euclidean / D
    (normalized for numerical stability).
    """
    # squared Euclidean
    d = ((z.unsqueeze(1) - proto.unsqueeze(0)) ** 2).sum(dim=-1)  # (B, K)
    return -d / max(1, z.shape[-1])


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
# Eval (uses fused logit only)
# ============================================================
@torch.no_grad()
def eval_one_epoch(model: PMRClassifier, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    bce = nn.BCEWithLogitsLoss(reduction="mean")
    y_true: List[int] = []
    y_score: List[float] = []
    loss_sum = 0.0; n = 0
    pbar = tqdm(loader, desc="[VAL]", leave=False)
    for (x0, x1, x2), y, _sid in pbar:
        x0 = x0.to(device); x1 = x1.to(device); x2 = x2.to(device); y = y.to(device)
        logit, _ = model((x0, x1, x2))
        loss = bce(logit, y)
        y_score.extend(torch.sigmoid(logit).view(-1).cpu().numpy().tolist())
        y_true.extend(y.view(-1).cpu().numpy().astype(int).tolist())
        loss_sum += float(loss.item()); n += 1
        pbar.set_postfix(loss=f"{loss_sum/max(1,n):.4f}")
    return {"loss": loss_sum/max(1,n), "auc": safe_auc(y_true, y_score), "ap": safe_ap(y_true, y_score)}


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
    alpha: float = 1.0,             # PCE weight
    mu: float = 0.01,               # PER weight
    er_epochs: int = 10,            # PER applied for first Er epochs
    proto_subset_frac: float = 0.1, # fraction of train data for prototype computation
    proto_momentum: float = 0.5,
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

    print(f"\n========== PMR Fold {fold} ==========")
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

    # Prototype loader: random subset of train data, no shuffle.
    n_proto = max(1, int(len(train_ds) * proto_subset_frac))
    rng = np.random.default_rng(seed=fold)
    proto_indices = rng.choice(len(train_ds), size=n_proto, replace=False).tolist()
    proto_ds = Subset(train_ds, proto_indices)
    proto_ld = DataLoader(proto_ds, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True, drop_last=False)
    print(f"[PMR] prototype subset size: {n_proto}/{len(train_ds)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PMRClassifier(base=base, feat_dim=feat_dim, dropout=dropout,
                          fusion_hidden=fusion_hidden).to(device)

    pos_w = torch.tensor([compute_pos_weight(train_cases)], dtype=torch.float32, device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_w, reduction="mean")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    print(f"[PMR] alpha={alpha} mu={mu} er_epochs={er_epochs}")

    prototypes: Optional[List[torch.Tensor]] = None
    best_auc = -1.0
    history: List[Dict[str, Any]] = []
    EPS = 1e-6
    n_classes = 2
    n_mods = 3

    for ep in range(1, epochs + 1):
        # update prototypes at start of each epoch
        prototypes = compute_prototypes(
            model, proto_ld, device,
            n_classes=n_classes, feat_dim=feat_dim, n_modalities=n_mods,
            prev_protos=prototypes, momentum=proto_momentum,
        )
        # detach prototypes (they are EMA targets, not optimized)
        prototypes = [p.detach() for p in prototypes]

        model.train()
        loss_sum = bce_sum = pce_sum = per_sum = 0.0
        n_iters = 0
        s_acc = torch.zeros(n_mods, device=device)  # cumulative s_m for monitoring
        pbar = tqdm(train_ld, desc=f"[TRAIN][PMR] f{fold} ep{ep}/{epochs}", leave=True)
        for (x0, x1, x2), y, _sid in pbar:
            x0 = x0.to(device); x1 = x1.to(device); x2 = x2.to(device)
            y = y.to(device)
            yc = y.view(-1).long()  # class indices in {0,1}

            # presence
            def pres(x):
                dims = tuple(range(1, x.ndim))
                return (x.abs().sum(dim=dims) > 0)
            keep = pres(x0) & pres(x1) & pres(x2)  # (B,) only fully-present samples for PCE/PER

            logit, (z0, z1, z2) = model((x0, x1, x2))
            l_bce = bce(logit, y)

            # Per-modality prototype-based probabilities (only for keep samples)
            l_pce_total = x0.new_tensor(0.0)
            l_per_total = x0.new_tensor(0.0)
            if keep.any():
                zs = [z0[keep], z1[keep], z2[keep]]
                yk = yc[keep]
                # per-mod soft scores s_m = sum_i p^m_i(y=y_i)
                p_correct = []
                pces = []
                logits_m = []
                for m in range(n_mods):
                    pl = prototype_logits(zs[m], prototypes[m])  # (Nk, K)
                    logits_m.append(pl)
                    p = F.softmax(pl, dim=-1)                     # (Nk, K)
                    p_y = p.gather(1, yk.view(-1, 1)).view(-1)    # (Nk,)
                    p_correct.append(p_y.sum())                   # scalar s_m
                    pces.append(F.cross_entropy(pl, yk, reduction="mean"))
                s = torch.stack(p_correct)                        # (3,)
                s_acc += s.detach()

                # determine dominant
                dom = int(torch.argmax(s).item())
                s_dom = s[dom].clamp(min=EPS)
                s_min = s.min().clamp(min=EPS)
                betas = torch.zeros(n_mods, device=device)
                for m in range(n_mods):
                    if m == dom:
                        continue
                    sm = s[m].clamp(min=EPS)
                    betas[m] = torch.clamp(s_dom / sm - 1.0, 0.0, 1.0)
                gamma_dom = torch.clamp(s_dom / s_min - 1.0, 0.0, 1.0)

                # accumulate PCE
                for m in range(n_mods):
                    l_pce_total = l_pce_total + alpha * betas[m] * pces[m]

                # PER on dominant during early epochs
                if ep <= er_epochs and gamma_dom.item() > 0.0:
                    p_dom = F.softmax(logits_m[dom], dim=-1)
                    H = -(p_dom * (p_dom.clamp(min=EPS).log())).sum(dim=-1).mean()
                    # subtract entropy => maximize entropy => slow down convergence
                    l_per_total = l_per_total - mu * gamma_dom * H

            loss = l_bce + l_pce_total + l_per_total

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            loss_sum += float(loss.item())
            bce_sum += float(l_bce.item())
            pce_sum += float(l_pce_total.item())
            per_sum += float(l_per_total.item())
            n_iters += 1
            pbar.set_postfix(loss=f"{loss_sum/n_iters:.4f}",
                             bce=f"{bce_sum/n_iters:.3f}",
                             pce=f"{pce_sum/n_iters:.3f}",
                             per=f"{per_sum/n_iters:.4f}")

        s_avg = (s_acc / max(1, n_iters)).cpu().numpy().tolist()
        va = eval_one_epoch(model, val_ld, device)
        row = {"epoch": ep, "train_loss": loss_sum / max(1, n_iters),
               "train_bce": bce_sum / max(1, n_iters),
               "train_pce": pce_sum / max(1, n_iters),
               "train_per": per_sum / max(1, n_iters),
               "s_avg": s_avg, "val": va}
        history.append(row)
        print(f"[EP{ep:03d}] loss={row['train_loss']:.4f} | s_avg={s_avg} | "
              f"val auc={va['auc']:.4f} ap={va['ap']:.4f}")

        torch.save({"model": model.state_dict(), "epoch": ep, "fold": fold,
                    "prototypes": [p.cpu() for p in prototypes],
                    "val": va, "method": "pmr"}, last_ckpt_path)

        if not math.isnan(va["auc"]) and va["auc"] > best_auc:
            best_auc = va["auc"]
            torch.save({"model": model.state_dict(), "epoch": ep, "fold": fold,
                        "prototypes": [p.cpu() for p in prototypes],
                        "best_auc": best_auc, "val": va, "method": "pmr"},
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
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--mu", type=float, default=0.01)
    p.add_argument("--er_epochs", type=int, default=10)
    p.add_argument("--proto_subset_frac", type=float, default=0.1)
    p.add_argument("--proto_momentum", type=float, default=0.5)
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
        alpha=float(a.alpha), mu=float(a.mu), er_epochs=int(a.er_epochs),
        proto_subset_frac=float(a.proto_subset_frac), proto_momentum=float(a.proto_momentum),
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fixed-KO training + evaluation utilities (3 modalities) + combo-AUC evaluation script.

This file provides:
1) FixedKnockoutWrapper (TRAIN-time only knockout; eval() disables KO)
2) train_fold / run_10fold_and_summarize (as in your snippet)
3) NEW: evaluate_all_modality_combos() for a given checkpoint (best/last)
   - Computes loss/AUC/AP for each KEEP subset of modalities by zeroing DROPPED modalities.

Run eval only
-------------
python -u -m my_nnunet_cls.eval_fixedko_combos \
  --images ... --labels_csv ... --splits ... \
  --ckpt /path/to/fold_0/best.pt \
  --fold 0 \
  --split val \
  --mod_names "T2W,ADC,HBV" \
  --out_json /path/to/eval_combos_fold0.json

Run training
------------
python -u -m my_nnunet_cls.eval_fixedko_combos \
  --images ... --labels_csv ... --splits ... --outdir ... \
  --epochs 50 --fold 0 \
  --ko_p 0.2 --keep_at_least_one

(If --ckpt is provided -> eval mode. Otherwise -> train mode.)
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
from torch.utils.data import DataLoader
from tqdm import tqdm

# Package-style imports (recommended on your cluster)
from my_nnunet_cls.dataset import build_case_list_classification, Case3DClassificationDataset
from my_nnunet_cls.model import ThreeModalityClassifier
from my_nnunet_cls.utils import safe_auc, safe_ap


# ============================================================
# Fixed KO rate formula
# ============================================================
def compute_base_knockout_rate(num_modalities: int) -> float:
    """
    r = 1 - (0.5 ** (1/d)) so expected keep prob ~0.5 when d modalities exist.
    """
    d = max(1, int(num_modalities))
    return float(1.0 - (0.5 ** (1.0 / d)))


# ============================================================
# Fixed KO wrapper (TRAIN only; placeholders for observed-missing + KO are 0)
# ============================================================
class FixedKnockoutWrapper(nn.Module):
    """
    Fixed Knockout for 3 modalities (x0, x1, x2).

    Presence flag:
      present[b] = sum(abs(x[b])) > 0
    Dataset encodes observed missingness as all-zero tensors.

    Knockout (TRAIN only):
      For each present modality, zero out the entire modality tensor with prob p_mod.
      p_mod fixed; if not provided => p = 1 - (0.5)^(1/3)

    Placeholders:
      observed missing placeholder = 0
      knockout placeholder         = 0
    """

    def __init__(
        self,
        base_model: nn.Module,
        fixed_p: Optional[float] = None,           # scalar prob for all 3 modalities
        fixed_ps: Optional[List[float]] = None,    # [p0,p1,p2]
        keep_at_least_one: bool = True,
    ):
        super().__init__()
        self.model = base_model

        if fixed_ps is not None:
            if len(fixed_ps) != 3:
                raise ValueError(f"fixed_ps must be length 3 for (x0,x1,x2). Got {len(fixed_ps)}")
            self.p = [float(x) for x in fixed_ps]
        else:
            if fixed_p is None:
                fixed_p = compute_base_knockout_rate(3)
            self.p = [float(fixed_p)] * 3

        self.keep_at_least_one = bool(keep_at_least_one)

    @torch.no_grad()
    def _is_present(self, x: torch.Tensor) -> torch.Tensor:
        dims = tuple(range(1, x.ndim))
        return (x.abs().sum(dim=dims) > 0)

    @torch.no_grad()
    def _apply_fixed_knockout(
        self,
        xs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x0, x1, x2 = xs

        # No KO in eval mode
        if not self.training:
            return x0, x1, x2

        B = x0.shape[0]
        device = x0.device

        pres0 = self._is_present(x0)
        pres1 = self._is_present(x1)
        pres2 = self._is_present(x2)

        ko0 = (torch.rand((B,), device=device) < self.p[0]) & pres0
        ko1 = (torch.rand((B,), device=device) < self.p[1]) & pres1
        ko2 = (torch.rand((B,), device=device) < self.p[2]) & pres2

        if self.keep_at_least_one:
            present_count = pres0.long() + pres1.long() + pres2.long()
            knocked_count = ko0.long() + ko1.long() + ko2.long()
            all_knocked = (present_count > 0) & (knocked_count >= present_count)

            if all_knocked.any():
                idx = torch.where(all_knocked)[0]
                for b in idx.tolist():
                    candidates: List[int] = []
                    if pres0[b].item(): candidates.append(0)
                    if pres1[b].item(): candidates.append(1)
                    if pres2[b].item(): candidates.append(2)
                    if len(candidates) > 0:
                        keep_mod = candidates[int(torch.randint(0, len(candidates), (1,), device=device).item())]
                        if keep_mod == 0:
                            ko0[b] = False
                        elif keep_mod == 1:
                            ko1[b] = False
                        else:
                            ko2[b] = False

        def zero_out(x: torch.Tensor, ko: torch.Tensor) -> torch.Tensor:
            mask = (~ko).to(dtype=x.dtype, device=x.device)
            view = (x.shape[0],) + (1,) * (x.ndim - 1)
            return x * mask.view(view)

        x0k = zero_out(x0, ko0)
        x1k = zero_out(x1, ko1)
        x2k = zero_out(x2, ko2)
        return x0k, x1k, x2k

    def forward(self, xs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        x0, x1, x2 = xs
        x0, x1, x2 = self._apply_fixed_knockout((x0, x1, x2))
        return self.model((x0, x1, x2))


# ============================================================
# Helpers
# ============================================================
def compute_pos_weight(cases) -> float:
    ys = np.array([c["y"] for c in cases], dtype=np.int64)
    pos = float((ys == 1).sum())
    neg = float((ys == 0).sum())
    if pos < 1:
        return 1.0
    return neg / pos


def _load_fold_split(splits_json: Path, fold: int) -> Dict[str, List[str]]:
    splits = json.load(open(splits_json, "r"))

    if isinstance(splits, list):
        fold_split = splits[fold]
    else:
        if str(fold) in splits:
            fold_split = splits[str(fold)]
        else:
            fold_split = splits[fold]

    if "train" not in fold_split or "val" not in fold_split:
        raise ValueError(f"Fold split must contain keys train/val. Got keys: {list(fold_split.keys())}")

    train_ids = [str(x).strip() for x in fold_split["train"]]
    val_ids = [str(x).strip() for x in fold_split["val"]]
    test_ids = [str(x).strip() for x in fold_split.get("test", [])]
    return {"train": train_ids, "val": val_ids, "test": test_ids}


def _debug_match(split_ids: List[str], discovered_ids: List[str], tag: str) -> None:
    split_set = set(split_ids)
    disc_set = set(discovered_ids)
    inter = len(split_set & disc_set)

    print(f"\n[DEBUG MATCH] {tag}")
    print("  discovered:", len(disc_set))
    print("  split_ids :", len(split_set), " matched:", inter)
    print("  example split ids      :", split_ids[:5])
    print("  example discovered ids :", discovered_ids[:5])

    missing = list(split_set - disc_set)[:10]
    if missing:
        print("  example split ids NOT found in discovered:", missing[:5])


class DropModalitiesWrapper(torch.utils.data.Dataset):
    """
    Wrap a dataset returning ((x0,x1,x2), y, sid) and zero out dropped modalities.
    """
    def __init__(self, base_ds: torch.utils.data.Dataset, dropped: List[int]):
        self.base_ds = base_ds
        self.dropped = [int(i) for i in dropped]

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        (x0, x1, x2), y, sid = self.base_ds[idx]
        xs = [x0, x1, x2]
        for m in self.dropped:
            if m < 0 or m > 2:
                raise ValueError(f"dropped index must be in [0,1,2], got {m}")
            xs[m] = torch.zeros_like(xs[m])
        return (xs[0], xs[1], xs[2]), y, sid


# ============================================================
# Eval
# ============================================================
@torch.no_grad()
def eval_one_epoch(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    y_true: List[int] = []
    y_score: List[float] = []
    loss_sum = 0.0
    n = 0

    bce = nn.BCEWithLogitsLoss(reduction="mean")

    pbar = tqdm(loader, desc="[EVAL]", leave=False)
    for (x0, x1, x2), y, sid in pbar:
        x0 = x0.to(device, non_blocking=True)
        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logit = model((x0, x1, x2))
        loss = bce(logit, y)

        prob = torch.sigmoid(logit).view(-1).detach().cpu().numpy().tolist()
        yt = y.view(-1).detach().cpu().numpy().astype(int).tolist()

        y_score.extend(prob)
        y_true.extend(yt)

        loss_sum += float(loss.item())
        n += 1
        pbar.set_postfix(loss=f"{(loss_sum / max(1, n)):.4f}")

    auc = safe_auc(y_true, y_score)
    ap = safe_ap(y_true, y_score)
    return {"loss": loss_sum / max(1, n), "auc": float(auc), "ap": float(ap)}


def _combo_name_from_keep(keep: List[int], mod_names: List[str]) -> str:
    if len(keep) == 0:
        return "NONE"
    return "+".join([mod_names[i] for i in keep])


@torch.no_grad()
def evaluate_all_modality_combos(
    base_model: nn.Module,
    eval_ds: torch.utils.data.Dataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    mod_names: List[str],
    include_empty: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate checkpoint model on all KEEP subsets by zeroing DROPPED modalities.
    NOTE: we evaluate the BASE model (no FixedKO wrapper), because KO is train-time only.
    """
    M = 3
    if len(mod_names) != M:
        mod_names = ["M0", "M1", "M2"]

    keep_sets: List[List[int]] = []
    for mask in range(0, 1 << M):
        keep = [i for i in range(M) if (mask >> i) & 1]
        if (not include_empty) and len(keep) == 0:
            continue
        keep_sets.append(keep)

    results: List[Dict[str, Any]] = []
    for keep in keep_sets:
        dropped = [i for i in range(M) if i not in keep]
        ds_combo = DropModalitiesWrapper(eval_ds, dropped=dropped)
        loader = DataLoader(
            ds_combo,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )
        metrics = eval_one_epoch(base_model, loader, device)
        results.append({
            "keep": keep,
            "dropped": dropped,
            "name": _combo_name_from_keep(keep, mod_names),
            **metrics,
        })

    results.sort(key=lambda r: (len(r["keep"]), r["name"]))

    aucs = np.array([r["auc"] for r in results], dtype=np.float64)
    aps = np.array([r["ap"] for r in results], dtype=np.float64)

    summary = {
        "n_combos": len(results),
        "mean_auc": float(np.nanmean(aucs)) if np.isfinite(aucs).any() else float("nan"),
        "mean_ap": float(np.nanmean(aps)) if np.isfinite(aps).any() else float("nan"),
        "best_auc": float(np.nanmax(aucs)) if np.isfinite(aucs).any() else float("nan"),
        "best_auc_combo": results[int(np.nanargmax(aucs))]["name"] if np.isfinite(aucs).any() else None,
    }
    return {"summary": summary, "per_combo": results}


# ============================================================
# Checkpoint loading
# ============================================================
def load_base_model_from_ckpt(
    ckpt_path: Path,
    device: torch.device,
    base: int,
    feat_dim: int,
    dropout: float,
    fusion_hidden: int,
) -> nn.Module:
    """
    Loads weights into ThreeModalityClassifier base model.
    Supports checkpoints saved from FixedKnockoutWrapper (keys prefixed with model.)
    and/or DataParallel (module.).
    """
    model = ThreeModalityClassifier(
        base=base,
        feat_dim=feat_dim,
        dropout=dropout,
        fusion_hidden=fusion_hidden,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model", ckpt.get("state_dict", ckpt))

    if isinstance(state, dict):
        if any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
        if any(k.startswith("model.") for k in state.keys()):
            state = {k.replace("model.", "", 1): v for k, v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("[WARN] Missing keys (up to 20):", missing[:20])
    if unexpected:
        print("[WARN] Unexpected keys (up to 20):", unexpected[:20])

    model.eval()
    return model


# ============================================================
# Train (unchanged from your structure)
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
    base=32,
    feat_dim=256,
    dropout=0.1,
    fusion_hidden=256,
    ko_p: Optional[float] = None,
    ko_ps: Optional[List[float]] = None,
    keep_at_least_one: bool = True,
) -> Dict[str, Any]:
    outdir.mkdir(parents=True, exist_ok=True)
    fold_dir = outdir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    best_ckpt_path = fold_dir / "best.pt"
    last_ckpt_path = fold_dir / "last.pt"
    history_path = fold_dir / "history.json"

    all_cases = build_case_list_classification(
        images_root=images_root,
        labels_csv=labels_csv,
        strict=True,
        verbose=True,
    )

    id_to_case: Dict[str, Dict[str, Any]] = {str(c["sid"]).strip(): c for c in all_cases}

    fold_split = _load_fold_split(splits_json, fold)
    train_ids = fold_split["train"]
    val_ids = fold_split["val"]

    train_cases = [id_to_case[s] for s in train_ids if s in id_to_case]
    val_cases = [id_to_case[s] for s in val_ids if s in id_to_case]

    print(f"\n========== Fold {fold} ==========")
    print(f"All discovered: {len(all_cases)} | train={len(train_cases)} val={len(val_cases)}")

    if len(train_cases) == 0 or len(val_cases) == 0:
        _debug_match(train_ids, list(id_to_case.keys())[:2000], tag=f"fold{fold}:train")
        _debug_match(val_ids, list(id_to_case.keys())[:2000], tag=f"fold{fold}:val")
        raise RuntimeError("Empty train/val after matching splits to discovered cases. Check sid parsing.")

    train_ds = Case3DClassificationDataset(
        train_cases, target_shape=target_shape, normalize=True, align_to_ref=True, return_sid=True
    )
    val_ds = Case3DClassificationDataset(
        val_cases, target_shape=target_shape, normalize=True, align_to_ref=True, return_sid=True
    )

    train_ld = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_ld = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_model = ThreeModalityClassifier(
        base=base, feat_dim=feat_dim, dropout=dropout, fusion_hidden=fusion_hidden
    ).to(device)

    model = FixedKnockoutWrapper(
        base_model=base_model,
        fixed_p=ko_p,
        fixed_ps=ko_ps,
        keep_at_least_one=keep_at_least_one,
    ).to(device)

    print(f"[KO] fixed rates p={model.p} (formula default {compute_base_knockout_rate(3):.6f}) "
          f"| keep_at_least_one={keep_at_least_one} | placeholders observed=0 knockout=0")

    pos_w = compute_pos_weight(train_cases)
    pos_w_t = torch.tensor([pos_w], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w_t, reduction="mean")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_auc = -1.0
    history: List[Dict[str, Any]] = []

    for ep in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        n = 0

        pbar = tqdm(train_ld, desc=f"[TRAIN] fold{fold} ep{ep}/{epochs}", leave=True)
        for (x0, x1, x2), y, sid in pbar:
            x0 = x0.to(device, non_blocking=True)
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logit = model((x0, x1, x2))
            loss = criterion(logit, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            loss_sum += float(loss.item())
            n += 1
            pbar.set_postfix(loss=f"{(loss_sum / max(1, n)):.4f}")

        va = eval_one_epoch(model, val_ld, device)

        row = {"epoch": ep, "train_loss": loss_sum / max(1, n), "val": va}
        history.append(row)

        print(
            f"[EPOCH {ep:03d}] train_loss={row['train_loss']:.4f} | "
            f"val_loss={va['loss']:.4f} auc={va['auc']:.4f} ap={va['ap']:.4f}"
        )

        torch.save({"model": model.state_dict(), "epoch": ep, "fold": fold, "val": va}, last_ckpt_path)

        if not math.isnan(va["auc"]) and va["auc"] > best_auc:
            best_auc = va["auc"]
            torch.save(
                {"model": model.state_dict(), "epoch": ep, "fold": fold, "best_auc": best_auc, "val": va},
                best_ckpt_path,
            )
            print(f"  ✅ New best checkpoint: {best_ckpt_path} (auc={best_auc:.4f})")

        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

    return {"fold": fold, "best_auc": best_auc, "best_ckpt": str(best_ckpt_path), "history": str(history_path)}


def run_10fold_and_summarize(
    images_root: Path,
    labels_csv: Path,
    splits_json: Path,
    outdir: Path,
    epochs: int = 50,
    batch_size: int = 2,
    lr: float = 3e-4,
    num_workers: int = 4,
    target_shape=(128, 192, 192),
    ko_p: Optional[float] = None,
    ko_ps: Optional[List[float]] = None,
    keep_at_least_one: bool = True,
):
    outdir.mkdir(parents=True, exist_ok=True)
    results = []
    for fold in range(10):
        results.append(
            train_fold(
                images_root=images_root,
                labels_csv=labels_csv,
                splits_json=splits_json,
                outdir=outdir,
                fold=fold,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                num_workers=num_workers,
                target_shape=target_shape,
                ko_p=ko_p,
                ko_ps=ko_ps,
                keep_at_least_one=keep_at_least_one,
            )
        )

    scores = np.array([r["best_auc"] for r in results], dtype=np.float64)
    mean = float(np.nanmean(scores))
    std = float(np.nanstd(scores, ddof=1)) if np.isfinite(scores).sum() > 1 else float("nan")

    summary = {
        "n_folds": 10,
        "best_auc_per_fold": scores.tolist(),
        "mean_best_auc": mean,
        "std_best_auc": std,
        "results": results,
        "ko_p": ko_p,
        "ko_ps": ko_ps,
        "formula_default_p": compute_base_knockout_rate(3),
    }
    summary_path = outdir / "cv_summary.json"
    json.dump(summary, open(summary_path, "w"), indent=2)

    print("\n========== 10-Fold Summary ==========")
    for i, s in enumerate(scores.tolist()):
        print(f"Fold {i}: best_auc={s:.4f}")
    print(f"MEAN best_auc: {mean:.4f}")
    print(f"STD  best_auc:  {std:.4f}")
    print("Saved:", summary_path)


# ============================================================
# CLI
# ============================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True)
    p.add_argument("--labels_csv", required=True)
    p.add_argument("--splits", required=True)

    # If ckpt is provided => eval mode
    p.add_argument("--ckpt", type=str, default=None, help="Checkpoint path. If set, runs eval mode.")

    # Training args
    p.add_argument("--outdir", type=str, default=None, help="Output directory for training (train mode).")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=3e-4)

    # Shared
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--tZ", type=int, default=128)
    p.add_argument("--tY", type=int, default=192)
    p.add_argument("--tX", type=int, default=192)

    p.add_argument("--base", type=int, default=32)
    p.add_argument("--feat_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--fusion_hidden", type=int, default=256)

    # Fixed KO (train mode)
    p.add_argument("--ko_p", type=float, default=None,
                   help="Fixed KO prob for all 3 modalities (train only). If omitted, uses formula-based p.")
    p.add_argument("--ko_p0", type=float, default=None, help="Optional per-modality KO prob for x0.")
    p.add_argument("--ko_p1", type=float, default=None, help="Optional per-modality KO prob for x1.")
    p.add_argument("--ko_p2", type=float, default=None, help="Optional per-modality KO prob for x2.")
    p.add_argument("--keep_at_least_one", action="store_true",
                   help="Ensure at least one present modality remains (train only).")

    # Eval combo args
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--mod_names", type=str, default="M0,M1,M2",
                   help='Comma-separated modality names (e.g., "T2W,ADC,HBV")')
    p.add_argument("--include_empty", action="store_true",
                   help="Also evaluate NONE (all modalities dropped). Usually not meaningful.")
    p.add_argument("--out_json", type=str, default="eval_fixedko_combos.json")

    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[DEVICE]", device)

    images_root = Path(args.images)
    labels_csv = Path(args.labels_csv)
    splits_json = Path(args.splits)

    target_shape = (args.tZ, args.tY, args.tX)
    mod_names = [s.strip() for s in args.mod_names.split(",")]

    # Discover all labeled cases and select the requested split
    all_cases = build_case_list_classification(
        images_root=images_root,
        labels_csv=labels_csv,
        strict=True,
        verbose=True,
    )
    id_to_case: Dict[str, Dict[str, Any]] = {str(c["sid"]).strip(): c for c in all_cases}

    fold_split = _load_fold_split(splits_json, args.fold)
    split_ids = fold_split[args.split]
    split_cases = [id_to_case[s] for s in split_ids if s in id_to_case]
    print(f"[SPLIT] fold={args.fold} split={args.split} cases={len(split_cases)} discovered={len(all_cases)}")

    if len(split_cases) == 0:
        _debug_match(split_ids, list(id_to_case.keys())[:2000], tag=f"fold{args.fold}:{args.split}")
        raise RuntimeError("Empty eval split after matching split IDs to discovered cases. Check sid parsing.")

    eval_ds = Case3DClassificationDataset(
        split_cases,
        target_shape=target_shape,
        normalize=True,
        align_to_ref=True,
        return_sid=True,
    )

    # ------------------
    # EVAL MODE
    # ------------------
    if args.ckpt is not None:
        ckpt_path = Path(args.ckpt)
        print("[EVAL] ckpt:", ckpt_path)

        base_model = load_base_model_from_ckpt(
            ckpt_path=ckpt_path,
            device=device,
            base=args.base,
            feat_dim=args.feat_dim,
            dropout=args.dropout,
            fusion_hidden=args.fusion_hidden,
        )

        report = evaluate_all_modality_combos(
            base_model=base_model,
            eval_ds=eval_ds,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            mod_names=mod_names,
            include_empty=bool(args.include_empty),
        )

        print("\n========== Modality Combo Results ==========")
        for r in report["per_combo"]:
            print(
                f"{r['name']:<20} | keep={r['keep']} drop={r['dropped']} "
                f"| loss={r['loss']:.4f} auc={r['auc']:.4f} ap={r['ap']:.4f}"
            )
        print("\n========== Summary ==========")
        for k, v in report["summary"].items():
            print(f"{k}: {v}")

        payload = {
            "fold": int(args.fold),
            "split": args.split,
            "ckpt": str(ckpt_path),
            "mod_names": mod_names,
            "target_shape": list(target_shape),
            "batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
            **report,
        }
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(payload, f, indent=2)
        print("\nSaved:", out_json)
        return

    # ------------------
    # TRAIN MODE
    # ------------------
    if args.outdir is None:
        raise ValueError("Train mode requires --outdir (or provide --ckpt for eval mode).")

    ko_ps = None
    if args.ko_p0 is not None or args.ko_p1 is not None or args.ko_p2 is not None:
        base_p = args.ko_p if args.ko_p is not None else compute_base_knockout_rate(3)
        ko_ps = [
            float(args.ko_p0 if args.ko_p0 is not None else base_p),
            float(args.ko_p1 if args.ko_p1 is not None else base_p),
            float(args.ko_p2 if args.ko_p2 is not None else base_p),
        ]

    train_fold(
        images_root=images_root,
        labels_csv=labels_csv,
        splits_json=splits_json,
        outdir=Path(args.outdir),
        fold=int(args.fold),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        num_workers=int(args.num_workers),
        target_shape=target_shape,
        base=int(args.base),
        feat_dim=int(args.feat_dim),
        dropout=float(args.dropout),
        fusion_hidden=int(args.fusion_hidden),
        ko_p=args.ko_p,
        ko_ps=ko_ps,
        keep_at_least_one=bool(args.keep_at_least_one),
    )


if __name__ == "__main__":
    main()

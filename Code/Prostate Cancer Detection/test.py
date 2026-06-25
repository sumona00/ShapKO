#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eval script for the *baseline* (no KO) 3-modality classifier training you pasted.

What it does
------------
- Loads fold split (val or test) from splits_10fold.json
- Loads checkpoint (best.pt / last.pt) into ThreeModalityClassifier
- Evaluates:
    (1) FULL input (whatever modalities are present in the dataset)
    (2) ALL modality combinations (keep-subsets) by zeroing the dropped modalities
- Saves a JSON report (and prints a nice table)

Run (single fold)
-----------------
python -u -m my_nnunet_cls.eval_baseline_combos \
  --images /path/to/images \
  --labels_csv /path/to/marksheet.csv \
  --splits /path/to/splits_10fold.json \
  --fold 0 \
  --ckpt /path/to/checkpoints_10fold/fold_0/best.pt \
  --split val \
  --mod_names "T2W,ADC,HBV" \
  --out_json /path/to/checkpoints_10fold/fold_0/eval_baseline_combos_fold0.json

Notes
-----
- Dataset must return: ((x0,x1,x2), y, sid)
- "Observed missingness" encoded as all-zero tensors is respected.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import argparse
import json
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# IMPORTANT: use package imports (works with python -m ...)
from my_nnunet_cls.dataset import build_case_list_classification, Case3DClassificationDataset
from my_nnunet_cls.model import ThreeModalityClassifier
from my_nnunet_cls.utils import safe_auc, safe_ap


# -------------------------
# Split helpers
# -------------------------
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


# -------------------------
# Dataset wrapper for combos
# -------------------------
class DropModalitiesWrapper(torch.utils.data.Dataset):
    """
    Wraps a dataset returning ((x0,x1,x2), y, sid) and zeros dropped modalities.
    dropped: list of indices in {0,1,2}
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


# -------------------------
# Eval core
# -------------------------
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
    return "+".join(mod_names[i] for i in keep)


@torch.no_grad()
def eval_all_combos(
    model: nn.Module,
    base_ds: torch.utils.data.Dataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    mod_names: List[str],
    include_empty: bool = False,
) -> Dict[str, Any]:
    """
    Evaluates AUC/AP for every KEEP subset by zeroing the DROPPED modalities.
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

    per_combo: List[Dict[str, Any]] = []
    for keep in keep_sets:
        dropped = [i for i in range(M) if i not in keep]
        ds_combo = DropModalitiesWrapper(base_ds, dropped=dropped)
        loader = DataLoader(
            ds_combo,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )
        metrics = eval_one_epoch(model, loader, device)
        per_combo.append({
            "name": _combo_name_from_keep(keep, mod_names),
            "keep": keep,
            "dropped": dropped,
            **metrics,
        })

    per_combo.sort(key=lambda r: (len(r["keep"]), r["name"]))

    aucs = np.array([r["auc"] for r in per_combo], dtype=np.float64)
    aps = np.array([r["ap"] for r in per_combo], dtype=np.float64)

    summary = {
        "n_combos": len(per_combo),
        "mean_auc": float(np.nanmean(aucs)) if np.isfinite(aucs).any() else float("nan"),
        "mean_ap": float(np.nanmean(aps)) if np.isfinite(aps).any() else float("nan"),
        "best_auc": float(np.nanmax(aucs)) if np.isfinite(aucs).any() else float("nan"),
        "best_auc_combo": per_combo[int(np.nanargmax(aucs))]["name"] if np.isfinite(aucs).any() else None,
    }
    return {"summary": summary, "per_combo": per_combo}


# -------------------------
# Checkpoint loading
# -------------------------
def load_model_from_ckpt(
    ckpt_path: Path,
    device: torch.device,
    base: int,
    feat_dim: int,
    dropout: float,
    fusion_hidden: int,
) -> nn.Module:
    model = ThreeModalityClassifier(
        base=base,
        feat_dim=feat_dim,
        dropout=dropout,
        fusion_hidden=fusion_hidden,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model", ckpt.get("state_dict", ckpt))

    # Support DataParallel and wrapper-prefixed keys
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


# -------------------------
# Main
# -------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True)
    p.add_argument("--labels_csv", required=True)
    p.add_argument("--splits", required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--split", choices=["val", "test"], default="val")

    p.add_argument("--ckpt", required=True, help="Path to fold checkpoint (best.pt/last.pt).")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("--tZ", type=int, default=128)
    p.add_argument("--tY", type=int, default=192)
    p.add_argument("--tX", type=int, default=192)

    p.add_argument("--base", type=int, default=32)
    p.add_argument("--feat_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--fusion_hidden", type=int, default=256)

    p.add_argument("--mod_names", type=str, default="M0,M1,M2")
    p.add_argument("--include_empty", action="store_true")
    p.add_argument("--out_json", type=str, default="eval_baseline_combos.json")

    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[DEVICE]", device)

    images_root = Path(args.images)
    labels_csv = Path(args.labels_csv)
    splits_json = Path(args.splits)
    ckpt_path = Path(args.ckpt)

    target_shape = (args.tZ, args.tY, args.tX)
    mod_names = [s.strip() for s in args.mod_names.split(",")]

    # Discover cases
    all_cases = build_case_list_classification(
        images_root=images_root,
        labels_csv=labels_csv,
        strict=True,
        verbose=True,
    )
    id_to_case: Dict[str, Dict[str, Any]] = {str(c["sid"]).strip(): c for c in all_cases}

    fold_split = _load_fold_split(splits_json, args.fold)
    split_ids = fold_split[args.split]
    if args.split == "test" and len(split_ids) == 0:
        raise RuntimeError("Requested --split test but no 'test' entries exist for this fold.")

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

    # Load model
    model = load_model_from_ckpt(
        ckpt_path=ckpt_path,
        device=device,
        base=args.base,
        feat_dim=args.feat_dim,
        dropout=args.dropout,
        fusion_hidden=args.fusion_hidden,
    )

    # Full (no artificial drops)
    full_loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    full_metrics = eval_one_epoch(model, full_loader, device)

    # Combos
    combo_report = eval_all_combos(
        model=model,
        base_ds=eval_ds,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        mod_names=mod_names,
        include_empty=bool(args.include_empty),
    )

    print("\n========== FULL (as-is) ==========")
    print(f"loss={full_metrics['loss']:.4f} auc={full_metrics['auc']:.4f} ap={full_metrics['ap']:.4f}")

    print("\n========== COMBOS ==========")
    for r in combo_report["per_combo"]:
        print(
            f"{r['name']:<20} | keep={r['keep']} drop={r['dropped']} "
            f"| loss={r['loss']:.4f} auc={r['auc']:.4f} ap={r['ap']:.4f}"
        )

    # Save JSON
    payload = {
        "fold": int(args.fold),
        "split": args.split,
        "ckpt": str(ckpt_path),
        "mod_names": mod_names,
        "target_shape": list(target_shape),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "full": full_metrics,
        **combo_report,
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print("\nSaved:", out_json)


if __name__ == "__main__":
    main()

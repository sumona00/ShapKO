#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate AUC/AP for ALL modality combinations (3-modality classifier).

What it does
------------
- Loads your dataset + a trained checkpoint
- Builds an eval split (val by default) using your splits json
- Evaluates the model under every modality subset by ZEROING the dropped modalities:
    keep: (x0) (x1) (x2) (x0+x1) (x0+x2) (x1+x2) (x0+x1+x2) plus (none kept) (optional)
- Reports loss/AUC/AP per combo and saves JSON.

Assumptions (matching your code)
--------------------------------
Dataset __getitem__ returns: ((x0, x1, x2), y, sid)
- "Observed missingness" is already encoded as all-zero tensors
- We enforce artificial drops by zeroing out those modalities (same placeholder=0)

Run
---
python eval_auc_modality_combos.py \
  --images /path/to/images_root \
  --labels_csv /path/to/labels.csv \
  --splits /path/to/splits_10fold.json \
  --ckpt /path/to/fold_0/best.pt \
  --fold 0 \
  --split val \
  --batch_size 2 \
  --num_workers 4 \
  --out_json /path/to/eval_combos_fold0.json
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
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
# Your project imports
from my_nnunet_cls.dataset import build_case_list_classification, Case3DClassificationDataset
from my_nnunet_cls.model import ThreeModalityClassifier
from my_nnunet_cls.utils import safe_auc, safe_ap


# ============================================================
# Split utilities (same behavior as your train script)
# ============================================================
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
    # Optional: allow "test" if present
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


# ============================================================
# Modality dropping wrapper
# ============================================================
class DropModalitiesWrapper(torch.utils.data.Dataset):
    """
    Wraps a dataset that returns: ((x0, x1, x2), y, sid)
    and enforces a dropped-modality combination by zeroing the dropped modalities.
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
# Eval core
# ============================================================
@torch.no_grad()
def eval_loader_auc_ap(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
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


def evaluate_all_combinations(
    model: nn.Module,
    base_ds: torch.utils.data.Dataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    mod_names: Optional[List[str]] = None,
    include_empty: bool = False,
) -> Dict[str, Any]:
    """
    Returns dict with per-combo metrics + a compact summary.
    """
    M = 3
    if mod_names is None or len(mod_names) != M:
        mod_names = ["M0", "M1", "M2"]

    # All subsets of modalities to KEEP
    keep_sets: List[List[int]] = []
    for mask in range(0, 1 << M):
        keep = [i for i in range(M) if (mask >> i) & 1]
        if (not include_empty) and len(keep) == 0:
            continue
        keep_sets.append(keep)

    results: List[Dict[str, Any]] = []
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

        metrics = eval_loader_auc_ap(model, loader, device)
        row = {
            "keep": keep,
            "dropped": dropped,
            "name": _combo_name_from_keep(keep, mod_names),
            **metrics,
        }
        results.append(row)

    # Sort by number of modalities kept (then name)
    results.sort(key=lambda r: (len(r["keep"]), r["name"]))

    # Simple summaries
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
    # Accept common checkpoint formats
    state = ckpt.get("model", ckpt.get("state_dict", ckpt))
    # If saved from wrapper, keys might be "model.xxx" (or "module.xxx")
    # Try a few safe normalizations.
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    if any(k.startswith("model.") for k in state.keys()):
        state = {k.replace("model.", "", 1): v for k, v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("[WARN] Missing keys (showing up to 20):", missing[:20])
    if unexpected:
        print("[WARN] Unexpected keys (showing up to 20):", unexpected[:20])

    model.eval()
    return model


# ============================================================
# Main
# ============================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True, type=str)
    p.add_argument("--labels_csv", required=True, type=str)
    p.add_argument("--splits", required=True, type=str)
    p.add_argument("--ckpt", required=True, type=str)
    p.add_argument("--fold", required=True, type=int)
    p.add_argument("--split", default="val", choices=["val", "test"], help="Which split to evaluate.")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--tZ", type=int, default=128)
    p.add_argument("--tY", type=int, default=192)
    p.add_argument("--tX", type=int, default=192)
    p.add_argument("--base", type=int, default=32)
    p.add_argument("--feat_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--fusion_hidden", type=int, default=256)

    p.add_argument("--mod_names", type=str, default="M0,M1,M2",
                   help='Comma-separated modality names (e.g., "CT,ECG,TAB")')
    p.add_argument("--include_empty", action="store_true",
                   help="If set, also evaluate NONE (all modalities dropped). Usually not meaningful.")
    p.add_argument("--out_json", type=str, default="eval_modality_combos.json")

    args = p.parse_args()

    images_root = Path(args.images)
    labels_csv = Path(args.labels_csv)
    splits_json = Path(args.splits)
    ckpt_path = Path(args.ckpt)
    out_json = Path(args.out_json)

    mod_names = [s.strip() for s in args.mod_names.split(",")]
    target_shape = (args.tZ, args.tY, args.tX)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[DEVICE]", device)

    # Discover all labeled cases
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
        raise RuntimeError(
            "You requested --split test but no 'test' key exists (or it's empty) in this fold split."
        )

    split_cases = [id_to_case[s] for s in split_ids if s in id_to_case]
    print(f"[SPLIT] fold={args.fold} split={args.split} cases={len(split_cases)} (discovered={len(all_cases)})")

    if len(split_cases) == 0:
        _debug_match(split_ids, list(id_to_case.keys())[:2000], tag=f"fold{args.fold}:{args.split}")
        raise RuntimeError("Empty eval split after matching split IDs to discovered cases. Check sid construction.")

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

    # Evaluate all modality combos
    report = evaluate_all_combinations(
        model=model,
        base_ds=eval_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        mod_names=mod_names,
        include_empty=bool(args.include_empty),
    )

    # Pretty print
    print("\n========== Modality Combo Results ==========")
    for r in report["per_combo"]:
        print(
            f"{r['name']:<20} | keep={r['keep']} drop={r['dropped']} "
            f"| loss={r['loss']:.4f} auc={r['auc']:.4f} ap={r['ap']:.4f}"
        )

    print("\n========== Summary ==========")
    for k, v in report["summary"].items():
        print(f"{k}: {v}")

    # Save
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
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print("\nSaved:", out_json)


if __name__ == "__main__":
    main()

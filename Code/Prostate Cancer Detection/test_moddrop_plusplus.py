"""
Evaluate a ModDrop++ checkpoint over all 7 modality keep-subsets.

For each keep-subset S, the dropped modalities are zeroed in the input AND the
modality code passed to the dynamic head is set accordingly. AUC/AP/loss are
reported for every combination.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List
import argparse
import json
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from my_nnunet_cls.dataset import build_case_list_classification, Case3DClassificationDataset
from my_nnunet_cls.utils import safe_auc, safe_ap
from my_nnunet_cls.train_moddrop_plusplus import (
    ModDropPPClassifier, presence_from_inputs, apply_mcode_to_inputs, _load_fold_split,
)


def load_checkpoint(ckpt_path: Path, device: torch.device, base: int, feat_dim: int,
                    dropout: float, fusion_hidden: int) -> ModDropPPClassifier:
    model = ModDropPPClassifier(base=base, feat_dim=feat_dim, dropout=dropout,
                                fusion_hidden=fusion_hidden).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model", ckpt.get("state_dict", ckpt))
    if isinstance(state, dict):
        if any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("[WARN] missing keys (up to 20):", missing[:20])
    if unexpected:
        print("[WARN] unexpected keys (up to 20):", unexpected[:20])
    model.eval()
    return model


@torch.no_grad()
def eval_one_combo(model: ModDropPPClassifier, loader: DataLoader, device: torch.device,
                   keep: List[int]) -> Dict[str, float]:
    model.eval()
    bce = nn.BCEWithLogitsLoss(reduction="mean")
    keep_set = set(int(i) for i in keep)
    y_true: List[int] = []
    y_score: List[float] = []
    loss_sum = 0.0
    n = 0

    pbar = tqdm(loader, desc=f"[EVAL keep={keep}]", leave=False)
    for (x0, x1, x2), y, _sid in pbar:
        x0 = x0.to(device, non_blocking=True)
        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        B = x0.shape[0]

        # observed-presence mask AND keep-subset
        pres = presence_from_inputs((x0, x1, x2)).to(device)
        keep_vec = torch.zeros(3, dtype=torch.bool, device=device)
        for i in keep_set:
            keep_vec[i] = True
        mcode_bool = pres & keep_vec.view(1, 3).expand(B, 3)
        mcode = mcode_bool.float()

        xs = apply_mcode_to_inputs((x0, x1, x2), mcode)
        logit, _ = model(xs, mcode)
        loss = bce(logit, y)

        y_score.extend(torch.sigmoid(logit).view(-1).cpu().numpy().tolist())
        y_true.extend(y.view(-1).cpu().numpy().astype(int).tolist())
        loss_sum += float(loss.item()); n += 1
        pbar.set_postfix(loss=f"{loss_sum/max(1,n):.4f}")

    return {"loss": loss_sum / max(1, n), "auc": float(safe_auc(y_true, y_score)),
            "ap": float(safe_ap(y_true, y_score))}


def evaluate_all_combos(model: ModDropPPClassifier, ds, device: torch.device,
                        batch_size: int, num_workers: int, mod_names: List[str],
                        include_empty: bool = False) -> Dict[str, Any]:
    M = 3
    keep_sets = []
    for mask in range(0, 1 << M):
        keep = [i for i in range(M) if (mask >> i) & 1]
        if (not include_empty) and len(keep) == 0:
            continue
        keep_sets.append(keep)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True, drop_last=False)
    results = []
    for keep in keep_sets:
        m = eval_one_combo(model, loader, device, keep)
        name = "+".join(mod_names[i] for i in keep) if keep else "NONE"
        results.append({"keep": keep, "dropped": [i for i in range(M) if i not in keep],
                        "name": name, **m})

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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True)
    p.add_argument("--labels_csv", required=True)
    p.add_argument("--splits", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--tZ", type=int, default=128)
    p.add_argument("--tY", type=int, default=192)
    p.add_argument("--tX", type=int, default=192)
    p.add_argument("--base", type=int, default=32)
    p.add_argument("--feat_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--fusion_hidden", type=int, default=256)
    p.add_argument("--mod_names", type=str, default="T2W,ADC,HBV")
    p.add_argument("--include_empty", action="store_true")
    p.add_argument("--out_json", type=str, default="eval_moddrop_plusplus.json")
    return p.parse_args()


def main():
    a = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[DEVICE]", device)

    images_root = Path(a.images); labels_csv = Path(a.labels_csv); splits_json = Path(a.splits)
    target_shape = (a.tZ, a.tY, a.tX)
    mod_names = [s.strip() for s in a.mod_names.split(",")]

    all_cases = build_case_list_classification(
        images_root=images_root, labels_csv=labels_csv, strict=True, verbose=True,
    )
    id_to_case = {str(c["sid"]).strip(): c for c in all_cases}
    sp = _load_fold_split(splits_json, a.fold)
    split_ids = sp.get(a.split, [])
    split_cases = [id_to_case[s] for s in split_ids if s in id_to_case]
    print(f"[SPLIT] fold={a.fold} split={a.split} n={len(split_cases)}")
    if len(split_cases) == 0:
        raise RuntimeError("Empty eval split.")

    ds = Case3DClassificationDataset(split_cases, target_shape=target_shape,
                                     normalize=True, align_to_ref=True, return_sid=True)
    model = load_checkpoint(Path(a.ckpt), device, a.base, a.feat_dim, a.dropout, a.fusion_hidden)
    report = evaluate_all_combos(model, ds, device, a.batch_size, a.num_workers,
                                 mod_names, include_empty=bool(a.include_empty))

    print("\n========== ModDrop++ combo results ==========")
    for r in report["per_combo"]:
        print(f"{r['name']:<20} | keep={r['keep']} drop={r['dropped']} | "
              f"loss={r['loss']:.4f} auc={r['auc']:.4f} ap={r['ap']:.4f}")
    print("\n========== Summary ==========")
    for k, v in report["summary"].items():
        print(f"{k}: {v}")

    payload = {
        "method": "moddrop++", "fold": int(a.fold), "split": a.split,
        "ckpt": str(a.ckpt), "mod_names": mod_names, "target_shape": list(target_shape),
        **report,
    }
    out_json = Path(a.out_json); out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print("\nSaved:", out_json)


if __name__ == "__main__":
    main()

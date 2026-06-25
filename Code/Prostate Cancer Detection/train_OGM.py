from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import json
import math
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import build_case_list_classification, Case3DClassificationDataset
from .model import ThreeModalityClassifier
from .utils import safe_auc, safe_ap


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
    return {"train": train_ids, "val": val_ids}


def _debug_match(train_ids: List[str], val_ids: List[str], discovered_ids: List[str]) -> None:
    train_set = set(train_ids)
    val_set = set(val_ids)
    disc_set = set(discovered_ids)

    inter_train = len(train_set & disc_set)
    inter_val = len(val_set & disc_set)

    print("\n[DEBUG MATCH]")
    print("  discovered:", len(disc_set))
    print("  train_ids :", len(train_set), " matched:", inter_train)
    print("  val_ids   :", len(val_set), " matched:", inter_val)

    print("  example split train ids:", train_ids[:5])
    print("  example discovered ids :", discovered_ids[:5])

    missing_train = list(train_set - disc_set)[:10]
    missing_val = list(val_set - disc_set)[:10]
    if missing_train:
        print("  example train ids NOT found in discovered:", missing_train[:5])
    if missing_val:
        print("  example val ids NOT found in discovered:", missing_val[:5])


# ============================================================
# OGM-GE (modality rebalancing) utilities
#   - OGM: downscale gradients of the dominant modality based on
#          per-modality "correct-class confidence" discrepancy ratio.
#   - GE : optionally inject Gaussian noise to restore gradient stochasticity.
#
# This implementation is "late-fusion friendly": we estimate per-modality
# confidence by running 3 extra unimodal forward passes (no grad).
# Gradient scaling is applied by multiplying parameter .grad for each branch.
# ============================================================
def _binary_correct_confidence(prob: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    prob: (B,1) sigmoid output
    y   : (B,1) in {0,1}
    returns s_i = p if y=1 else 1-p
    """
    y = y.float()
    return prob * y + (1.0 - prob) * (1.0 - y)


@torch.no_grad()
def estimate_modality_confidences_binary(
    model: nn.Module,
    xmods: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    y: torch.Tensor,
) -> torch.Tensor:
    """
    Returns per-modality batch-mean correct-class confidence S_m (m=0..2).
    We compute unimodal logits by feeding only one modality and zeroing others.
    """
    x0, x1, x2 = xmods
    zeros0 = torch.zeros_like(x0)
    zeros1 = torch.zeros_like(x1)
    zeros2 = torch.zeros_like(x2)

    logits = []
    logits.append(model((x0, zeros1, zeros2)))
    logits.append(model((zeros0, x1, zeros2)))
    logits.append(model((zeros0, zeros1, x2)))

    Ss = []
    for logit_m in logits:
        prob_m = torch.sigmoid(logit_m)
        s_m = _binary_correct_confidence(prob_m, y).view(-1)  # (B,)
        Ss.append(s_m.mean())

    return torch.stack(Ss, dim=0)  # (3,)


def ogm_discrepancy_ratio(S: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Generalized discrepancy ratio for >=2 modalities.
    Paper uses rho_v = S_v / S_a (2-mod case).
    Here we use rho_m = S_m / mean(S_all).
    """
    mu = S.mean().clamp_min(eps)
    return (S / mu).clamp_min(eps)  # (3,)


def ogm_kappa_from_rho(rho: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Paper (2-mod): k_t^u = 1 - tanh(alpha * rho_t^u) if rho>1 else 1.
    We keep the same shape for multi-mod.
    """
    k = torch.ones_like(rho)
    dominant = rho > 1.0
    k[dominant] = 1.0 - torch.tanh(alpha * rho[dominant])
    return k.clamp(0.0, 1.0)


def _split_params_by_prefix(
    model: nn.Module,
    modality_prefixes: List[str],
) -> List[List[nn.Parameter]]:
    """
    Groups parameters by name prefix (string startswith any prefix in modality_prefixes[m]).
    Example expected naming:
      enc0.*, enc1.*, enc2.*
    You can pass multiple comma-separated prefixes per modality (see CLI parsing below).
    """
    # modality_prefixes: length=3, each element like "enc0,branch0,encoder0"
    per_mod_prefixes = [tuple(p.strip() for p in s.split(",") if p.strip()) for s in modality_prefixes]

    groups: List[List[nn.Parameter]] = [[] for _ in range(3)]
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        for mi, prefs in enumerate(per_mod_prefixes):
            if len(prefs) == 0:
                continue
            if any(name.startswith(pref) for pref in prefs):
                groups[mi].append(p)
                break
    return groups


def apply_ogm_ge_to_grads(
    param_groups: List[List[nn.Parameter]],
    kappa: torch.Tensor,
    ge_std: float = 0.0,
    ge_enabled: bool = False,
    generator: Optional[torch.Generator] = None,
) -> None:
    """
    Multiply gradients of each modality branch by kappa[m].
    Optionally add Gaussian noise to branch grads (GE).
    """
    assert len(param_groups) == 3
    kappa = kappa.detach().view(-1).cpu()

    for m in range(3):
        km = float(kappa[m].item())
        if len(param_groups[m]) == 0:
            continue

        for p in param_groups[m]:
            if p.grad is None:
                continue

            # OGM: scale grad for dominant modality (km <= 1)
            p.grad.mul_(km)

            # GE: add noise to restore stochasticity (simple practical variant)
            if ge_enabled and ge_std > 0.0 and km < 1.0:
                # noise magnitude proportional to (1-km)
                noise_scale = ge_std * (1.0 - km)
                if noise_scale > 0:
                    noise = torch.randn_like(p.grad, generator=generator) * noise_scale
                    p.grad.add_(noise)


# ============================================================
# Eval
# ============================================================
@torch.no_grad()
def eval_one_epoch(model, loader, device) -> Dict[str, float]:
    model.eval()
    y_true: List[int] = []
    y_score: List[float] = []
    loss_sum = 0.0
    n = 0

    bce = nn.BCEWithLogitsLoss(reduction="mean")

    pbar = tqdm(loader, desc="[VAL]", leave=True)
    for batch in pbar:
        (x0, x1, x2), y, sid = batch
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
    return {"loss": loss_sum / max(1, n), "auc": auc, "ap": ap}


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
    base=32,
    feat_dim=256,
    dropout=0.1,
    fusion_hidden=256,
    # ---------------- OGM-GE knobs ----------------
    use_ogmge: bool = True,
    ogm_alpha: float = 0.1,
    ge_enabled: bool = False,
    ge_std: float = 0.02,
    # parameter-name prefixes for modality encoders (edit to match your model)
    # e.g., ["enc0", "enc1", "enc2"]  OR  ["encoder0,branch0", "encoder1,branch1", "encoder2,branch2"]
    modality_param_prefixes: Tuple[str, str, str] = ("enc0", "enc1", "enc2"),
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
        _debug_match(train_ids, val_ids, list(id_to_case.keys())[:2000])
        raise RuntimeError(
            "Empty train/val after matching splits to discovered cases.\n"
            "Expected split IDs to equal case['sid'] = patient_id_study_id (e.g., 10000_1000000).\n"
            "If discovered IDs differ, fix dataset.py label parsing to construct sid correctly."
        )

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
    model = ThreeModalityClassifier(
        base=base, feat_dim=feat_dim, dropout=dropout, fusion_hidden=fusion_hidden
    ).to(device)

    # Map parameters -> 3 modality groups (needed for OGM-GE scaling)
    param_groups = _split_params_by_prefix(model, list(modality_param_prefixes))
    if use_ogmge:
        print("[OGM-GE] enabled")
        print("  ogm_alpha:", ogm_alpha)
        print("  ge_enabled:", ge_enabled, "ge_std:", ge_std)
        print("  modality_param_prefixes:", modality_param_prefixes)
        print("  grouped params sizes:", [len(g) for g in param_groups])
        if sum(len(g) for g in param_groups) == 0:
            print(
                "  ⚠️  No params matched your prefixes. "
                "Update modality_param_prefixes to match model.named_parameters()."
            )

    pos_w = compute_pos_weight(train_cases)
    pos_w_t = torch.tensor([pos_w], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w_t, reduction="mean")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_auc = -1.0
    history: List[Dict[str, Any]] = []

    # For reproducible GE noise (optional)
    ge_gen = torch.Generator(device=device)
    ge_gen.manual_seed(1234 + fold)

    for ep in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        n = 0

        pbar = tqdm(train_ld, desc=f"[TRAIN] fold{fold} ep{ep}/{epochs}", leave=True)
        for batch in pbar:
            (x0, x1, x2), y, sid = batch
            x0 = x0.to(device, non_blocking=True)
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # ----- forward (main multimodal loss) -----
            logit = model((x0, x1, x2))
            loss = criterion(logit, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()

            # ----- OGM-GE: compute kappa then rescale modality branch grads -----
            ogm_info = None
            if use_ogmge:
                with torch.no_grad():
                    S = estimate_modality_confidences_binary(model, (x0, x1, x2), y)  # (3,)
                    rho = ogm_discrepancy_ratio(S)  # (3,)
                    kappa = ogm_kappa_from_rho(rho, alpha=ogm_alpha)  # (3,)

                apply_ogm_ge_to_grads(
                    param_groups=param_groups,
                    kappa=kappa,
                    ge_std=ge_std,
                    ge_enabled=ge_enabled,
                    generator=ge_gen,
                )
                ogm_info = {
                    "S": [float(v) for v in S.detach().cpu().tolist()],
                    "rho": [float(v) for v in rho.detach().cpu().tolist()],
                    "kappa": [float(v) for v in kappa.detach().cpu().tolist()],
                }

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            loss_sum += float(loss.item())
            n += 1

            if ogm_info is not None:
                pbar.set_postfix(
                    loss=f"{(loss_sum / max(1, n)):.4f}",
                    k0=f"{ogm_info['kappa'][0]:.2f}",
                    k1=f"{ogm_info['kappa'][1]:.2f}",
                    k2=f"{ogm_info['kappa'][2]:.2f}",
                )
            else:
                pbar.set_postfix(loss=f"{(loss_sum / max(1, n)):.4f}")

        va = eval_one_epoch(model, val_ld, device)

        row: Dict[str, Any] = {"epoch": ep, "train_loss": loss_sum / max(1, n), "val": va}
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
    # OGM-GE knobs
    use_ogmge: bool = True,
    ogm_alpha: float = 0.1,
    ge_enabled: bool = False,
    ge_std: float = 0.02,
    modality_param_prefixes: Tuple[str, str, str] = ("enc0", "enc1", "enc2"),
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
                use_ogmge=use_ogmge,
                ogm_alpha=ogm_alpha,
                ge_enabled=ge_enabled,
                ge_std=ge_std,
                modality_param_prefixes=modality_param_prefixes,
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
        "ogmge": {
            "use_ogmge": use_ogmge,
            "ogm_alpha": ogm_alpha,
            "ge_enabled": ge_enabled,
            "ge_std": ge_std,
            "modality_param_prefixes": list(modality_param_prefixes),
        },
    }
    summary_path = outdir / "cv_summary.json"
    json.dump(summary, open(summary_path, "w"), indent=2)

    print("\n========== 10-Fold Summary ==========")
    for i, s in enumerate(scores.tolist()):
        print(f"Fold {i}: best_auc={s:.4f}")
    print(f"MEAN best_auc: {mean:.4f}")
    print(f"STD  best_auc: {std:.4f}")
    print("Saved:", summary_path)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, default=None)
    p.add_argument("--images", required=True)
    p.add_argument("--labels_csv", required=True)
    p.add_argument("--splits", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--tZ", type=int, default=128)
    p.add_argument("--tY", type=int, default=192)
    p.add_argument("--tX", type=int, default=192)

    # -------- OGM-GE flags --------
    p.add_argument("--use_ogmge", action="store_true", help="Enable OGM-GE modality rebalancing.")
    p.add_argument("--ogm_alpha", type=float, default=0.1, help="OGM alpha (strength of suppression).")
    p.add_argument("--ge_enabled", action="store_true", help="Enable GE (Gaussian noise enhancement).")
    p.add_argument("--ge_std", type=float, default=0.02, help="Std multiplier for GE noise (practical variant).")
    p.add_argument(
        "--modality_prefixes",
        type=str,
        default="net0;net1;net2",
        help=(
            "Three semicolon-separated prefix lists (commas allowed per modality). "
            "Example: 'enc0,branch0;enc1,branch1;enc2,branch2'"
        ),
    )

    args = p.parse_args()

    target_shape = (args.tZ, args.tY, args.tX)

    # Parse prefixes
    parts = [s.strip() for s in args.modality_prefixes.split(";")]
    if len(parts) != 3:
        raise ValueError("--modality_prefixes must have exactly 3 ';'-separated fields.")
    modality_param_prefixes = (parts[0], parts[1], parts[2])

    if args.fold is None:
        run_10fold_and_summarize(
            images_root=Path(args.images),
            labels_csv=Path(args.labels_csv),
            splits_json=Path(args.splits),
            outdir=Path(args.outdir),
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            num_workers=args.num_workers,
            target_shape=target_shape,
            use_ogmge=args.use_ogmge,
            ogm_alpha=args.ogm_alpha,
            ge_enabled=args.ge_enabled,
            ge_std=args.ge_std,
            modality_param_prefixes=modality_param_prefixes,
        )
    else:
        train_fold(
            images_root=Path(args.images),
            labels_csv=Path(args.labels_csv),
            splits_json=Path(args.splits),
            outdir=Path(args.outdir),
            fold=args.fold,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            num_workers=args.num_workers,
            target_shape=target_shape,
            use_ogmge=args.use_ogmge,
            ogm_alpha=args.ogm_alpha,
            ge_enabled=args.ge_enabled,
            ge_std=args.ge_std,
            modality_param_prefixes=modality_param_prefixes,
        )
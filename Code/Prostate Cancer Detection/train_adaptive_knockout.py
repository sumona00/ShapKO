from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Tuple
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
# Base KO rate formula (same as your previous codebase)
# ============================================================
def compute_base_knockout_rate(num_modalities: int) -> float:
    """
    r = 1 - (0.5 ** (1/d)) so expected keep prob ~0.5 when d modalities exist.
    """
    d = max(1, int(num_modalities))
    return float(1.0 - (0.5 ** (1.0 / d)))


# ============================================================
# Shapley utilities (binary task, utility = AUC by default)
# ============================================================
def _bitcount(x: int) -> int:
    return int(bin(x).count("1"))


def dropped_to_keep_mask(dropped: List[int], M: int) -> int:
    all_mask = (1 << M) - 1
    drop_mask = 0
    for i in dropped:
        drop_mask |= (1 << int(i))
    return int(all_mask & (~drop_mask))


def normalize_nonneg(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - float(np.min(x))
    x = np.maximum(x, 0.0)
    s = float(x.sum())
    if (not np.isfinite(s)) or s <= 1e-12:
        return np.ones_like(x) / float(len(x))
    return x / s


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


def shapley_from_utilities(utilities: Dict[int, float], M: int) -> np.ndarray:
    """
    Standard Shapley:
      phi_i = sum_S w(S) [u(S∪{i}) - u(S)]
      w(S) = |S|!(M-|S|-1)! / M!
    utilities: map keep_mask(int) -> utility(float)
    """
    fact = [math.factorial(k) for k in range(M + 1)]
    denom = float(fact[M]) + 1e-12
    phi = np.zeros((M,), dtype=np.float64)

    for S in range(0, 1 << M):
        utilities.setdefault(S, 0.0)

    for i in range(M):
        for S, uS in utilities.items():
            if (S >> i) & 1:
                continue
            S2 = S | (1 << i)
            uS2 = utilities.get(S2, 0.0)
            k = _bitcount(S)
            w = (fact[k] * fact[M - k - 1]) / denom
            phi[i] += w * (uS2 - uS)
    return phi


# ============================================================
# Shapley-based Adaptive KO wrapper (KO starts after ko_start_epoch)
# ============================================================
class ShapleyAdaptiveKnockoutWrapper(nn.Module):
    """
    Shapley-based adaptive knockout for 3 modalities (x0, x1, x2).

    KO application:
      - only in training mode
      - only when current_epoch >= ko_start_epoch

    Presence flag:
      present[b] = sum(abs(x[b])) > 0
    (dataset encodes observed missingness as all-zero tensors)

    Placeholders:
      observed missing placeholder = 0
      knockout placeholder         = 0
    => both observed-missing and KO are zeros.
    """

    def __init__(
        self,
        base_model: nn.Module,
        num_modalities: int = 3,
        ko_min: float = 0.02,
        ko_max: float = 0.60,
        delta_importance: float = 1.0,
        drop_strong_more: bool = True,
        keep_at_least_one: bool = True,
        ko_start_epoch: int = 20,   # <-- KO starts at/after this epoch
    ):
        super().__init__()
        self.model = base_model

        self.M = int(num_modalities)
        assert self.M == 3, "This wrapper is written for 3 modalities as in your code."

        self.base_r = compute_base_knockout_rate(self.M)
        self.ko_min = float(ko_min)
        self.ko_max = float(ko_max)
        self.delta = float(delta_importance)
        self.drop_strong_more = bool(drop_strong_more)
        self.keep_at_least_one = bool(keep_at_least_one)

        # scheduling
        self.ko_start_epoch = int(ko_start_epoch)
        self.current_epoch = 0

        # current KO rates
        self.r = np.array([self.base_r] * self.M, dtype=np.float64)

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    @torch.no_grad()
    def _is_present(self, x: torch.Tensor) -> torch.Tensor:
        dims = tuple(range(1, x.ndim))
        return (x.abs().sum(dim=dims) > 0)

    @staticmethod
    def _clampf(x: float, lo: float, hi: float) -> float:
        return float(max(lo, min(hi, x)))

    def update_from_shapley(self, phi: np.ndarray) -> None:
        phi = np.asarray(phi, dtype=np.float64).reshape(-1)
        if phi.shape[0] != self.M:
            raise ValueError(f"phi must have shape ({self.M},) but got {phi.shape}")

        imp = normalize_nonneg(phi)
        mean_imp = float(np.mean(imp) + 1e-12)
        new_r = np.zeros_like(imp)

        for i in range(self.M):
            if self.drop_strong_more:
                factor = (imp[i] / mean_imp) ** self.delta
            else:
                factor = (mean_imp / (imp[i] + 1e-12)) ** self.delta
            new_r[i] = self._clampf(self.base_r * factor, self.ko_min, self.ko_max)

        self.r = new_r

    @torch.no_grad()
    def _apply_adaptive_knockout(
        self,
        xs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x0, x1, x2 = xs

        # No KO in eval mode
        if not self.training:
            return x0, x1, x2

        # KO starts only after ko_start_epoch
        if self.current_epoch < self.ko_start_epoch:
            return x0, x1, x2

        B = x0.shape[0]
        device = x0.device

        pres0 = self._is_present(x0)
        pres1 = self._is_present(x1)
        pres2 = self._is_present(x2)
        present = torch.stack([pres0, pres1, pres2], dim=1)  # (B,3)

        r_t = torch.tensor(self.r, device=device, dtype=torch.float32).view(1, 3)
        u = torch.rand((B, 3), device=device)
        drop = (u < r_t) & present

        if self.keep_at_least_one:
            present_count = present.long().sum(dim=1)
            dropped_count = drop.long().sum(dim=1)
            all_knocked = (present_count > 0) & (dropped_count >= present_count)

            if all_knocked.any():
                idx = torch.where(all_knocked)[0]
                for b in idx.tolist():
                    candidates: List[int] = []
                    if pres0[b].item(): candidates.append(0)
                    if pres1[b].item(): candidates.append(1)
                    if pres2[b].item(): candidates.append(2)
                    if len(candidates) > 0:
                        keep_mod = candidates[int(torch.randint(0, len(candidates), (1,), device=device).item())]
                        drop[b, keep_mod] = False

        def zero_out(x: torch.Tensor, drop_b: torch.Tensor) -> torch.Tensor:
            mask = (~drop_b).to(dtype=x.dtype, device=x.device)
            view = (x.shape[0],) + (1,) * (x.ndim - 1)
            return x * mask.view(view)

        x0k = zero_out(x0, drop[:, 0])
        x1k = zero_out(x1, drop[:, 1])
        x2k = zero_out(x2, drop[:, 2])
        return x0k, x1k, x2k

    def forward(self, xs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        x0, x1, x2 = xs
        x0, x1, x2 = self._apply_adaptive_knockout((x0, x1, x2))
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


@torch.no_grad()
def eval_one_epoch_on_loader_with_combo(
    base_model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate the BASE model on the given loader (which already encodes a modality drop combo),
    WITHOUT applying train-time KO.
    Utility here is AUC (primary) and AP (secondary).
    """
    base_model.eval()
    y_true: List[int] = []
    y_score: List[float] = []
    loss_sum = 0.0
    n = 0

    bce = nn.BCEWithLogitsLoss(reduction="mean")

    pbar = tqdm(loader, desc="[VAL combo]", leave=False)
    for batch in pbar:
        (x0, x1, x2), y, sid = batch
        x0 = x0.to(device, non_blocking=True)
        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logit = base_model((x0, x1, x2))
        loss = bce(logit, y)

        prob = torch.sigmoid(logit).view(-1).detach().cpu().numpy().tolist()
        yt = y.view(-1).detach().cpu().numpy().astype(int).tolist()

        y_score.extend(prob)
        y_true.extend(yt)

        loss_sum += float(loss.item())
        n += 1

    auc = safe_auc(y_true, y_score)
    ap = safe_ap(y_true, y_score)
    return {"loss": loss_sum / max(1, n), "auc": auc, "ap": ap}


def build_shapley_utilities_auc(
    base_model: nn.Module,
    val_ds: torch.utils.data.Dataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> Dict[int, float]:
    """
    Compute utilities u(S) for all subsets S of modalities (M=3).
    Utility = validation AUC when only modalities in S are available
    (dropped modalities are zeroed by DropModalitiesWrapper).
    """
    combos = [
        [], [0], [1], [2],
        [0, 1], [0, 2], [1, 2],
        [0, 1, 2],
    ]

    utilities: Dict[int, float] = {}
    for dropped in combos:
        ds_combo = DropModalitiesWrapper(val_ds, dropped=dropped)
        ld = DataLoader(
            ds_combo,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )

        va = eval_one_epoch_on_loader_with_combo(base_model, ld, device=device)
        keep_mask = dropped_to_keep_mask(dropped, M=3)
        u = float(va["auc"]) if (va["auc"] is not None and not math.isnan(va["auc"])) else 0.0
        utilities[keep_mask] = u

    for S in range(0, 1 << 3):
        utilities.setdefault(S, 0.0)
    return utilities


# ============================================================
# Train
# ============================================================
def train_fold(
    images_root: Path,
    labels_csv: Path,
    splits_json: Path,
    outdir: Path,
    fold: int = 0,
    epochs: int = 200,
    batch_size: int = 2,
    lr: float = 3e-4,
    num_workers: int = 4,
    target_shape=(128, 192, 192),
    base=32,
    feat_dim=256,
    dropout=0.1,
    fusion_hidden=256,
    # ---- KO schedule ----
    ko_start_epoch: int = 20,          # <-- KO starts after 20 epochs
    # ---- Shapley Adaptive KO knobs ----
    ko_min: float = 0.02,
    ko_max: float = 0.30,
    ko_delta_importance: float = 0.5,
    drop_strong_more: bool = True,
    keep_at_least_one: bool = True,
    shapley_start_epoch: int = 50,     # <-- Shapley starts after 50 epochs
    shapley_update_every: int = 15,
) -> Dict[str, Any]:
    outdir.mkdir(parents=True, exist_ok=True)
    fold_dir = outdir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    best_ckpt_path = fold_dir / "best.pt"
    last_ckpt_path = fold_dir / "last.pt"
    history_path = fold_dir / "history.json"

    # Discover all labeled cases.
    all_cases = build_case_list_classification(
        images_root=images_root,
        labels_csv=labels_csv,
        strict=True,
        verbose=True,
    )

    # Key by sid (patient_id_study_id) to match splits_10fold.json exactly
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

    # Dataset returns sid as the 3rd element
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

    model = ShapleyAdaptiveKnockoutWrapper(
        base_model=base_model,
        num_modalities=3,
        ko_min=ko_min,
        ko_max=ko_max,
        delta_importance=ko_delta_importance,
        drop_strong_more=drop_strong_more,
        keep_at_least_one=keep_at_least_one,
        ko_start_epoch=ko_start_epoch,
    ).to(device)

    print(
        f"[KO] ko_start_epoch={ko_start_epoch} | base_r={model.base_r:.6f} init r={model.r.tolist()} "
        f"| ko_min={ko_min} ko_max={ko_max} delta={ko_delta_importance} "
        f"| drop_strong_more={drop_strong_more} keep_at_least_one={keep_at_least_one} "
        f"| placeholders: observed=0 knockout=0"
    )
    print(f"[Shapley] shapley_start_epoch={shapley_start_epoch} update_every={shapley_update_every}")

    pos_w = compute_pos_weight(train_cases)
    pos_w_t = torch.tensor([pos_w], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w_t, reduction="mean")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_auc = -1.0
    history: List[Dict[str, Any]] = []

    for ep in range(1, epochs + 1):
        model.train()
        model.set_epoch(ep)  # <-- IMPORTANT: enables KO only when ep >= ko_start_epoch

        loss_sum = 0.0
        n = 0

        pbar = tqdm(train_ld, desc=f"[TRAIN] fold{fold} ep{ep}/{epochs}", leave=True)
        for batch in pbar:
            (x0, x1, x2), y, sid = batch
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
            pbar.set_postfix(loss=f"{(loss_sum / max(1, n)):.4f}", ko=("ON" if ep >= ko_start_epoch else "OFF"))

        # normal validation (no KO because eval())
        va = eval_one_epoch(model, val_ld, device)

        # ---- Shapley update (starts at shapley_start_epoch) ----
        if ep >= shapley_start_epoch and ((ep - shapley_start_epoch) % max(1, int(shapley_update_every)) == 0):
            utilities = build_shapley_utilities_auc(
                base_model=base_model,  # IMPORTANT: base model, not KO-wrapped forward
                val_ds=val_ds,
                batch_size=batch_size,
                num_workers=num_workers,
                device=device,
            )
            phi = shapley_from_utilities(utilities, M=3)
            model.update_from_shapley(phi)
            print(f"  [Shapley@ep{ep}] phi={phi.tolist()}  => r={model.r.tolist()}")
        else:
            if ep >= shapley_start_epoch:
                print(f"  [Shapley] skip (update_every={shapley_update_every})")

        row = {
            "epoch": ep,
            "train_loss": loss_sum / max(1, n),
            "val": va,
            "ko_r": model.r.tolist(),
            "ko_active": bool(ep >= ko_start_epoch),
            "ko_start_epoch": int(ko_start_epoch),
            "shapley_start_epoch": int(shapley_start_epoch),
            "shapley_update_every": int(shapley_update_every),
        }
        history.append(row)

        print(
            f"[EPOCH {ep:03d}] KO={'ON' if ep >= ko_start_epoch else 'OFF'} "
            f"| train_loss={row['train_loss']:.4f} "
            f"| val_loss={va['loss']:.4f} auc={va['auc']:.4f} ap={va['ap']:.4f} "
            f"| ko_r={model.r.tolist()}"
        )

        # save last
        torch.save(
            {"model": model.state_dict(), "epoch": ep, "fold": fold, "val": va, "ko_r": model.r.tolist()},
            last_ckpt_path,
        )

        # save best by AUC
        if not math.isnan(va["auc"]) and va["auc"] > best_auc:
            best_auc = va["auc"]
            torch.save(
                {"model": model.state_dict(), "epoch": ep, "fold": fold, "best_auc": best_auc, "val": va, "ko_r": model.r.tolist()},
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
    epochs: int = 200,
    batch_size: int = 2,
    lr: float = 3e-4,
    num_workers: int = 4,
    target_shape=(128, 192, 192),
    # KO schedule
    ko_start_epoch: int = 20,
    # KO/Shapley knobs
    ko_min: float = 0.02,
    ko_max: float = 0.30,
    ko_delta_importance: float = 0.5,
    drop_strong_more: bool = True,
    keep_at_least_one: bool = True,
    shapley_start_epoch: int = 50,
    shapley_update_every: int = 15,
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
                ko_start_epoch=ko_start_epoch,
                ko_min=ko_min,
                ko_max=ko_max,
                ko_delta_importance=ko_delta_importance,
                drop_strong_more=drop_strong_more,
                keep_at_least_one=keep_at_least_one,
                shapley_start_epoch=shapley_start_epoch,
                shapley_update_every=shapley_update_every,
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
        "ko_start_epoch": ko_start_epoch,
        "ko_min": ko_min,
        "ko_max": ko_max,
        "ko_delta_importance": ko_delta_importance,
        "drop_strong_more": drop_strong_more,
        "shapley_start_epoch": shapley_start_epoch,
        "shapley_update_every": shapley_update_every,
        "formula_default_base_r": compute_base_knockout_rate(3),
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
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--tZ", type=int, default=128)
    p.add_argument("--tY", type=int, default=192)
    p.add_argument("--tX", type=int, default=192)

    # NEW: KO schedule
    p.add_argument("--ko_start_epoch", type=int, default=5, help="Start applying KO at/after this epoch.")

    # Shapley-adaptive KO args20
    p.add_argument("--ko_min", type=float, default=0.02)
    p.add_argument("--ko_max", type=float, default=0.30)
    p.add_argument("--ko_delta", type=float, default=0.5)
    p.add_argument("--drop_strong_more", action="store_true",
                   help="If set, drop more important modalities more.")
    p.add_argument("--drop_strong_less", action="store_true",
                   help="If set, drop important modalities less (inverse weighting).")
    p.add_argument("--keep_at_least_one", action="store_true",
                   help="Ensure at least one present modality remains (recommended).")

    # Shapley schedule (requested: start after 50 epochs)
    p.add_argument("--shapley_start_epoch", type=int, default=50)
    p.add_argument("--shapley_update_every", type=int, default=15)

    args = p.parse_args()
    target_shape = (args.tZ, args.tY, args.tX)

    drop_strong_more = True
    if args.drop_strong_less:
        drop_strong_more = False
    if args.drop_strong_more:
        drop_strong_more = True

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
            ko_start_epoch=int(args.ko_start_epoch),
            ko_min=args.ko_min,
            ko_max=args.ko_max,
            ko_delta_importance=args.ko_delta,
            drop_strong_more=drop_strong_more,
            keep_at_least_one=bool(args.keep_at_least_one),
            shapley_start_epoch=int(args.shapley_start_epoch),
            shapley_update_every=int(args.shapley_update_every),
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
            ko_start_epoch=int(args.ko_start_epoch),
            ko_min=args.ko_min,
            ko_max=args.ko_max,
            ko_delta_importance=args.ko_delta,
            drop_strong_more=drop_strong_more,
            keep_at_least_one=bool(args.keep_at_least_one),
            shapley_start_epoch=int(args.shapley_start_epoch),
            shapley_update_every=int(args.shapley_update_every),
        )

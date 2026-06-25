# SHAPE_v_acc.py  (accuracy-based Shapley + full combo metrics on val & test)
from __future__ import annotations
import os, sys, math, datetime, random, argparse, io
from os import path as osp
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# TensorBoard (use tensorboardX to avoid distutils issues)
from tensorboardX import SummaryWriter

# ---- project imports ----
from utils import Discretizer, Normalizer, my_metrics, is_ascending
from dataset.dataloader import get_multimodal_datasets
from mymodel.model_v2 import FlexCare

# =========================
# Argparse (extend existing)
# =========================
from arguments import args_parser
parser = args_parser()
args, extras = parser.parse_known_args()

# Extra args: Shapley weighting + early stopping + TB
ko = argparse.ArgumentParser(add_help=False)

# Early stopping: we monitor ALL modality combos (LOSS-based)
ko.add_argument(
    '--early_stop_patience', type=int, default=5,
    help='Number of epochs with no improvement (for any combo) before early stopping.'
)
ko.add_argument(
    '--early_stop_min_delta', type=float, default=0.0,
    help='Minimum change to qualify as an improvement.'
)

# TensorBoard
ko.add_argument(
    '--tb_log_dir', type=str, default='tb_logs',
    help='Base directory (or run dir) for TensorBoard logs.'
)

# Shapley-based loss weighting (NO effect on knockout/masks)
ko.add_argument(
    '--shapley_alpha', type=float, default=1.0,
    help='Exponent applied to Shapley scores before normalization (smooth importance).'
)
ko.add_argument(
    '--shapley_start_epoch', type=int, default=2,
    help='Epoch from which Shapley-based loss weighting starts (before that: uniform).'
)
ko.add_argument(
    '--shapley_weight_scale', type=float, default=1.0,
    help='Exponent applied when inverting Shapley importance to rebalancing weights.'
)

ko.add_argument(
    '--weight_decay', type=float, default=1e-2,
    help='Weight decay for AdamW optimizer.'
)

ko_args, _ = ko.parse_known_args(extras)
for k, v in vars(ko_args).items():
    setattr(args, k, v)

os.environ['CUDA_VISIBLE_DEVICES'] = str(getattr(args, 'device', 0))

# =========================
# Logging
# =========================
import logging
from logging.handlers import RotatingFileHandler


class Tee(io.TextIOBase):
    def __init__(self, stream, logger, level=logging.INFO):
        self.stream, self.logger, self.level, self._buffer = stream, logger, level, ''

    def write(self, buf):
        self.stream.write(buf)
        self.stream.flush()
        self._buffer += buf
        while '\n' in self._buffer:
            line, self._buffer = self._buffer.split('\n', 1)
            if line.strip():
                self.logger.log(self.level, line)
        return len(buf)

    def flush(self):
        self.stream.flush()
        if self._buffer.strip():
            self.logger.log(self.level, self._buffer.strip())
            self._buffer = ''


def setup_logging(args):
    os.makedirs('log', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    task_tag = '-'.join((args.task if isinstance(args.task, list) else [args.task]))
    model_tag = getattr(args, 'model', 'model')
    log_path = os.path.join(
        'log',
        f"[{model_tag}]_lr{args.lr}_seed{args.seed}_ep{args.epochs}_{task_tag}_{stamp}.log"
    )
    logger = logging.getLogger('flexcare')
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    fh = RotatingFileHandler(log_path, maxBytes=50_000_000, backupCount=1)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.__stdout__)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    sys.stdout = Tee(sys.__stdout__, logger, level=logging.INFO)
    sys.stderr = Tee(sys.__stderr__, logger, level=logging.ERROR)
    return logger, log_path, dict(file=sys.stdout)


# =========================
# Plotting
# =========================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_combo_losses(combo_history, out_path):
    plt.figure(figsize=(10, 6))
    for name, vals in combo_history.items():
        if len(vals) == 0:
            continue
        epochs = list(range(1, len(vals) + 1))
        plt.plot(epochs, vals, label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Validation loss")
    plt.title("Validation loss per modality combination")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_shapley_and_rebal(
    shapley_history: dict,
    rebal_history: dict,
    knockout_rate: float,
    out_dir: str
):
    """
    Plot:
      1) Shapley importance for each modality vs. epoch, with a horizontal
         line at the knockout rate r.
      2) Rebalancing weights for each modality vs. epoch, also with r.
    """
    os.makedirs(out_dir, exist_ok=True)
    # assume all lists have same length
    epochs = list(range(1, len(next(iter(shapley_history.values()))) + 1))

    # ---- Shapley vs knockout ----
    plt.figure(figsize=(10, 6))
    for m in ['ehr', 'cxr', 'note']:
        plt.plot(epochs, shapley_history[m], label=f"Shapley {m}")
    plt.axhline(y=knockout_rate, linestyle='--', label=f"KO rate={knockout_rate:.2f}")
    plt.xlabel("Epoch")
    plt.ylabel("Shapley importance")
    plt.title("Per-modality Shapley importance vs. knockout rate")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    shap_path = os.path.join(out_dir, "shapley_vs_knockout.png")
    plt.savefig(shap_path)
    plt.close()

    # ---- Rebal weights vs knockout ----
    plt.figure(figsize=(10, 6))
    for m in ['ehr', 'cxr', 'note']:
        plt.plot(epochs, rebal_history[m], label=f"Rebal {m}")
    plt.axhline(y=knockout_rate, linestyle='--', label=f"KO rate={knockout_rate:.2f}")
    plt.xlabel("Epoch")
    plt.ylabel("Rebalancing weight")
    plt.title("Per-modality rebalancing weights vs. knockout rate")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    rebal_path = os.path.join(out_dir, "rebal_vs_knockout.png")
    plt.savefig(rebal_path)
    plt.close()

    return shap_path, rebal_path


# =========================
# Helpers
# =========================

def _ensure_metric_shapes(task_now, yt, yp):
    """
    Ensure shapes match what utils.my_metrics expects.

    - For CE tasks (length-of-stay, drg):
        yt: (N,) long; yp: (N,) long indices.
    - For BCE/multi-label tasks:
        yt: (N,L); yp: (N,L) probabilities.
      my_metrics is assumed to handle possible -1 in yt internally.
    """
    ce_tasks = {'length-of-stay', 'drg'}
    if task_now in ce_tasks:
        return yt.long().view(-1), yp.long().view(-1)
    if yt.dim() == 1:
        yt = yt.unsqueeze(1)
    if yp.dim() == 1:
        yp = yp.unsqueeze(1)
    return yt, yp


def compute_accuracy(yt: torch.Tensor, yp: torch.Tensor, task_now: str) -> float:
    """
    Compute accuracy for:
      - CE tasks: yt, yp are (N,) class indices; ignore yt == -1.
      - BCE/multi-label tasks: yt, yp are (N,L) or (N,); we threshold yp at 0.5
        and ignore yt < 0 (missing labels). Accuracy is element-wise.
    """
    ce_tasks = {'length-of-stay', 'drg'}
    with torch.no_grad():
        if task_now in ce_tasks:
            # yt, yp: (N,)
            yt = yt.view(-1)
            yp = yp.view(-1)
            mask = (yt >= 0)
            if mask.sum() == 0:
                return float('nan')
            yt_sel = yt[mask]
            yp_sel = yp[mask]
            acc = (yp_sel == yt_sel).float().mean().item()
            return float(acc)

        # BCE / multi-label
        if yt.dim() > 1:
            yt_flat = yt.view(-1)
            yp_flat = yp.view(-1)
        else:
            yt_flat = yt.view(-1)
            yp_flat = yp.view(-1)

        mask = (yt_flat >= 0)
        if mask.sum() == 0:
            return float('nan')

        yt_sel = yt_flat[mask]
        yp_sel = yp_flat[mask]

        # yt is {0,1} (or -1), yp is in [0,1] (probabilities)
        yt_bin = (yt_sel > 0.5).float()
        yp_bin = (yp_sel >= 0.5).float()

        acc = (yp_bin == yt_bin).float().mean().item()
        return float(acc)


def compute_feature_knockout_rate(num_modalities: int) -> float:
    """
    From (1 - r)^d = 0.5  =>  r = 1 - 0.5^(1/d)
    So that the probability all d modalities are kept is 0.5.
    """
    d = max(1, int(num_modalities))
    return 1.0 - (0.5 ** (1.0 / d))


def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


# -------------------------
# BCE-with-logits helper: ignore -1 labels
# -------------------------
import torch.nn.functional as F


def bce_with_logits_ignore_neg(
    logits: torch.Tensor,
    y_true: torch.Tensor,
    reduction: str = "mean"
) -> torch.Tensor:
    """
    BCE *with logits* that ignores entries where y_true < 0 (missing labels).

    - logits: arbitrary real values (from linear head)
    - y_true: same shape; entries <0 are treated as missing
    - For reduction='none', returns tensor same shape as y_true with 0 where missing.
    - For reduction='mean', returns scalar averaged over non-missing entries.
    """
    device = logits.device
    mask = (y_true >= 0)

    if mask.sum() == 0:
        if reduction == "none":
            return torch.zeros_like(y_true, dtype=torch.float, device=device)
        return torch.zeros((), device=device)

    if not torch.isfinite(logits).all():
        raise RuntimeError("Non-finite logits passed to BCEWithLogitsLoss")

    # clamp targets to [0,1] for safety
    y_true_clamped = torch.clamp(y_true.float(), 0.0, 1.0)

    if not torch.isfinite(y_true_clamped).all():
        raise RuntimeError("Non-finite targets passed to BCEWithLogitsLoss")

    loss_raw = F.binary_cross_entropy_with_logits(
        logits, y_true_clamped, reduction="none"
    )

    loss_raw = loss_raw * mask.float()

    if reduction == "none":
        return loss_raw

    return loss_raw.sum() / mask.float().sum().clamp_min(1.0)


# ====================================================
# Shapley on modality COMBOS
# ====================================================

def losses_to_utilities(epoch_combo_avg: dict) -> dict:
    """
    (Kept for backwards compatibility, but not used anymore for Shapley.)

    Convert per-combo average losses L(S) into utilities f_u(S).

    Conceptually: for cross-entropy/BCE, lower loss ≈ higher
    mutual information I(S;Y). We approximate a "utility" by

        f_u(S) = L_const - L(S)

    where L_const is the worst (max) loss across combos in
    this epoch. So utilities are >= 0, and monotone in "better loss".
    """
    finite_losses = [v for v in epoch_combo_avg.values() if math.isfinite(v)]
    if len(finite_losses) == 0:
        return {k: 0.0 for k in epoch_combo_avg}

    L_const = max(finite_losses)
    utilities = {}
    for name, loss in epoch_combo_avg.items():
        if not math.isfinite(loss):
            utilities[name] = 0.0
        else:
            utilities[name] = max(0.0, L_const - loss)
    return utilities


def metrics_to_utilities_acc(
    global_combo_acc_sum: dict,
    global_combo_metric_task_count: dict,
) -> dict:
    """
    Build SHAPE-style utilities v(S) directly from validation performance
    using **accuracy** for each modality subset S.

    For each combo, we use the mean ACC across tasks as the "utility":
        v(S) ≈ mean accuracy when that subset S is available.
    """
    utilities = {}
    for name in global_combo_acc_sum.keys():
        count = global_combo_metric_task_count.get(name, 0)
        if count > 0:
            mean_acc = global_combo_acc_sum[name] / count
            utilities[name] = max(0.0, float(mean_acc))
        else:
            utilities[name] = 0.0
    return utilities


def compute_shapley_importance_from_utilities(
    combo_util: dict,
    alpha: float = 1.0,
):
    """
    Estimate per-modality Shapley *importance* from utilities v(S),
    following the SHAPE-style 3-player Shapley formula.

    combo_util: dict with utility v(S) for the 7 combos:
        nat, ehr_only, cxr_only, note_only,
        ehr_cxr, ehr_note, cxr_note

    We treat v(S) as SHAPE's V_f(S): performance when subset S is available.
    Higher Shapley = more useful modality under current model.

    Returns:
        importance: dict { 'ehr': w_e, 'cxr': w_c, 'note': w_n },
        w_* >=0, sum=1.
    """
    # utilities for each subset
    U_nat = combo_util.get("nat", 0.0)        # v({e,c,n})
    U_e   = combo_util.get("ehr_only", 0.0)   # v({e})
    U_c   = combo_util.get("cxr_only", 0.0)   # v({c})
    U_n   = combo_util.get("note_only", 0.0)  # v({n})
    U_ec  = combo_util.get("ehr_cxr", 0.0)    # v({e,c})
    U_en  = combo_util.get("ehr_note", 0.0)   # v({e,n})
    U_cn  = combo_util.get("cxr_note", 0.0)   # v({c,n})

    # value function v(S)
    v_nat, v_e, v_c, v_n = U_nat, U_e, U_c, U_n
    v_ec, v_en, v_cn     = U_ec, U_en, U_cn

    # 3-player Shapley (ehr=e, cxr=c, note=n), with empty-set utility v(∅) ≈ 0
    v_0 = 0.0

    # --- ehr ---
    phi_e = (
        (1.0 / 3.0) * (v_e  - v_0) +
        (1.0 / 6.0) * (v_ec - v_c) +
        (1.0 / 6.0) * (v_en - v_n) +
        (1.0 / 3.0) * (v_nat - v_cn)
    )

    # --- cxr ---
    phi_c = (
        (1.0 / 3.0) * (v_c  - v_0) +
        (1.0 / 6.0) * (v_ec - v_e) +
        (1.0 / 6.0) * (v_cn - v_n) +
        (1.0 / 3.0) * (v_nat - v_en)
    )

    # --- note ---
    phi_n = (
        (1.0 / 3.0) * (v_n  - v_0) +
        (1.0 / 6.0) * (v_en - v_e) +
        (1.0 / 6.0) * (v_cn - v_c) +
        (1.0 / 3.0) * (v_nat - v_ec)
    )

    phi_arr = np.array([phi_e, phi_c, phi_n], dtype=np.float64)

    # shift so min=0, clamp negative to 0 (keep relative importance only)
    phi_arr = phi_arr - float(phi_arr.min())
    phi_arr = np.maximum(phi_arr, 0.0)

    if not np.isfinite(phi_arr).all() or phi_arr.sum() <= 0:
        # fallback to uniform importance
        return {'ehr': 1.0 / 3.0, 'cxr': 1.0 / 3.0, 'note': 1.0 / 3.0}

    # smooth / sharpen with alpha
    alpha = max(0.1, float(alpha))
    phi_arr = phi_arr ** alpha

    s = float(phi_arr.sum())
    phi_arr = phi_arr / s

    return {
        'ehr': float(phi_arr[0]),
        'cxr': float(phi_arr[1]),
        'note': float(phi_arr[2]),
    }


def compute_sample_rebalancing_weights(
    mask_ehr_t: torch.Tensor,
    mask_cxr_t: torch.Tensor,
    mask_note_t: torch.Tensor,
    rebal_weights: dict,
) -> torch.Tensor:
    """
    Compute per-sample scalar weights in [~0.5, ~2.0] based on modality presence
    and modality-level rebalancing weights.

    Args:
        mask_ehr_t, mask_cxr_t, mask_note_t: (B,) 0/1
        rebal_weights: dict {'ehr': w_e, 'cxr': w_c, 'note': w_n}, sum w_m = 1,
                       higher = more *upweight* when that modality is present.

    Returns:
        weights: (B,) float, mean ≈ 1.0, clamped to [0.5, 2.0]
    """
    device = mask_ehr_t.device
    me = mask_ehr_t.float()
    mc = mask_cxr_t.float()
    mn = mask_note_t.float()

    w_e = float(rebal_weights.get('ehr', 1.0 / 3.0))
    w_c = float(rebal_weights.get('cxr', 1.0 / 3.0))
    w_n = float(rebal_weights.get('note', 1.0 / 3.0))

    raw = me * w_e + mc * w_c + mn * w_n
    present = me + mc + mn

    # If no modality present (rare), fallback to 1.0
    raw = torch.where(
        present > 0,
        raw / present.clamp_min(1.0),  # average of weights for present modalities
        torch.ones_like(raw),
    )

    # normalize mean to ~1.0 so we don't change global LR
    mean_w = raw.mean().clamp_min(1e-6)
    raw = raw / mean_w

    # clamp to keep training stable
    raw = raw.clamp(0.5, 2.0)
    return raw


# =========================
# Collate / shapes
# =========================

def pad_zeros(arr, min_length=None):
    dtype = arr[0].dtype
    seq_length = [x.shape[0] for x in arr]
    max_len = max(seq_length)
    ret = [
        np.concatenate(
            [x, np.zeros((max_len - x.shape[0],) + x.shape[1:], dtype=dtype)],
            axis=0
        )
        for x in arr
    ]
    if (min_length is not None) and ret[0].shape[0] < min_length:
        ret = [
            np.concatenate(
                [x, np.zeros((min_length - x.shape[0],) + x.shape[1:], dtype=dtype)],
                axis=0
            )
            for x in ret
        ]
    return np.array(ret), seq_length


def my_collate(batch):
    # EHR: [T,F]; when missing, [1,76] zeros
    ehr = [
        item[0][-512:] if item[0] is not None else np.zeros((1, 76), dtype=np.float32)
        for item in batch
    ]
    ehr, ehr_length = pad_zeros(ehr)

    # 1 = present, 0 = missing
    mask_ehr = np.array([1 if item[0] is not None else 0 for item in batch])

    # If missing, length = 0; otherwise keep true length
    ehr_length = [
        ehr_length[i] if mask_ehr[i] == 1 else 0
        for i in range(len(ehr_length))
    ]

    # CXR: [3,224,224]; when missing, zeros (dummy tensor; true missingness is mask_cxr)
    cxr = torch.stack([
        item[1] if item[1] is not None else torch.zeros(3, 224, 224)
        for item in batch
    ])
    mask_cxr = np.array([1 if item[1] is not None else 0 for item in batch])

    # Notes (raw strings)
    note = [item[2] for item in batch]
    mask_note = np.array([1 if item[2] != '' else 0 for item in batch])

    # Labels (may contain -1 for missing)
    label = np.array([item[3] for item in batch]).reshape(len(batch), -1)

    # Task index
    replace_dict = {
        'in-hospital-mortality': 0, 'decompensation': 1, 'phenotyping': 2,
        'length-of-stay': 3, 'readmission': 4, 'diagnosis': 5, 'drg': 6
    }
    task_index = np.array([
        replace_dict[item[6]] if item[6] in replace_dict else -1
        for item in batch
    ])

    return [ehr, ehr_length, mask_ehr,
            cxr, mask_cxr,
            note, mask_note,
            label, task_index]


# =========================
# Main training / eval
# =========================

def main():
    logger, log_file, tqdm_kwargs = setup_logging(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(
        "cuda" if (getattr(args, 'device', 'cpu') != "cpu" and torch.cuda.is_available())
        else "cpu"
    )
    args.task = args.task.split(',')

    num_workers = args.num_workers
    early_patience = args.early_stop_patience
    early_min_delta = args.early_stop_min_delta

    shapley_alpha = float(args.shapley_alpha)
    shapley_start_epoch = int(args.shapley_start_epoch)
    shapley_weight_scale = float(args.shapley_weight_scale)

    # ----- TensorBoard writer -----
    tb_base = args.tb_log_dir
    if os.path.isdir(tb_base):
        os.makedirs(tb_base, exist_ok=True)
        tb_run = f"flexcare_seed{args.seed}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        tb_logdir = os.path.join(tb_base, tb_run)
    else:
        tb_logdir = tb_base
        os.makedirs(tb_logdir, exist_ok=True)

    writer = SummaryWriter(log_dir=tb_logdir)
    logger.info(f"[TB] Logging to {tb_logdir}")

    # ----- Data plumbing -----
    discretizer = Discretizer(
        timestep=float(args.timestep),
        store_masks=True,
        impute_strategy='previous',
        start_time='zero'
    )

    def cont_channels_from_template():
        path = f'{args.ehr_path}/10002430_episode1_timeseries.csv'
        with open(path, "r") as tsfile:
            header = tsfile.readline().strip().split(',')
        cont_channels = [i for (i, x) in enumerate(header)
                         if x.find("->") == -1 and x != "Hours"]
        return cont_channels

    mutli_train_dl, mutli_val_dl, mutli_test_dl = [], [], []
    for t, task in enumerate(args.task):
        normalizer = Normalizer(fields=cont_channels_from_template())
        normalizer_state = args.normalizer_state or osp.join(
            osp.dirname(__file__),
            'normalizers/ph_ts{}.input_str_previous.start_time_zero.normalizer'.format(1.0)
        )
        normalizer.load_params(normalizer_state)

        # Original datasets from your dataloader
        train_ds, val_ds, test_ds = get_multimodal_datasets(discretizer, normalizer, args, task)

        # ---- 60% of TRAIN ----
        N_train = len(train_ds)
        train_60_size = int(0.6 * N_train)

        g_train = torch.Generator()
        g_train.manual_seed(args.seed + 10 * t)  # deterministic per task

        perm_train = torch.randperm(N_train, generator=g_train)
        train_indices_60 = perm_train[:train_60_size].tolist()
        train_ds_60 = Subset(train_ds, train_indices_60)

        # ---- 60% of VAL ----
        N_val = len(val_ds)
        val_60_size = int(0.6 * N_val)

        g_val = torch.Generator()
        g_val.manual_seed(args.seed + 1000 + 10 * t)  # deterministic per task

        perm_val = torch.randperm(N_val, generator=g_val)
        val_indices_60 = perm_val[:val_60_size].tolist()
        val_ds_60 = Subset(val_ds, val_indices_60)

        logger.info(
            f"[DATA] Task={task} | train={N_train} -> train60={len(train_ds_60)} | "
            f"val={N_val} -> val60={len(val_ds_60)} | test={len(test_ds)}"
        )

        # ---- DataLoaders ----
        mutli_train_dl.append(
            DataLoader(
                train_ds_60,
                args.batch_size,
                shuffle=True,
                collate_fn=my_collate,
                pin_memory=True,
                num_workers=num_workers,
                drop_last=True,
            )
        )
        mutli_val_dl.append(
            DataLoader(
                val_ds_60,
                args.batch_size,
                shuffle=False,
                collate_fn=my_collate,
                pin_memory=True,
                num_workers=num_workers,
                drop_last=False,
            )
        )
        mutli_test_dl.append(
            DataLoader(
                test_ds,   # same test set as before
                args.batch_size,
                shuffle=False,
                collate_fn=my_collate,
                pin_memory=True,
                num_workers=num_workers,
                drop_last=False,
            )
        )

    # ----- Model / loss / opt -----
    num_modalities = 3  # ehr, cxr, note
    r = clamp01(compute_feature_knockout_rate(num_modalities))  # ~0.2

    model = FlexCare(
        hidden_dim=args.hidden_dim,
        layers=4,
        expert_k=2,
        expert_total=10,
        device=device
    ).to(device)

    # Fixed modality-level knockout (inside model, after z-score).
    # Shapley will NOT change this: Shapley is only for loss weighting.
    if hasattr(model, "mod_dropout"):
        model.mod_dropout = {'ehr': r, 'cxr': r, 'note': r}
    model.feature_knockout_rate = r

    logger.info(f"[KO] Modality-level knockout prob r = {r:.4f} for d={num_modalities}.")
    if hasattr(model, "mod_dropout"):
        logger.info(f"[KO] model.mod_dropout = {model.mod_dropout}")
    writer.add_scalar("KO/feature_knockout_rate", r, 0)

    # CE with ignore_index=-1 for LOS/DRG (missing labels)
    criterion_ce = nn.CrossEntropyLoss(ignore_index=-1)
    criterion_ce_nored = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=getattr(args, 'weight_decay', 1e-2),
    )

    best_epoch = 0
    best_valid_loss = float('inf')  # for logging only (mean of best per-combo losses)
    best_valid_res = -1e9          # sum(AUC+AUPR_nat) over tasks, just for info
    no_improve_epochs = 0

    # New: best global average AUC across all combos, for checkpointing
    best_global_avg_auc = -float('inf')

    ortho_coeff = 0.01
    ortho_warmup_epochs = 0  # set >0 to warmup without ortho

    # ---- modality combinations for validation & test ----
    combo_cfgs = {
        "nat":       {"ehr": None, "cxr": None, "note": None},  # observed pattern
        "ehr_only":  {"ehr": 1,    "cxr": 0,    "note": 0},
        "cxr_only":  {"ehr": 0,    "cxr": 1,    "note": 0},
        "note_only": {"ehr": 0,    "cxr": 0,    "note": 1},
        "ehr_cxr":   {"ehr": 1,    "cxr": 1,    "note": 0},
        "ehr_note":  {"ehr": 1,    "cxr": 0,    "note": 1},
        "cxr_note":  {"ehr": 0,    "cxr": 1,    "note": 1},
    }
    combo_names = list(combo_cfgs.keys())
    best_combo_loss = {name: float('inf') for name in combo_names}

    # history for plotting losses
    combo_history = {name: [] for name in combo_names}

    # ---- Shapley-based MODALITY REBALANCING (loss weighting only) ----
    # Start with uniform rebalancing weights
    current_rebal_weights = {'ehr': 1.0 / 3.0, 'cxr': 1.0 / 3.0, 'note': 1.0 / 3.0}
    last_shapley_importance = {'ehr': 1.0 / 3.0, 'cxr': 1.0 / 3.0, 'note': 1.0 / 3.0}

    # histories for Shapley curves and rebal curves
    shapley_history = {'ehr': [], 'cxr': [], 'note': []}
    rebal_history = {'ehr': [], 'cxr': [], 'note': []}

    logger.info(f"[SHAPLEY] Initial rebalancing weights (uniform): {current_rebal_weights}")

    # ====== Training loop (KNOCKOUT in model, Shapley for LOSS) ======
    for epoch in tqdm(range(1, args.epochs + 1), **tqdm_kwargs):
        model.train()
        epoch_train_loss = 0.0
        epoch_train_count = 0

        epoch_core_loss_sum = 0.0
        epoch_ortho_loss_sum = 0.0

        for t_idx in range(len(mutli_train_dl)):
            task_now = args.task[t_idx]

            with tqdm(mutli_train_dl[t_idx], position=0, ncols=120, **tqdm_kwargs) as tq:
                for _, data in enumerate(tq):
                    optimizer.zero_grad(set_to_none=True)

                    ehr, ehr_length, mask_ehr, cxr, mask_cxr, note, mask_note, label, task_index = data
                    ehr = torch.from_numpy(ehr).float().to(device)
                    cxr = cxr.to(device)
                    mask_ehr_t = torch.from_numpy(mask_ehr).long().to(device)
                    mask_cxr_t = torch.from_numpy(mask_cxr).long().to(device)
                    mask_note_t = torch.from_numpy(mask_note).long().to(device)
                    y_true = torch.from_numpy(label).float().to(device)
                    task_index = torch.from_numpy(task_index).long().to(device)

                    out = model(
                        ehr, ehr_length, mask_ehr_t,
                        cxr, mask_cxr_t,
                        note, mask_note_t,
                        task_index
                    )

                    # model may return (logits, ortho_loss, moe_loss) or (logits, ortho_loss)
                    if isinstance(out, (tuple, list)):
                        if len(out) >= 2:
                            y_pred, ortho_loss = out[0], out[1]
                        else:
                            y_pred, ortho_loss = out[0], 0.0
                    else:
                        y_pred, ortho_loss = out, 0.0

                    # ----- core task loss (per-sample) -----
                    if task_now in ['length-of-stay', 'drg']:
                        # CE branch: targets are class indices; may contain -1 (missing)
                        y_true_use = y_true.long().view(-1)  # (B,)
                        num_classes = y_pred.size(1)

                        # Check label range only on valid labels (!= -1)
                        valid_mask_ce = (y_true_use != -1)
                        if valid_mask_ce.any():
                            y_valid = y_true_use[valid_mask_ce]
                            if (y_valid < 0).any() or (y_valid >= num_classes).any():
                                bad_min = int(y_valid.min().item())
                                bad_max = int(y_valid.max().item())
                                raise RuntimeError(
                                    f"[TRAIN] Invalid CE labels for task '{task_now}': "
                                    f"min={bad_min}, max={bad_max}, num_classes={num_classes}"
                                )

                            # Compute per-sample CE loss only on valid indices
                            y_pred_valid = y_pred[valid_mask_ce]        # (N_valid,C)
                            y_true_valid = y_true_use[valid_mask_ce]    # (N_valid,)
                            loss_vec_valid = criterion_ce_nored(
                                y_pred_valid, y_true_valid
                            )  # (N_valid,)

                            # Shapley-based rebalancing weights (B,) -> select valid ones
                            sample_w = compute_sample_rebalancing_weights(
                                mask_ehr_t, mask_cxr_t, mask_note_t,
                                current_rebal_weights
                            )  # (B,)
                            sample_w_valid = sample_w[valid_mask_ce]   # (N_valid,)

                            core_loss = (loss_vec_valid * sample_w_valid).mean()
                        else:
                            core_loss = torch.tensor(0.0, device=device)

                    else:
                        # BCE / multi-label branch with -1 masking (logits)
                        y_true_use = y_true
                        loss_raw = bce_with_logits_ignore_neg(
                            y_pred, y_true_use, reduction="none"
                        )  # (B,L) or (B,1)

                        if loss_raw.dim() > 1:
                            valid_mask = (y_true_use >= 0).float()
                            denom = valid_mask.sum(dim=1).clamp_min(1.0)
                            loss_vec = loss_raw.sum(dim=1) / denom  # (B,)
                        else:
                            loss_vec = loss_raw  # (B,)

                        sample_w = compute_sample_rebalancing_weights(
                            mask_ehr_t, mask_cxr_t, mask_note_t,
                            current_rebal_weights
                        )  # (B,)
                        core_loss = (loss_vec * sample_w).mean()

                    # ----- Orthogonality loss (optional) -----
                    if isinstance(ortho_loss, torch.Tensor):
                        ortho_loss_val = ortho_loss
                        if epoch > ortho_warmup_epochs:
                            total_loss = core_loss + ortho_coeff * ortho_loss_val
                        else:
                            total_loss = core_loss
                    else:
                        ortho_loss_val = torch.tensor(0.0, device=device)
                        total_loss = core_loss

                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                    bsz = ehr.size(0)
                    epoch_train_loss += float(total_loss.item()) * bsz
                    epoch_train_count += bsz
                    epoch_core_loss_sum += float(core_loss.item()) * bsz
                    epoch_ortho_loss_sum += float(ortho_loss_val.item()) * bsz

        avg_train_loss = epoch_train_loss / max(1, epoch_train_count)
        avg_core_loss = epoch_core_loss_sum / max(1, epoch_train_count)
        avg_ortho_loss = epoch_ortho_loss_sum / max(1, epoch_train_count)

        logger.info(
            f"[TRAIN][Epoch {epoch:03d}] "
            f"total={avg_train_loss:.4f} core={avg_core_loss:.4f} "
            f"ortho={avg_ortho_loss:.4f} | rebal={current_rebal_weights}"
        )

        writer.add_scalar("Train/total_loss", avg_train_loss, epoch)
        writer.add_scalar("Train/core_loss", avg_core_loss, epoch)
        writer.add_scalar("Train/ortho_loss_raw", avg_ortho_loss, epoch)
        writer.add_scalar("Train/rebal_ehr", current_rebal_weights['ehr'], epoch)
        writer.add_scalar("Train/rebal_cxr", current_rebal_weights['cxr'], epoch)
        writer.add_scalar("Train/rebal_note", current_rebal_weights['note'], epoch)

        # ====== Validation: all modality combos (NO KNOCKOUT change in eval()) ======
        model.eval()
        # Sum over tasks of AUC+AUPR for nat combination (for logging)
        valid_res_sum = 0.0

        # global combo aggregates over all tasks (loss)
        global_combo_loss_sum = {name: 0.0 for name in combo_names}
        global_combo_count = {name: 0 for name in combo_names}

        # global combo aggregates over all tasks (AUC / AUPR / ACC)
        global_combo_auc_sum = {name: 0.0 for name in combo_names}
        global_combo_aupr_sum = {name: 0.0 for name in combo_names}
        global_combo_acc_sum = {name: 0.0 for name in combo_names}
        global_combo_metric_task_count = {name: 0 for name in combo_names}

        with torch.no_grad():
            for t_idx in range(len(mutli_val_dl)):
                task_now = args.task[t_idx]

                # per-task per-combo sums (loss)
                task_combo_loss_sum = {name: 0.0 for name in combo_names}
                task_combo_count = {name: 0 for name in combo_names}

                # per-task per-combo accumulators (metrics)
                combo_outGT = {
                    name: torch.FloatTensor().to(device) for name in combo_names
                }
                combo_outPRED = {
                    name: torch.FloatTensor().to(device) for name in combo_names
                }

                for _, data in enumerate(mutli_val_dl[t_idx]):
                    ehr, ehr_length, mask_ehr, cxr, mask_cxr, note, mask_note, label, task_index = data
                    ehr = torch.from_numpy(ehr).float().to(device)
                    cxr = cxr.to(device)
                    mask_ehr_t = torch.from_numpy(mask_ehr).long().to(device)
                    mask_cxr_t = torch.from_numpy(mask_cxr).long().to(device)
                    mask_note_t = torch.from_numpy(mask_note).long().to(device)
                    y_true = torch.from_numpy(label).float().to(device)
                    task_index = torch.from_numpy(task_index).long().to(device)

                    B = ehr.size(0)

                    for combo_name, cfg in combo_cfgs.items():
                        # Start from original masks
                        use_ehr  = mask_ehr_t.clone()
                        use_img  = mask_cxr_t.clone()
                        use_note = mask_note_t.clone()

                        # Start from original features
                        ehr_in  = ehr
                        cxr_in  = cxr
                        note_in = note  # typically raw strings, you just gate with mask

                        # ---- EHR gating ----
                        if cfg["ehr"] is None:
                            # natural pattern: do nothing
                            pass
                        elif cfg["ehr"] == 1:
                            # force "on if available": still only if originally present (mask_ehr_t == 1)
                            use_ehr = mask_ehr_t
                        elif cfg["ehr"] == 0:
                            # treat as observed missingness: mask=0, features=0
                            use_ehr = torch.zeros_like(mask_ehr_t)
                            ehr_in = torch.zeros_like(ehr)

                        # ---- CXR gating ----
                        if cfg["cxr"] is None:
                            pass
                        elif cfg["cxr"] == 1:
                            use_img = mask_cxr_t
                        elif cfg["cxr"] == 0:
                            # treat as observed missingness: mask=0, features=0
                            use_img = torch.zeros_like(mask_cxr_t)
                            cxr_in = torch.zeros_like(cxr)

                        # ---- Note gating ----
                        if cfg["note"] is None:
                            pass
                        elif cfg["note"] == 1:
                            use_note = mask_note_t
                        elif cfg["note"] == 0:
                            use_note = torch.zeros_like(mask_note_t)

                        pack = model(
                            ehr_in, ehr_length, use_ehr,
                            cxr_in, use_img,
                            note_in, use_note,
                            task_index
                        )

                        if isinstance(pack, (tuple, list)):
                            y_full = pack[0]
                        else:
                            y_full = pack
                        y_full = y_full.reshape(B, -1)

                        # ----- loss for this combo -----
                        if task_now in ['length-of-stay', 'drg']:
                            y_true_use = y_true.long().view(-1)
                            num_classes = y_full.size(1)

                            valid_mask_ce = (y_true_use != -1)
                            if valid_mask_ce.any():
                                y_valid = y_true_use[valid_mask_ce]
                                if (y_valid < 0).any() or (y_valid >= num_classes).any():
                                    bad_min = int(y_valid.min().item())
                                    bad_max = int(y_valid.max().item())
                                    raise RuntimeError(
                                        f"[VAL] Invalid CE labels for task '{task_now}': "
                                        f"min={bad_min}, max={bad_max}, num_classes={num_classes}"
                                    )

                                y_pred_valid = y_full[valid_mask_ce]
                                y_true_valid = y_true_use[valid_mask_ce]
                                loss_val = criterion_ce(y_pred_valid, y_true_valid)
                            else:
                                loss_val = torch.tensor(0.0, device=device)

                        else:
                            y_true_use = y_true
                            loss_val = bce_with_logits_ignore_neg(
                                y_full, y_true_use, reduction="mean"
                            )

                        task_combo_loss_sum[combo_name] += loss_val.item() * B
                        task_combo_count[combo_name] += B
                        global_combo_loss_sum[combo_name] += loss_val.item() * B
                        global_combo_count[combo_name] += B

                        # ----- metrics accumulation for this combo -----
                        if task_now in ['length-of-stay', 'drg']:
                            valid_mask_ce = (y_true_use != -1)
                            if valid_mask_ce.any():
                                y_pred_valid = y_full[valid_mask_ce]
                                y_true_valid = y_true_use[valid_mask_ce]
                                _, y_cls = torch.max(y_pred_valid, dim=1)
                                combo_outPRED[combo_name] = torch.cat(
                                    (combo_outPRED[combo_name], y_cls), 0
                                )
                                combo_outGT[combo_name] = torch.cat(
                                    (combo_outGT[combo_name], y_true_valid), 0
                                )
                        else:
                            # logits -> probabilities for metrics
                            y_probs = torch.sigmoid(y_full)
                            combo_outPRED[combo_name] = torch.cat(
                                (combo_outPRED[combo_name], y_probs), 0
                            )
                            combo_outGT[combo_name] = torch.cat(
                                (combo_outGT[combo_name], y_true_use), 0
                            )

                # ---- per-task combo losses ----
                for combo_name in combo_names:
                    if task_combo_count[combo_name] > 0:
                        avg_loss_combo = (
                            task_combo_loss_sum[combo_name] / task_combo_count[combo_name]
                        )
                        logger.info(
                            f"[VAL][Epoch {epoch:03d}] Task={task_now:20s} "
                            f"Combo={combo_name:9s} AvgLoss={avg_loss_combo:.4f}"
                        )
                        writer.add_scalar(
                            f"Loss/val_{task_now}_{combo_name}",
                            avg_loss_combo,
                            epoch
                        )

                # ---- per-task combo metrics (AUC/AUPR/ACC) ----
                for combo_name in combo_names:
                    gt = combo_outGT[combo_name]
                    pred = combo_outPRED[combo_name]
                    if gt.numel() == 0:
                        continue

                    yt_full, yp_full = _ensure_metric_shapes(task_now, gt, pred)
                    auc_full, aupr_full = my_metrics(yt_full, yp_full, task_now)
                    acc_full = compute_accuracy(yt_full, yp_full, task_now)

                    if combo_name == "nat":
                        logger.info(
                            f"[VAL][Epoch {epoch:03d}] Task={task_now:20s} NAT "
                            f"AUC={auc_full:.4f}  AUPR={aupr_full:.4f}  ACC={acc_full:.4f}"
                        )
                        writer.add_scalar(
                            f"Metric/val_{task_now}_nat_AUC",
                            auc_full,
                            epoch
                        )
                        writer.add_scalar(
                            f"Metric/val_{task_now}_nat_AUPR",
                            aupr_full,
                            epoch
                        )
                        writer.add_scalar(
                            f"Metric/val_{task_now}_nat_ACC",
                            acc_full,
                            epoch
                        )
                        # keep backward-compatible nat metric sum
                        valid_res_sum += float(auc_full + aupr_full)
                    else:
                        logger.info(
                            f"[VAL][Epoch {epoch:03d}] Task={task_now:20s} "
                            f"Combo={combo_name:9s} AUC={auc_full:.4f}  "
                            f"AUPR={aupr_full:.4f}  ACC={acc_full:.4f}"
                        )
                        writer.add_scalar(
                            f"Metric/val_{task_now}_{combo_name}_AUC",
                            auc_full,
                            epoch
                        )
                        writer.add_scalar(
                            f"Metric/val_{task_now}_{combo_name}_AUPR",
                            aupr_full,
                            epoch
                        )
                        writer.add_scalar(
                            f"Metric/val_{task_now}_{combo_name}_ACC",
                            acc_full,
                            epoch
                        )

                    global_combo_auc_sum[combo_name] += auc_full
                    global_combo_aupr_sum[combo_name] += aupr_full
                    global_combo_acc_sum[combo_name] += acc_full
                    global_combo_metric_task_count[combo_name] += 1

        # ---- global per-combo LOSS averages ----
        epoch_combo_avg = {}
        for combo_name in combo_names:
            if global_combo_count[combo_name] > 0:
                epoch_combo_avg[combo_name] = (
                    global_combo_loss_sum[combo_name] / global_combo_count[combo_name]
                )
            else:
                epoch_combo_avg[combo_name] = float('inf')

            combo_history[combo_name].append(epoch_combo_avg[combo_name])

            # log global combo averages
            if global_combo_count[combo_name] > 0:
                writer.add_scalar(
                    f"Loss/val_global_{combo_name}",
                    epoch_combo_avg[combo_name],
                    epoch
                )
                logger.info(
                    f"[VAL][Epoch {epoch:03d}] GLOBAL Combo={combo_name:9s} "
                    f"AvgLoss={epoch_combo_avg[combo_name]:.4f}"
                )

        # For backward compatibility, we'll define avg_valid_loss as nat combo global loss
        avg_valid_loss = epoch_combo_avg.get("nat", float('inf'))
        writer.add_scalar("Loss/val_nat", avg_valid_loss, epoch)
        writer.add_scalar("Metric/val_score_nat_auc_aupr_sum", valid_res_sum, epoch)

        # ---- global per-combo AUC/AUPR/ACC averages + global avg AUC/ACC ----
        global_auc_list = []
        global_acc_list = []
        for combo_name in combo_names:
            if global_combo_metric_task_count[combo_name] > 0:
                mean_auc = (
                    global_combo_auc_sum[combo_name] /
                    global_combo_metric_task_count[combo_name]
                )
                mean_aupr = (
                    global_combo_aupr_sum[combo_name] /
                    global_combo_metric_task_count[combo_name]
                )
                mean_acc = (
                    global_combo_acc_sum[combo_name] /
                    global_combo_metric_task_count[combo_name]
                )
                writer.add_scalar(
                    f"Metric/val_global_{combo_name}_AUC",
                    mean_auc,
                    epoch
                )
                writer.add_scalar(
                    f"Metric/val_global_{combo_name}_AUPR",
                    mean_aupr,
                    epoch
                )
                writer.add_scalar(
                    f"Metric/val_global_{combo_name}_ACC",
                    mean_acc,
                    epoch
                )
                logger.info(
                    f"[VAL][Epoch {epoch:03d}] GLOBAL Combo={combo_name:9s} "
                    f"AUC={mean_auc:.4f} AUPR={mean_aupr:.4f} ACC={mean_acc:.4f}"
                )
                global_auc_list.append(mean_auc)
                global_acc_list.append(mean_acc)

        if len(global_auc_list) > 0:
            global_avg_auc = float(np.mean(global_auc_list))
        else:
            global_avg_auc = float('nan')

        if len(global_acc_list) > 0:
            global_avg_acc = float(np.mean(global_acc_list))
        else:
            global_avg_acc = float('nan')

        writer.add_scalar(
            "Metric/val_global_avg_AUC_all_combos",
            global_avg_auc,
            epoch
        )
        writer.add_scalar(
            "Metric/val_global_avg_ACC_all_combos",
            global_avg_acc,
            epoch
        )

        logger.info(
            f"[VAL][Epoch {epoch:03d}] NAT AvgValLoss={avg_valid_loss:.4f} "
            f"| Sum(AUC+AUPR_nat)={valid_res_sum:.4f} "
            f"| GlobalAvgAUC(all combos)={global_avg_auc:.4f} "
            f"| GlobalAvgACC(all combos)={global_avg_acc:.4f} "
            f"| Best epoch={best_epoch} (mean best combo loss={best_valid_loss:.4f})"
        )

        # ====== Shapley-based LOSS WEIGHTING update (now ACC-based utilities) ======
        if epoch >= shapley_start_epoch:
            # SHAPE-style utilities from per-combo performance (ACCURACY)
            combo_util = metrics_to_utilities_acc(
                global_combo_acc_sum,
                global_combo_metric_task_count,
            )

            # Compute Shapley importance for each modality from accuracy utilities
            shapley_importance = compute_shapley_importance_from_utilities(
                combo_util, alpha=shapley_alpha
            )
            last_shapley_importance = shapley_importance
            logger.info(f"[SHAPLEY][Epoch {epoch:03d}] Importance (higher = stronger modality, ACC-based): "
                        f"{shapley_importance}")
            writer.add_scalar("Shapley/importance_ehr", shapley_importance['ehr'], epoch)
            writer.add_scalar("Shapley/importance_cxr", shapley_importance['cxr'], epoch)
            writer.add_scalar("Shapley/importance_note", shapley_importance['note'], epoch)

            # We want REBALANCING: upweight under-utilized / weak modalities.
            # So we invert importance:
            imp_vec = np.array([
                shapley_importance['ehr'],
                shapley_importance['cxr'],
                shapley_importance['note']
            ], dtype=np.float64)

            rebal_un = 1.0 - imp_vec
            rebal_un = np.maximum(rebal_un, 0.0)

            if not np.isfinite(rebal_un).all() or rebal_un.sum() <= 0:
                new_rebal = {'ehr': 1.0 / 3.0, 'cxr': 1.0 / 3.0, 'note': 1.0 / 3.0}
            else:
                beta = max(0.5, float(shapley_weight_scale))
                rebal_un = rebal_un ** beta
                s_rebal = float(rebal_un.sum())
                rebal_un = rebal_un / s_rebal
                new_rebal = {
                    'ehr': float(rebal_un[0]),
                    'cxr': float(rebal_un[1]),
                    'note': float(rebal_un[2]),
                }

            current_rebal_weights = new_rebal
            logger.info(
                f"[SHAPLEY][Epoch {epoch:03d}] Updated rebalancing weights "
                f"(used for loss weighting next epoch): {current_rebal_weights} | KO rate={r:.4f}"
            )
            writer.add_scalar("Shapley/rebal_ehr", current_rebal_weights['ehr'], epoch)
            writer.add_scalar("Shapley/rebal_cxr", current_rebal_weights['cxr'], epoch)
            writer.add_scalar("Shapley/rebal_note", current_rebal_weights['note'], epoch)
        else:
            # Before Shapley kicks in, we can still log the uniform importance
            writer.add_scalar("Shapley/importance_ehr", last_shapley_importance['ehr'], epoch)
            writer.add_scalar("Shapley/importance_cxr", last_shapley_importance['cxr'], epoch)
            writer.add_scalar("Shapley/importance_note", last_shapley_importance['note'], epoch)
            logger.info(
                f"[SHAPLEY][Epoch {epoch:03d}] Pre-start: using uniform importance "
                f"{last_shapley_importance} and rebal={current_rebal_weights} | KO rate={r:.4f}"
            )

        # --- append to histories for curve plotting (after possible update) ---
        for m in ['ehr', 'cxr', 'note']:
            shapley_history[m].append(last_shapley_importance[m])
            rebal_history[m].append(current_rebal_weights[m])

        # Track best nat metric sum (for info only)
        if valid_res_sum > best_valid_res:
            best_valid_res = valid_res_sum

        # ---- Checkpointing based on GLOBAL AVG AUC (metrics-based, independent of early stopping) ----
        if math.isfinite(global_avg_auc) and global_avg_auc > best_global_avg_auc:
            best_global_avg_auc = global_avg_auc
            logger.info(
                f"[CKPT] New best global avg AUC={best_global_avg_auc:.4f} at epoch {epoch}. "
                f"Saving checkpoint."
            )
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(), 'checkpoints/best_shape_v_acc.pt')

        # ====== Early stopping: SAME LOSS-BASED LOGIC AS BEFORE ======
        improved_any = False
        for combo_name in combo_names:
            cur_loss = epoch_combo_avg[combo_name]
            if cur_loss < best_combo_loss[combo_name] - early_min_delta:
                best_combo_loss[combo_name] = cur_loss
                improved_any = True

        if improved_any:
            no_improve_epochs = 0
            best_epoch = epoch
            # for logging: mean of best per-combo losses
            finite_best_losses = [
                v for v in best_combo_loss.values() if math.isfinite(v)
            ]
            if finite_best_losses:
                best_valid_loss = float(np.mean(finite_best_losses))
            logger.info(
                f"[EARLY] Improvement detected in at least one combo loss at epoch {epoch}. "
                f"Mean best combo loss={best_valid_loss:.4f}."
            )
        else:
            no_improve_epochs += 1
            logger.info(
                f"[EARLY] No improvement in ANY combo loss for {no_improve_epochs} epoch(s)."
            )
            if no_improve_epochs >= early_patience:
                logger.info(
                    f"[EARLY] Patience {early_patience} reached. "
                    f"Stopping training at epoch {epoch}."
                )
                break

    # ---- plot validation loss for all combos ----
    os.makedirs('results', exist_ok=True)
    plot_path = os.path.join('results', 'val_loss_combos.png')
    plot_combo_losses(combo_history, plot_path)
    logger.info(f"[PLOT] Saved validation loss curves to {plot_path}")

    # ---- plot Shapley + rebal vs knockout ----
    shap_plot, rebal_plot = plot_shapley_and_rebal(
        shapley_history, rebal_history, r, out_dir='results'
    )
    logger.info(f"[PLOT] Saved Shapley vs KO curves to {shap_plot}")
    logger.info(f"[PLOT] Saved rebal vs KO curves to {rebal_plot}")

    writer.close()

    # ====== Test with best checkpoint: ALL modality combinations ======
    with torch.no_grad():
        ckpt_path = 'checkpoints/best_shape_v_acc.pt'
        state_dict = torch.load(ckpt_path, map_location=device)
        model_state = model.state_dict()
        filtered = {}

        for k, v in state_dict.items():
            if k not in model_state:
                print(f"[LOAD] Skipping unexpected key: {k}")
                continue
            if v.shape != model_state[k].shape:
                print(
                    f"[LOAD] Shape mismatch for {k}: ckpt {v.shape} vs model {model_state[k].shape}, skipping"
                )
                continue
            filtered[k] = v

        model_state.update(filtered)
        model.load_state_dict(model_state)
        model.eval()

        for t_idx in range(len(mutli_test_dl)):
            task_now = args.task[t_idx]
            print(f"\n[TEST] ===== Task={task_now} =====")

            # per-combo accumulators over entire test set
            combo_outGT = {
                name: torch.FloatTensor().to(device) for name in combo_names
            }
            combo_outPRED = {
                name: torch.FloatTensor().to(device) for name in combo_names
            }

            for _, data in enumerate(mutli_test_dl[t_idx]):
                ehr, ehr_length, mask_ehr, cxr, mask_cxr, note, mask_note, label, task_index = data
                ehr = torch.from_numpy(ehr).float().to(device)
                cxr = cxr.to(device)
                mask_ehr_t = torch.from_numpy(mask_ehr).long().to(device)
                mask_cxr_t = torch.from_numpy(mask_cxr).long().to(device)
                mask_note_t = torch.from_numpy(mask_note).long().to(device)
                y_true = torch.from_numpy(label).float().to(device)
                task_index = torch.from_numpy(task_index).long().to(device)

                B = ehr.size(0)

                for combo_name, cfg in combo_cfgs.items():
                    # Start from original masks
                    use_ehr  = mask_ehr_t.clone()
                    use_img  = mask_cxr_t.clone()
                    use_note = mask_note_t.clone()

                    # Start from original features
                    ehr_in  = ehr
                    cxr_in  = cxr
                    note_in = note

                    # ---- EHR gating ----
                    if cfg["ehr"] is None:
                        pass
                    elif cfg["ehr"] == 1:
                        use_ehr = mask_ehr_t
                    elif cfg["ehr"] == 0:
                        use_ehr = torch.zeros_like(mask_ehr_t)
                        ehr_in = torch.zeros_like(ehr)

                    # ---- CXR gating ----
                    if cfg["cxr"] is None:
                        pass
                    elif cfg["cxr"] == 1:
                        use_img = mask_cxr_t
                    elif cfg["cxr"] == 0:
                        use_img = torch.zeros_like(mask_cxr_t)
                        cxr_in = torch.zeros_like(cxr)

                    # ---- Note gating ----
                    if cfg["note"] is None:
                        pass
                    elif cfg["note"] == 1:
                        use_note = mask_note_t
                    elif cfg["note"] == 0:
                        use_note = torch.zeros_like(mask_note_t)

                    y_pred_pack = model(
                        ehr_in, ehr_length, use_ehr,
                        cxr_in, use_img,
                        note_in, use_note,
                        task_index
                    )
                    if isinstance(y_pred_pack, (tuple, list)):
                        y_pred = y_pred_pack[0]
                    else:
                        y_pred = y_pred_pack
                    y_pred = y_pred.reshape(B, -1)

                    if task_now in ['length-of-stay', 'drg']:
                        y_true_use = y_true.long().view(-1)
                        num_classes = y_pred.size(1)

                        valid_mask_ce = (y_true_use != -1)
                        if valid_mask_ce.any():
                            y_valid = y_true_use[valid_mask_ce]
                            if (y_valid < 0).any() or (y_valid >= num_classes).any():
                                bad_min = int(y_valid.min().item())
                                bad_max = int(y_valid.max().item())
                                raise RuntimeError(
                                    f"[TEST] Invalid CE labels for task '{task_now}': "
                                    f"min={bad_min}, max={bad_max}, num_classes={num_classes}"
                                )

                            y_pred_valid = y_pred[valid_mask_ce]
                            y_true_valid = y_true_use[valid_mask_ce]

                            _, y_cls = torch.max(y_pred_valid, dim=1)
                            combo_outPRED[combo_name] = torch.cat(
                                (combo_outPRED[combo_name], y_cls), 0
                            )
                            combo_outGT[combo_name] = torch.cat(
                                (combo_outGT[combo_name], y_true_valid), 0
                            )
                        else:
                            continue

                    else:
                        y_true_use = y_true
                        y_probs = torch.sigmoid(y_pred)
                        combo_outPRED[combo_name] = torch.cat(
                            (combo_outPRED[combo_name], y_probs), 0
                        )
                        combo_outGT[combo_name] = torch.cat(
                            (combo_outGT[combo_name], y_true_use), 0
                        )

            # ---- compute metrics per combo on test ----
            for combo_name in combo_names:
                gt = combo_outGT[combo_name]
                pred = combo_outPRED[combo_name]
                if gt.numel() == 0:
                    print(
                        f"[TEST] Task={task_now:20s} Combo={combo_name:9s} "
                        f"has no valid labels for metrics."
                    )
                    continue

                yt_full, yp_full = _ensure_metric_shapes(task_now, gt, pred)
                auc_full, aupr_full = my_metrics(yt_full, yp_full, task_now)
                acc_full = compute_accuracy(yt_full, yp_full, task_now)

                print(
                    f"[TEST] Task={task_now:20s} Combo={combo_name:9s} "
                    f"AUC={auc_full:.4f}  AUPR={aupr_full:.4f}  ACC={acc_full:.4f}"
                )


if __name__ == '__main__':
    main()

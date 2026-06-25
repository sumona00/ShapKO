# Baseline.py
from __future__ import annotations
import os, sys, math, datetime, random, argparse, io
from os import path as osp
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

# TensorBoard (use tensorboardX to avoid distutils issues)
from tensorboardX import SummaryWriter

# ---- project imports ----
from utils import Discretizer, Normalizer, my_metrics, is_ascending
from dataset.dataloader import get_multimodal_datasets
from mymodel.model_wo import FlexCare
from torch.utils.data import DataLoader, Subset, ConcatDataset

# =========================
# Argparse (reuse existing + early stopping)
# =========================
from arguments import args_parser
parser = args_parser()

# Add early-stopping flags here (we don't touch arguments.py)
es = argparse.ArgumentParser(add_help=False)
es.add_argument(
    '--early_stop_patience',
    type=int,
    default=5,
    help='Number of epochs with no improvement in val loss before early stopping.'
)
es.add_argument(
    '--early_stop_min_delta',
    type=float,
    default=0.0,
    help='Minimum decrease in val loss to count as an improvement.'
)

# First parse the base args, then parse extra ES args from leftovers
base_args, extras = parser.parse_known_args()
es_args, _ = es.parse_known_args(extras)

# Merge
for k, v in vars(es_args).items():
    setattr(base_args, k, v)
args = base_args

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
        self.stream.write(buf); self.stream.flush()
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
    os.makedirs('log', exist_ok=True); os.makedirs('results', exist_ok=True)
    stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    task_tag = '-'.join((args.task if isinstance(args.task, list) else [args.task]))
    model_tag = getattr(args, 'model', 'model')
    log_path = os.path.join('log',
                            f"[{model_tag}]_lr{args.lr}_seed{args.seed}_ep{args.epochs}_{task_tag}_{stamp}.log")
    logger = logging.getLogger('flexcare_baseline'); logger.setLevel(logging.INFO); logger.handlers = []
    fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    fh = RotatingFileHandler(log_path, maxBytes=50_000_000, backupCount=1); fh.setFormatter(fmt); logger.addHandler(fh)
    sh = logging.StreamHandler(sys.__stdout__); sh.setFormatter(fmt); logger.addHandler(sh)
    sys.stdout = Tee(sys.__stdout__, logger, level=logging.INFO)
    sys.stderr = Tee(sys.__stderr__, logger, level=logging.ERROR)
    return logger, log_path, dict(file=sys.stdout)

# -------------------------
# BCE helper: ignore -1 labels
# -------------------------
import torch.nn.functional as F

def bce_loss_ignore_neg(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    reduction: str = "mean"
) -> torch.Tensor:
    """
    Binary cross-entropy that ignores entries where y_true < 0.
    Works for shapes (B,), (B,1) or (B,L).

    For reduction='none', returns (B,L) tensor masked (0 where missing).
    For reduction='mean', returns scalar.
    """
    mask = (y_true >= 0)
    if mask.sum() == 0:
        # No valid labels
        if reduction == "none":
            return torch.zeros_like(y_true, dtype=torch.float, device=y_pred.device)
        return torch.zeros((), device=y_pred.device)

    y_pred = torch.clamp(y_pred, 1e-4, 1.0 - 1e-4)
    y_true_clamped = torch.clamp(y_true.float(), 0.0, 1.0)
    loss_raw = F.binary_cross_entropy(y_pred, y_true_clamped, reduction='none')
    loss_raw = loss_raw * mask.float()

    if reduction == "none":
        return loss_raw

    return loss_raw.sum() / mask.float().sum().clamp_min(1.0)

# =========================
# Collate / shapes (same as knockout)
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

    # CXR: [3,224,224]; when missing, zeros
    cxr = torch.stack([
        item[1] if item[1] is not None else torch.zeros(3, 224, 224)
        for item in batch
    ])
    mask_cxr = np.array([1 if item[1] is not None else 0 for item in batch])

    # Notes
    note = [item[2] for item in batch]
    mask_note = np.array([1 if item[2] != '' else 0 for item in batch])

    # Labels
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

def _ensure_metric_shapes(task_now, yt, yp):
    """
    Match shapes to what utils.my_metrics expects.
    This mirrors knockout.py:

    - length-of-stay and drg: treated as CE tasks -> long 1D labels/preds
    - Others: BCE-style; ensure (N,1) if needed.
    """
    ce_tasks = {'length-of-stay', 'drg'}
    if task_now in ce_tasks:
        return yt.long().view(-1), yp.long().view(-1)
    if yt.dim() == 1:
        yt = yt.unsqueeze(1)
    if yp.dim() == 1:
        yp = yp.unsqueeze(1)
    return yt, yp

# =========================
# Main training / eval with early stopping (val loss)
# =========================
def main():
    logger, log_file, tqdm_kwargs = setup_logging(args)

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    device = torch.device("cuda" if (getattr(args, 'device', 'cpu') != "cpu" and torch.cuda.is_available()) else "cpu")
    args.task = args.task.split(',')

    num_workers = args.num_workers

    # --- Early stopping config (val loss) ---
    early_patience   = args.early_stop_patience
    early_min_delta  = args.early_stop_min_delta
    best_epoch       = 0
    best_valid_loss  = float('inf')
    no_improve_epochs = 0

    # ----- TensorBoard writer -----
    tb_base = getattr(args, "tb_log_dir", "tb_logs_baseline")
    os.makedirs(tb_base, exist_ok=True)
    tb_run = f"flexcare_baseline_seed{args.seed}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    tb_logdir = os.path.join(tb_base, tb_run)
    writer = SummaryWriter(log_dir=tb_logdir)
    logger.info(f"[TB-BASE] Logging baseline to {tb_logdir}")

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
        cont_channels = [i for (i, x) in enumerate(header) if x.find("->") == -1 and x != "Hours"]
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
        g_train.manual_seed(args.seed + 10 * t)      # deterministic per task

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
    model = FlexCare(
        hidden_dim=args.hidden_dim,
        layers=4,
        expert_k=2,
        expert_total=10,
        device=device
    ).to(device)

    # Disable feature-space knockout (pure baseline)
    if hasattr(model, "mod_dropout"):
        model.mod_dropout = {'ehr': 0.0, 'cxr': 0.0, 'note': 0.0}
    if hasattr(model, "feature_knockout_rate"):
        model.feature_knockout_rate = 0.0

    logger.info("[BASELINE] Running without feature-space knockout or Shapley weighting.")

    criterion          = nn.BCELoss()
    criterion_ce       = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    rows_csv = []

    ortho_coeff = 0.01  # match knockout style: total = core + 0.01 * ortho

    # ====== Training loop ======
    for epoch in tqdm(range(1, args.epochs+1), **tqdm_kwargs):
        model.train()
        epoch_train_loss = 0.0
        epoch_train_count = 0

        epoch_core_loss_sum  = 0.0
        epoch_ortho_loss_sum = 0.0

        for t in range(len(mutli_train_dl)):
            task_now = args.task[t]

            with tqdm(mutli_train_dl[t], position=0, ncols=120, **tqdm_kwargs) as tq:
                for _, data in enumerate(tq):
                    optimizer.zero_grad(set_to_none=True)

                    ehr, ehr_length, mask_ehr, cxr, mask_cxr, note, mask_note, label, task_index = data
                    ehr = torch.from_numpy(ehr).float().to(device)
                    cxr = cxr.to(device)
                    mask_ehr = torch.from_numpy(mask_ehr).long().to(device)
                    mask_cxr = torch.from_numpy(mask_cxr).long().to(device)
                    mask_note = torch.from_numpy(mask_note).long().to(device)
                    y_true = torch.from_numpy(label).float().to(device)
                    task_index = torch.from_numpy(task_index).long().to(device)

                    out = model(ehr, ehr_length, mask_ehr,
                                cxr, mask_cxr,
                                note, mask_note,
                                task_index)

                    # model returns (scores, ortho_loss, ...) in training mode
                    if isinstance(out, (tuple, list)):
                        if len(out) >= 2:
                            y_pred, ortho_loss = out[0], out[1]
                        else:
                            y_pred, ortho_loss = out[0], 0.0
                    else:
                        y_pred, ortho_loss = out, 0.0

                    # ----- core loss -----
                    if task_now in ['length-of-stay', 'drg']:
                        y_true_use = y_true.long().view(-1)
                        core_loss = criterion_ce(y_pred, y_true_use)
                    else:
                        y_true_use = y_true
                        core_loss = bce_loss_ignore_neg(y_pred, y_true_use, reduction="mean")

                    # ----- orthogonality only (no other aux terms) -----
                    if isinstance(ortho_loss, torch.Tensor):
                        ortho_loss_val = ortho_loss
                        total_loss = core_loss + ortho_coeff * ortho_loss_val
                    else:
                        ortho_loss_val = torch.tensor(0.0, device=device)
                        total_loss = core_loss

                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                    bsz = ehr.size(0)
                    epoch_train_loss     += float(total_loss.item()) * bsz
                    epoch_train_count    += bsz
                    epoch_core_loss_sum  += float(core_loss.item()) * bsz
                    epoch_ortho_loss_sum += float(ortho_loss_val.item()) * bsz

        avg_train_loss = epoch_train_loss / max(1, epoch_train_count)
        avg_core_loss  = epoch_core_loss_sum  / max(1, epoch_train_count)
        avg_ortho_loss = epoch_ortho_loss_sum / max(1, epoch_train_count)

        logger.info(f"[TRAIN-BASE][Epoch {epoch:03d}] "
                    f"total={avg_train_loss:.4f} core={avg_core_loss:.4f} "
                    f"ortho={avg_ortho_loss:.4f}")

        writer.add_scalar("Train/total_loss", avg_train_loss, epoch)
        writer.add_scalar("Train/core_loss",  avg_core_loss,  epoch)
        writer.add_scalar("Train/ortho_loss_raw", avg_ortho_loss, epoch)

        # ====== Validation (FULL multimodal only; AUC style = knockout) ======
        model.eval()
        valid_res_sum = 0.0   # Sum(AUC+AUPR) across tasks, just for logging
        valid_loss_sum = 0.0
        val_sample_count = 0

        with torch.no_grad():
            for t in range(len(mutli_val_dl)):
                task_now = args.task[t]
                outGT_full = torch.FloatTensor().to(device)
                outPRED_full = torch.FloatTensor().to(device)

                for _, data in enumerate(mutli_val_dl[t]):
                    ehr, ehr_length, mask_ehr, cxr, mask_cxr, note, mask_note, label, task_index = data
                    ehr = torch.from_numpy(ehr).float().to(device)
                    cxr = cxr.to(device)
                    mask_ehr = torch.from_numpy(mask_ehr).long().to(device)
                    mask_cxr = torch.from_numpy(mask_cxr).long().to(device)
                    mask_note = torch.from_numpy(mask_note).long().to(device)
                    y_true = torch.from_numpy(label).float().to(device)
                    task_index = torch.from_numpy(task_index).long().to(device)

                    pack = model(ehr, ehr_length, mask_ehr,
                                 cxr, mask_cxr,
                                 note, mask_note,
                                 task_index)
                    if isinstance(pack, (tuple, list)):
                        y_full = pack[0]
                    else:
                        y_full = pack
                    y_full = y_full.reshape(ehr.shape[0], -1)

                    if task_now in ['length-of-stay','drg']:
                        y_true_use = y_true.long().view(-1)
                        loss_val = criterion_ce(y_full, y_true_use)
                    else:
                        y_true_use = y_true
                        loss_val = bce_loss_ignore_neg(y_full, y_true_use, reduction="mean")

                    valid_loss_sum += loss_val.item() * ehr.size(0)
                    val_sample_count += ehr.size(0)

                    # ---- accumulate preds/labels for AUC/AUPR (same as knockout) ----
                    if task_now in ['length-of-stay','drg']:
                        _, y_cls = torch.max(y_full, dim=1)
                        outPRED_full = torch.cat((outPRED_full, y_cls), 0)
                        outGT_full   = torch.cat((outGT_full,   y_true_use), 0)
                    else:
                        outPRED_full = torch.cat((outPRED_full, y_full), 0)
                        outGT_full   = torch.cat((outGT_full,   y_true_use), 0)

                yt_full, yp_full = _ensure_metric_shapes(task_now, outGT_full, outPRED_full)
                auc_full, aupr_full = my_metrics(yt_full, yp_full, task_now)
                logger.info(f"[VAL-BASE][Epoch {epoch:03d}] Task={task_now:20s}  FULL   AUC={auc_full:.4f}  AUPR={aupr_full:.4f}")
                rows_csv.append(('val', f'epoch_{epoch}', task_now, 'FULL', float(auc_full), float(aupr_full)))

                valid_res_sum += float(auc_full + aupr_full)

        avg_valid_loss = valid_loss_sum / max(1, val_sample_count)
        writer.add_scalar("Loss/val", avg_valid_loss, epoch)
        writer.add_scalar("Metric/val_score", valid_res_sum, epoch)

        logger.info(f"[VAL-BASE][Epoch {epoch:03d}] Sum(AUC+AUPR)={valid_res_sum:.4f} | "
                    f"Best epoch={best_epoch} best_val_loss={best_valid_loss:.4f} | "
                    f"Avg train loss={avg_train_loss:.4f} Avg val loss={avg_valid_loss:.4f}")

        # ==== Early stopping based on val loss ====
        if avg_valid_loss < (best_valid_loss - early_min_delta):
            best_valid_loss = avg_valid_loss
            best_epoch = epoch
            no_improve_epochs = 0

            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(), 'checkpoints/baseline_best.pt')
            logger.info(f"[CKPT-BASE] Saved new best at epoch {epoch} (val_loss={best_valid_loss:.4f})")
        else:
            no_improve_epochs += 1
            logger.info(f"[EARLY-BASE] No improvement in val_loss for {no_improve_epochs} epoch(s).")
            if no_improve_epochs >= early_patience:
                logger.info(f"[EARLY-BASE] Patience {early_patience} reached. Stopping training at epoch {epoch}.")
                break

    writer.close()

    # ====== Test with best baseline checkpoint (AUC style = knockout) ======
    with torch.no_grad():
        ckpt_path = 'checkpoints/baseline_best.pt'
        state_dict = torch.load(ckpt_path, map_location=device)
        model_state = model.state_dict()
        filtered = {}

        for k, v in state_dict.items():
            if k not in model_state:
                print(f"[LOAD-BASE] Skipping unexpected key: {k}")
                continue
            if v.shape != model_state[k].shape:
                print(f"[LOAD-BASE] Shape mismatch for {k}: ckpt {v.shape} vs model {model_state[k].shape}, skipping")
                continue
            filtered[k] = v

        model_state.update(filtered)
        model.load_state_dict(model_state)
        model.eval()

        for t in range(len(mutli_test_dl)):
            task_now = args.task[t]
            outGT_full = torch.FloatTensor().to(device)
            outPRED_full = torch.FloatTensor().to(device)

            for _, data in enumerate(mutli_test_dl[t]):
                ehr, ehr_length, mask_ehr, cxr, mask_cxr, note, mask_note, label, task_index = data
                ehr = torch.from_numpy(ehr).float().to(device)
                cxr = cxr.to(device)
                mask_ehr = torch.from_numpy(mask_ehr).long().to(device)
                mask_cxr = torch.from_numpy(mask_cxr).long().to(device)
                mask_note = torch.from_numpy(mask_note).long().to(device)
                y_true = torch.from_numpy(label).float().to(device)
                task_index = torch.from_numpy(task_index).long().to(device)

                y_pred_pack = model(ehr, ehr_length, mask_ehr,
                                    cxr, mask_cxr,
                                    note, mask_note,
                                    task_index)
                if isinstance(y_pred_pack, (tuple, list)):
                    y_pred = y_pred_pack[0]
                else:
                    y_pred = y_pred_pack
                y_pred = y_pred.reshape(ehr.shape[0], -1)

                if task_now in ['length-of-stay','drg']:
                    y_true_use = y_true.long().view(-1)
                else:
                    y_true_use = y_true
                    _ = bce_loss_ignore_neg(y_pred, y_true_use, reduction="mean")

                if task_now in ['length-of-stay','drg']:
                    _, y_cls = torch.max(y_pred, dim=1)
                    outPRED_full = torch.cat((outPRED_full, y_cls), 0)
                    outGT_full   = torch.cat((outGT_full,   y_true_use), 0)
                else:
                    outPRED_full = torch.cat((outPRED_full, y_pred), 0)
                    outGT_full   = torch.cat((outGT_full,   y_true_use), 0)

            yt_full, yp_full = _ensure_metric_shapes(task_now, outGT_full, outPRED_full)
            auc_full, aupr_full = my_metrics(yt_full, yp_full, task_now)
            print(f"[TEST-BASE] Task={task_now:20s}  FULL   AUC={auc_full:.4f}  AUPR={aupr_full:.4f}")
            rows_csv.append(('test', 'final', task_now, 'FULL', float(auc_full), float(aupr_full)))

    # ----- Write CSV -----
    try:
        import csv
        out_csv = getattr(args, "baseline_report_csv", "results/baseline_report.csv")
        os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
        with open(out_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['phase','epoch_or_final','task','modality_set','AUC','AUPR'])
            for r in rows_csv:
                w.writerow(r)
        print(f"[REPORT-BASE] Wrote baseline report to {out_csv}")
    except Exception as e:
        print(f"[REPORT-BASE] Failed to write baseline report CSV: {e}")

if __name__ == '__main__':
    main()

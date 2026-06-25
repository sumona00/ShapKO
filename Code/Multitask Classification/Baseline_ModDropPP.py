# Baseline_ModDropPP.py
# ModDrop++ training for FlexCare (Liu et al., 2022).
#
# Per training step we forward the model twice:
#   full: with all available modalities of the batch
#   miss: with a random ModDrop subset (each available modality kept independently with p=0.5,
#         with the constraint that at least one modality remains)
# and optimize:
#   L_total = α * L_task(full) + β * L_task(miss)
#             + γ * sim_loss(full_features, miss_features)
#             + ortho_coeff * (ortho_full + ortho_miss)
# where sim_loss = sum over modalities of (1 - mean cosine sim of pooled post-projection features).
# Validation/test use full available modality (same convention as Baseline_OGM.py).

from __future__ import annotations
import os, sys, math, datetime, random, argparse, io
from os import path as osp
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter

from utils import Discretizer, Normalizer, my_metrics
from dataset.dataloader import get_multimodal_datasets
from mymodel.model_moddroppp import FlexCareModDropPP

from arguments import args_parser
parser = args_parser()

es = argparse.ArgumentParser(add_help=False)
es.add_argument('--early_stop_patience', type=int, default=5)
es.add_argument('--early_stop_min_delta', type=float, default=0.0)

mp = argparse.ArgumentParser(add_help=False)
mp.add_argument('--moddrop_p', type=float, default=0.5,
                help='Per-modality keep probability for ModDrop (default 0.5).')
mp.add_argument('--moddrop_alpha', type=float, default=1.0,
                help='Weight on full-modality task loss.')
mp.add_argument('--moddrop_beta',  type=float, default=1.0,
                help='Weight on missing-modality task loss.')
mp.add_argument('--moddrop_gamma', type=float, default=0.05,
                help='Weight on intra-subject similarity loss (paper uses 0.05).')
mp.add_argument('--moddrop_warmup_epochs', type=int, default=0,
                help='Skip co-training paired forward before this epoch (full-only warmup).')

tb_args = argparse.ArgumentParser(add_help=False)
tb_args.add_argument('--tb_log_dir', type=str, default='tb_logs_moddroppp')
tb_args.add_argument('--baseline_report_csv', type=str, default='results/moddroppp_report.csv')

base_args, extras = parser.parse_known_args()
es_args, extras = es.parse_known_args(extras)
mp_args, extras = mp.parse_known_args(extras)
tb_a, _ = tb_args.parse_known_args(extras)

for d in (es_args, mp_args, tb_a):
    for k, v in vars(d).items():
        setattr(base_args, k, v)
args = base_args

os.environ['CUDA_VISIBLE_DEVICES'] = str(getattr(args, 'device', 0))

import logging
from logging.handlers import RotatingFileHandler


class Tee(io.TextIOBase):
    def __init__(self, stream, logger, level=logging.INFO):
        self.stream, self.logger, self.level, self._buf = stream, logger, level, ''
    def write(self, buf):
        self.stream.write(buf); self.stream.flush()
        self._buf += buf
        while '\n' in self._buf:
            line, self._buf = self._buf.split('\n', 1)
            if line.strip():
                self.logger.log(self.level, line)
        return len(buf)
    def flush(self):
        self.stream.flush()
        if self._buf.strip():
            self.logger.log(self.level, self._buf.strip()); self._buf = ''


def setup_logging(args):
    os.makedirs('log', exist_ok=True); os.makedirs('results', exist_ok=True)
    stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    task_tag = '-'.join((args.task if isinstance(args.task, list) else [args.task]))
    log_path = os.path.join('log', f"[moddroppp]_lr{args.lr}_seed{args.seed}_ep{args.epochs}_{task_tag}_{stamp}.log")
    logger = logging.getLogger('flexcare_moddroppp'); logger.setLevel(logging.INFO); logger.handlers = []
    fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    fh = RotatingFileHandler(log_path, maxBytes=50_000_000, backupCount=1); fh.setFormatter(fmt); logger.addHandler(fh)
    sh = logging.StreamHandler(sys.__stdout__); sh.setFormatter(fmt); logger.addHandler(sh)
    sys.stdout = Tee(sys.__stdout__, logger, level=logging.INFO)
    sys.stderr = Tee(sys.__stderr__, logger, level=logging.ERROR)
    return logger, log_path, dict(file=sys.stdout)


def bce_loss_ignore_neg(y_pred, y_true, reduction="mean"):
    mask = (y_true >= 0)
    if mask.sum() == 0:
        if reduction == "none":
            return torch.zeros_like(y_true, dtype=torch.float, device=y_pred.device)
        return torch.zeros((), device=y_pred.device)
    y_pred = torch.clamp(y_pred, 1e-4, 1.0 - 1e-4)
    y_true_c = torch.clamp(y_true.float(), 0.0, 1.0)
    raw = F.binary_cross_entropy(y_pred, y_true_c, reduction='none') * mask.float()
    if reduction == "none":
        return raw
    return raw.sum() / mask.float().sum().clamp_min(1.0)


def task_loss(y_pred, y_true, task_now, criterion_ce):
    if task_now in ('length-of-stay', 'drg'):
        return criterion_ce(y_pred, y_true.long().view(-1))
    return bce_loss_ignore_neg(y_pred, y_true, reduction="mean")


def pad_zeros(arr, min_length=None):
    dtype = arr[0].dtype
    seq_length = [x.shape[0] for x in arr]
    max_len = max(seq_length)
    ret = [np.concatenate([x, np.zeros((max_len - x.shape[0],) + x.shape[1:], dtype=dtype)], axis=0) for x in arr]
    if (min_length is not None) and ret[0].shape[0] < min_length:
        ret = [np.concatenate([x, np.zeros((min_length - x.shape[0],) + x.shape[1:], dtype=dtype)], axis=0) for x in ret]
    return np.array(ret), seq_length


def my_collate(batch):
    ehr = [item[0][-512:] if item[0] is not None else np.zeros((1, 76), dtype=np.float32) for item in batch]
    ehr, ehr_length = pad_zeros(ehr)
    mask_ehr = np.array([1 if item[0] is not None else 0 for item in batch])
    ehr_length = [ehr_length[i] if mask_ehr[i] == 1 else 0 for i in range(len(ehr_length))]
    cxr = torch.stack([item[1] if item[1] is not None else torch.zeros(3, 224, 224) for item in batch])
    mask_cxr = np.array([1 if item[1] is not None else 0 for item in batch])
    note = [item[2] for item in batch]
    mask_note = np.array([1 if item[2] != '' else 0 for item in batch])
    label = np.array([item[3] for item in batch]).reshape(len(batch), -1)
    replace_dict = {'in-hospital-mortality': 0, 'decompensation': 1, 'phenotyping': 2,
                    'length-of-stay': 3, 'readmission': 4, 'diagnosis': 5, 'drg': 6}
    task_index = np.array([replace_dict[item[6]] if item[6] in replace_dict else -1 for item in batch])
    return [ehr, ehr_length, mask_ehr, cxr, mask_cxr, note, mask_note, label, task_index]


def _ensure_metric_shapes(task_now, yt, yp):
    if task_now in {'length-of-stay', 'drg'}:
        return yt.long().view(-1), yp.long().view(-1)
    if yt.dim() == 1: yt = yt.unsqueeze(1)
    if yp.dim() == 1: yp = yp.unsqueeze(1)
    return yt, yp


def sample_moddrop_keep(p_keep: float = 0.5):
    """Sample a (3,) keep mask. Each modality kept independently with p_keep,
    but the all-zero combination is rejected (always at least one modality)."""
    while True:
        ke = int(random.random() < p_keep)
        kc = int(random.random() < p_keep)
        kn = int(random.random() < p_keep)
        if (ke + kc + kn) > 0:
            return ke, kc, kn


def make_missing_inputs(ehr, ehr_length, mask_ehr_t, cxr, mask_cxr_t, note, mask_note_t, keep):
    ke, kc, kn = keep
    if ke == 0:
        ehr_d = torch.zeros_like(ehr)
        ehr_length_d = [0] * ehr.shape[0]
        mask_ehr_d = torch.zeros_like(mask_ehr_t)
    else:
        ehr_d, ehr_length_d, mask_ehr_d = ehr, ehr_length, mask_ehr_t
    if kc == 0:
        cxr_d = torch.zeros_like(cxr); mask_cxr_d = torch.zeros_like(mask_cxr_t)
    else:
        cxr_d, mask_cxr_d = cxr, mask_cxr_t
    if kn == 0:
        note_d = [""] * ehr.shape[0]; mask_note_d = torch.zeros_like(mask_note_t)
    else:
        note_d, mask_note_d = note, mask_note_t
    return ehr_d, ehr_length_d, mask_ehr_d, cxr_d, mask_cxr_d, note_d, mask_note_d


def pooled_cosine_sim_loss(f_full: dict, f_miss: dict) -> torch.Tensor:
    """
    Compute 1 - mean cosine similarity for each modality's pooled post-projection features,
    summed across modalities. Uses mean-pooling over the sequence/patch axis (length 1 for cxr-like).
    """
    total = 0.0
    n = 0
    for m in ("ehr", "cxr", "note"):
        a = f_full.get(m); b = f_miss.get(m)
        if a is None or b is None:
            continue
        # If sequence dims differ between full and miss, mean-pool to (B, D)
        a_pool = a.mean(dim=1)
        b_pool = b.mean(dim=1)
        # Per-sample cosine similarity, mean over batch
        cos = F.cosine_similarity(a_pool, b_pool, dim=1, eps=1e-8).mean()
        total = total + (1.0 - cos)
        n += 1
    if n == 0:
        device = next(iter(f_full.values())).device if len(f_full) else torch.device("cpu")
        return torch.zeros((), device=device)
    return total


def main():
    logger, log_file, tqdm_kwargs = setup_logging(args)

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if (getattr(args, 'device', 'cpu') != "cpu" and torch.cuda.is_available()) else "cpu")
    args.task = args.task.split(',')
    num_workers = args.num_workers

    early_patience  = args.early_stop_patience
    early_min_delta = args.early_stop_min_delta
    best_epoch = 0; best_valid_loss = float('inf'); no_improve = 0

    tb_dir = os.path.join(args.tb_log_dir,
                          f"flexcare_moddroppp_seed{args.seed}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir)
    logger.info(f"[TB] Logging to {tb_dir}")

    discretizer = Discretizer(timestep=float(args.timestep), store_masks=True,
                              impute_strategy='previous', start_time='zero')

    def cont_channels_from_template():
        path = f'{args.ehr_path}/10002430_episode1_timeseries.csv'
        with open(path, "r") as tsfile:
            header = tsfile.readline().strip().split(',')
        return [i for (i, x) in enumerate(header) if x.find("->") == -1 and x != "Hours"]

    multi_train_dl, multi_val_dl, multi_test_dl = [], [], []
    for t, task in enumerate(args.task):
        normalizer = Normalizer(fields=cont_channels_from_template())
        normalizer_state = args.normalizer_state or osp.join(
            osp.dirname(__file__),
            'normalizers/ph_ts{}.input_str_previous.start_time_zero.normalizer'.format(1.0))
        normalizer.load_params(normalizer_state)

        train_ds, val_ds, test_ds = get_multimodal_datasets(discretizer, normalizer, args, task)
        logger.info(f"[DATA] Task={task} | train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
        multi_train_dl.append(DataLoader(train_ds, args.batch_size, shuffle=True,
                                         collate_fn=my_collate, pin_memory=True,
                                         num_workers=num_workers, drop_last=True))
        multi_val_dl.append(DataLoader(val_ds, args.batch_size, shuffle=False,
                                       collate_fn=my_collate, pin_memory=True,
                                       num_workers=num_workers, drop_last=False))
        multi_test_dl.append(DataLoader(test_ds, args.batch_size, shuffle=False,
                                        collate_fn=my_collate, pin_memory=True,
                                        num_workers=num_workers, drop_last=False))

    model = FlexCareModDropPP(hidden_dim=args.hidden_dim, layers=4, device=device).to(device)
    if hasattr(model, "mod_dropout"):
        model.mod_dropout = {'ehr': 0.0, 'cxr': 0.0, 'note': 0.0}
    model.feature_knockout_rate = 0.0

    criterion_ce = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    rows_csv = []
    ortho_coeff = 0.01
    alpha = float(args.moddrop_alpha)
    beta  = float(args.moddrop_beta)
    gamma = float(args.moddrop_gamma)

    global_step = 0
    for epoch in tqdm(range(1, args.epochs + 1), **tqdm_kwargs):
        model.train()
        ep_total = 0.0; ep_cnt = 0
        ep_lf = 0.0; ep_lm = 0.0; ep_sim = 0.0
        apply_co = (epoch > args.moddrop_warmup_epochs)

        for t in range(len(multi_train_dl)):
            task_now = args.task[t]
            with tqdm(multi_train_dl[t], position=0, ncols=120, **tqdm_kwargs) as tq:
                for _, data in enumerate(tq):
                    global_step += 1
                    optimizer.zero_grad(set_to_none=True)

                    ehr, ehr_length, mask_ehr, cxr, mask_cxr, note, mask_note, label, task_index = data
                    ehr = torch.from_numpy(ehr).float().to(device)
                    cxr = cxr.to(device)
                    mask_ehr_t  = torch.from_numpy(mask_ehr).long().to(device)
                    mask_cxr_t  = torch.from_numpy(mask_cxr).long().to(device)
                    mask_note_t = torch.from_numpy(mask_note).long().to(device)
                    y_true = torch.from_numpy(label).float().to(device)
                    task_index_t = torch.from_numpy(task_index).long().to(device)

                    # ----- full forward -----
                    out_f = model(ehr, ehr_length, mask_ehr_t,
                                  cxr, mask_cxr_t,
                                  note, mask_note_t,
                                  task_index_t)
                    if isinstance(out_f, (tuple, list)):
                        y_f = out_f[0]; ortho_f = out_f[1] if len(out_f) >= 2 else 0.0
                    else:
                        y_f = out_f; ortho_f = 0.0
                    L_full = task_loss(y_f, y_true, task_now, criterion_ce)
                    f_full = {k: v for k, v in model.last_proj.items() if v is not None}

                    if apply_co:
                        keep = sample_moddrop_keep(args.moddrop_p)
                        ehr_d, len_d, mehr_d, cxr_d, mcxr_d, note_d, mnote_d = make_missing_inputs(
                            ehr, ehr_length, mask_ehr_t, cxr, mask_cxr_t, note, mask_note_t, keep,
                        )
                        out_m = model(ehr_d, len_d, mehr_d,
                                      cxr_d, mcxr_d,
                                      note_d, mnote_d,
                                      task_index_t)
                        if isinstance(out_m, (tuple, list)):
                            y_m = out_m[0]; ortho_m = out_m[1] if len(out_m) >= 2 else 0.0
                        else:
                            y_m = out_m; ortho_m = 0.0
                        L_miss = task_loss(y_m, y_true, task_now, criterion_ce)
                        f_miss = {k: v for k, v in model.last_proj.items() if v is not None}
                        L_sim = pooled_cosine_sim_loss(f_full, f_miss)
                    else:
                        L_miss = torch.zeros((), device=device)
                        L_sim  = torch.zeros((), device=device)
                        ortho_m = torch.zeros((), device=device)

                    ortho_f_t = ortho_f if isinstance(ortho_f, torch.Tensor) else torch.tensor(0.0, device=device)
                    ortho_m_t = ortho_m if isinstance(ortho_m, torch.Tensor) else torch.tensor(0.0, device=device)

                    total_loss = (alpha * L_full
                                  + beta * L_miss
                                  + gamma * L_sim
                                  + ortho_coeff * (ortho_f_t + ortho_m_t))
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                    bsz = ehr.size(0)
                    ep_total += float(total_loss.item()) * bsz
                    ep_cnt += bsz
                    ep_lf += float(L_full.item()) * bsz
                    ep_lm += float(L_miss.item()) * bsz
                    ep_sim += float(L_sim.item()) * bsz

                    if (global_step % 50) == 0:
                        writer.add_scalar("Train/L_full", float(L_full.item()), global_step)
                        writer.add_scalar("Train/L_miss", float(L_miss.item()), global_step)
                        writer.add_scalar("Train/L_sim",  float(L_sim.item()),  global_step)

        avg_total = ep_total / max(1, ep_cnt)
        logger.info(f"[TRAIN][Epoch {epoch:03d}] total={avg_total:.4f} "
                    f"L_full={ep_lf/max(1,ep_cnt):.4f} L_miss={ep_lm/max(1,ep_cnt):.4f} "
                    f"L_sim={ep_sim/max(1,ep_cnt):.4f} | CO={'ON' if apply_co else 'OFF'}")
        writer.add_scalar("Train/total_loss", avg_total, epoch)

        # ====== Validation (full modality) ======
        model.eval()
        valid_res_sum = 0.0; valid_loss_sum = 0.0; val_n = 0
        with torch.no_grad():
            for t in range(len(multi_val_dl)):
                task_now = args.task[t]
                outGT = torch.FloatTensor().to(device); outPRED = torch.FloatTensor().to(device)
                for _, data in enumerate(multi_val_dl[t]):
                    ehr, ehr_length, mask_ehr, cxr, mask_cxr, note, mask_note, label, task_index = data
                    ehr = torch.from_numpy(ehr).float().to(device)
                    cxr = cxr.to(device)
                    mask_ehr_t  = torch.from_numpy(mask_ehr).long().to(device)
                    mask_cxr_t  = torch.from_numpy(mask_cxr).long().to(device)
                    mask_note_t = torch.from_numpy(mask_note).long().to(device)
                    y_true = torch.from_numpy(label).float().to(device)
                    task_index_t = torch.from_numpy(task_index).long().to(device)

                    pack = model(ehr, ehr_length, mask_ehr_t, cxr, mask_cxr_t,
                                 note, mask_note_t, task_index_t)
                    y_full = pack[0] if isinstance(pack, (tuple, list)) else pack
                    y_full = y_full.reshape(ehr.shape[0], -1)

                    if task_now in ('length-of-stay', 'drg'):
                        y_true_use = y_true.long().view(-1)
                        loss_v = criterion_ce(y_full, y_true_use)
                    else:
                        y_true_use = y_true
                        loss_v = bce_loss_ignore_neg(y_full, y_true_use, reduction="mean")
                    valid_loss_sum += loss_v.item() * ehr.size(0); val_n += ehr.size(0)

                    if task_now in ('length-of-stay', 'drg'):
                        _, y_cls = torch.max(y_full, dim=1)
                        outPRED = torch.cat((outPRED, y_cls), 0)
                        outGT   = torch.cat((outGT,   y_true_use), 0)
                    else:
                        outPRED = torch.cat((outPRED, y_full), 0)
                        outGT   = torch.cat((outGT,   y_true_use), 0)

                yt, yp = _ensure_metric_shapes(task_now, outGT, outPRED)
                auc, aupr = my_metrics(yt, yp, task_now)
                logger.info(f"[VAL][Epoch {epoch:03d}] Task={task_now:20s} AUC={auc:.4f}  AUPR={aupr:.4f}")
                rows_csv.append(('val', f'epoch_{epoch}', task_now, 'FULL', float(auc), float(aupr)))
                valid_res_sum += float(auc + aupr)

        avg_valid_loss = valid_loss_sum / max(1, val_n)
        writer.add_scalar("Loss/val", avg_valid_loss, epoch)
        writer.add_scalar("Metric/val_score", valid_res_sum, epoch)
        logger.info(f"[VAL][Epoch {epoch:03d}] Sum(AUC+AUPR)={valid_res_sum:.4f} val_loss={avg_valid_loss:.4f}")

        if avg_valid_loss < (best_valid_loss - early_min_delta):
            best_valid_loss = avg_valid_loss; best_epoch = epoch; no_improve = 0
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(), os.path.join('checkpoints', os.environ.get('CKPT_NAME', 'moddroppp_best.pt')))
            logger.info(f"[CKPT] Saved best at epoch {epoch} val_loss={best_valid_loss:.4f}")
        else:
            no_improve += 1
            logger.info(f"[EARLY] no improvement {no_improve} epochs")
            if no_improve >= early_patience:
                logger.info(f"[EARLY] stopping at epoch {epoch} (best={best_epoch})")
                break

    writer.close()

    # ====== Test with best ckpt ======
    with torch.no_grad():
        ckpt_path = os.path.join('checkpoints', os.environ.get('CKPT_NAME', 'moddroppp_best.pt'))
        sd = torch.load(ckpt_path, map_location=device)
        ms = model.state_dict()
        filt = {k: v for k, v in sd.items() if k in ms and v.shape == ms[k].shape}
        ms.update(filt); model.load_state_dict(ms); model.eval()

        for t in range(len(multi_test_dl)):
            task_now = args.task[t]
            outGT = torch.FloatTensor().to(device); outPRED = torch.FloatTensor().to(device)
            for _, data in enumerate(multi_test_dl[t]):
                ehr, ehr_length, mask_ehr, cxr, mask_cxr, note, mask_note, label, task_index = data
                ehr = torch.from_numpy(ehr).float().to(device)
                cxr = cxr.to(device)
                mask_ehr_t  = torch.from_numpy(mask_ehr).long().to(device)
                mask_cxr_t  = torch.from_numpy(mask_cxr).long().to(device)
                mask_note_t = torch.from_numpy(mask_note).long().to(device)
                y_true = torch.from_numpy(label).float().to(device)
                task_index_t = torch.from_numpy(task_index).long().to(device)

                pack = model(ehr, ehr_length, mask_ehr_t, cxr, mask_cxr_t, note, mask_note_t, task_index_t)
                y_pred = pack[0] if isinstance(pack, (tuple, list)) else pack
                y_pred = y_pred.reshape(ehr.shape[0], -1)
                if task_now in ('length-of-stay', 'drg'):
                    y_true_use = y_true.long().view(-1)
                    _, y_cls = torch.max(y_pred, dim=1)
                    outPRED = torch.cat((outPRED, y_cls), 0)
                    outGT   = torch.cat((outGT,   y_true_use), 0)
                else:
                    outPRED = torch.cat((outPRED, y_pred), 0)
                    outGT   = torch.cat((outGT,   y_true), 0)

            yt, yp = _ensure_metric_shapes(task_now, outGT, outPRED)
            auc, aupr = my_metrics(yt, yp, task_now)
            print(f"[TEST] Task={task_now:20s} AUC={auc:.4f}  AUPR={aupr:.4f}")
            rows_csv.append(('test', 'final', task_now, 'FULL', float(auc), float(aupr)))

    try:
        import csv
        out_csv = args.baseline_report_csv
        os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
        with open(out_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['phase', 'epoch_or_final', 'task', 'modality_set', 'AUC', 'AUPR'])
            for r in rows_csv:
                w.writerow(r)
        print(f"[REPORT] Wrote {out_csv}")
    except Exception as e:
        print(f"[REPORT] Failed to write CSV: {e}")


if __name__ == '__main__':
    main()

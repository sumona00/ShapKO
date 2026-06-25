# knockout_fixed_flexcare.py
# Fixed knockout (no Shapley). Includes:
# - Baseline-style normalizer fallback (NO NoneType crash)
# - Early stopping + best checkpoint save
# - Full validation metrics (AUC/AUPR) and test with best checkpoint
#
from __future__ import annotations

import os
import datetime
import random
import argparse
from os import path as osp
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter

from utils import Discretizer, Normalizer, my_metrics
from dataset.dataloader import get_multimodal_datasets
from mymodel.model_knockout import FlexCare
from arguments import args_parser


# =========================
# Argparse (reuse existing + add early stopping + fixed KO)
# =========================
parser = args_parser()

_extra = argparse.ArgumentParser(add_help=False)
_extra.add_argument("--early_stop_patience", type=int, default=10,
                    help="Epochs with no improvement in val loss before stopping.")
_extra.add_argument("--early_stop_min_delta", type=float, default=0.0,
                    help="Minimum val loss decrease to count as improvement.")
_extra.add_argument("--save_dir", type=str, default="checkpoints",
                    help="Directory to save best checkpoint.")
_extra.add_argument("--save_name", type=str, default="fixed_ko_flexcare_best.pt",
                    help="Filename for best checkpoint.")
_extra.add_argument("--tb_log_dir", type=str, default="tb_logs_fixed_ko",
                    help="TensorBoard base directory.")
_extra.add_argument("--max_val_batches", type=int, default=0,
                    help="If >0, limit #batches per task for val metrics/loss (0 = full val).")

# Fixed KO controls
_extra.add_argument("--p_all_keep", type=float, default=-1.0,
                    help="If >=0, derive shared drop rate r so P(keep all)=p_all_keep. "
                         "If <0, use --drop_ehr/--drop_cxr/--drop_note.")
_extra.add_argument("--drop_ehr", type=float, default=0.20, help="Fixed dropout prob for EHR branch.")
_extra.add_argument("--drop_cxr", type=float, default=0.20, help="Fixed dropout prob for CXR branch.")
_extra.add_argument("--drop_note", type=float, default=0.20, help="Fixed dropout prob for Note branch.")
_extra.add_argument("--ensure_one_kept", action="store_true",
                    help="Ensure at least one AVAILABLE modality is kept per sample.")
_extra.add_argument("--fixed_ko_seed_offset", type=int, default=0,
                    help="Optional offset for KO RNG seed.")

# Optional override: where the normalizer files live
_extra.add_argument("--normalizer_dir", type=str, default="",
                    help="Directory containing normalizer files (optional). "
                         "If empty, uses <repo>/normalizers then cluster path.")

# Parse and merge
base_args, extras = parser.parse_known_args()
extra_args, _ = _extra.parse_known_args(extras)
for k, v in vars(extra_args).items():
    setattr(base_args, k, v)
args = base_args


# =========================
# BCE helper: ignore -1 labels
# =========================
def bce_loss_ignore_neg(y_pred: torch.Tensor, y_true: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    mask = (y_true >= 0)
    if mask.sum() == 0:
        if reduction == "none":
            return torch.zeros_like(y_true, dtype=torch.float, device=y_pred.device)
        return torch.zeros((), device=y_pred.device)

    y_pred = torch.clamp(y_pred, 1e-4, 1.0 - 1e-4)
    y_true_clamped = torch.clamp(y_true.float(), 0.0, 1.0)
    loss_raw = F.binary_cross_entropy(y_pred, y_true_clamped, reduction="none")
    loss_raw = loss_raw * mask.float()

    if reduction == "none":
        return loss_raw

    return loss_raw.sum() / mask.float().sum().clamp_min(1.0)


def _ensure_metric_shapes(task_now, yt, yp):
    ce_tasks = {'length-of-stay', 'drg'}
    if task_now in ce_tasks:
        return yt.long().view(-1), yp.long().view(-1)
    if yt.dim() == 1:
        yt = yt.unsqueeze(1)
    if yp.dim() == 1:
        yp = yp.unsqueeze(1)
    return yt, yp


# =========================
# Collate (same as baseline/knockout)
# =========================
def pad_zeros(arr, min_length=None):
    dtype = arr[0].dtype
    seq_length = [x.shape[0] for x in arr]
    max_len = max(seq_length)
    ret = [
        np.concatenate([x, np.zeros((max_len - x.shape[0],) + x.shape[1:], dtype=dtype)], axis=0)
        for x in arr
    ]
    if (min_length is not None) and ret[0].shape[0] < min_length:
        ret = [
            np.concatenate([x, np.zeros((min_length - x.shape[0],) + x.shape[1:], dtype=dtype)], axis=0)
            for x in ret
        ]
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

    replace_dict = {
        'in-hospital-mortality': 0, 'decompensation': 1, 'phenotyping': 2,
        'length-of-stay': 3, 'readmission': 4, 'diagnosis': 5, 'drg': 6
    }
    task_index = np.array([replace_dict[item[6]] if item[6] in replace_dict else -1 for item in batch])

    return [ehr, ehr_length, mask_ehr,
            cxr, mask_cxr,
            note, mask_note,
            label, task_index]


# =========================
# Normalizer selection (Baseline-like, no None crash)
# =========================
def _ts_tag_from_args(args) -> str:
    # Make sure 1.0 -> "1.0", 0.8 -> "0.8"
    ts = float(getattr(args, "timestep", 1.0))
    return f"{ts:.1f}"


def pick_normalizer_state(args, task: str) -> str:
    """
    Priority:
      1) args.normalizer_state if provided and exists
      2) args.normalizer_dir (if provided) else <repo>/normalizers
      3) cluster fallback /midtier/.../FlexCare/normalizers

    Filenames based on your normalizers folder:
      - decomp: decomp_ts{ts}.input_str_previous.n1e5.start_time_zero.normalizer
      - others: ph_ts{ts}.input_str_previous.start_time_zero.normalizer
    """
    cli_path = getattr(args, "normalizer_state", None)
    if isinstance(cli_path, str) and len(cli_path) > 0:
        if osp.exists(cli_path):
            return cli_path
        raise FileNotFoundError(f"--normalizer_state provided but not found: {cli_path}")

    ts_tag = _ts_tag_from_args(args)
    if task == "decompensation":
        fname = f"decomp_ts{ts_tag}.input_str_previous.n1e5.start_time_zero.normalizer"
    else:
        fname = f"ph_ts{ts_tag}.input_str_previous.start_time_zero.normalizer"

    here = osp.dirname(__file__)
    repo_norm_dir = osp.join(here, "normalizers")

    norm_dir = getattr(args, "normalizer_dir", "")
    if isinstance(norm_dir, str) and len(norm_dir) > 0:
        cand = osp.join(norm_dir, fname)
        if osp.exists(cand):
            return cand

    cand = osp.join(repo_norm_dir, fname)
    if osp.exists(cand):
        return cand

    cluster_dir = "/midtier/sablab/scratch/gay9002/FlexCare/normalizers"
    cand2 = osp.join(cluster_dir, fname)
    if osp.exists(cand2):
        return cand2

    raise FileNotFoundError(
        f"Could not find normalizer '{fname}'. Tried:\n"
        f"  - {osp.join(norm_dir, fname) if norm_dir else '(normalizer_dir not set)'}\n"
        f"  - {cand}\n"
        f"  - {cand2}\n"
        f"Set --normalizer_state explicitly to fix."
    )


# =========================
# Fixed KO utilities
# =========================
def base_knockout_rate(num_mods: int, p_all_keep: float = 0.5) -> float:
    r = 1.0 - (p_all_keep ** (1.0 / float(num_mods)))
    return float(np.clip(r, 0.0, 0.95))


def sample_keep_mask(avail: torch.Tensor,
                     drop_probs: torch.Tensor,
                     generator: torch.Generator | None = None) -> torch.Tensor:
    B, M = avail.shape
    u = torch.rand((B, M), device=avail.device, generator=generator)
    drop = (u < drop_probs.view(1, M)).float()
    keep = avail.float() * (1.0 - drop)
    return keep


def enforce_one_kept(avail: torch.Tensor,
                     keep: torch.Tensor,
                     generator: torch.Generator | None = None) -> torch.Tensor:
    B, M = avail.shape
    for b in range(B):
        av = torch.where(avail[b] > 0.5)[0]
        if av.numel() == 0:
            continue
        if keep[b, av].sum() < 0.5:
            j_idx = torch.randint(0, av.numel(), (1,), device=avail.device, generator=generator).item()
            j = av[j_idx]
            keep[b, j] = 1.0
    return keep


@torch.no_grad()
def realized_drop_rates(avail: torch.Tensor, keep: torch.Tensor) -> np.ndarray:
    av = (avail > 0.5).float()
    dropped = av * (1.0 - (keep > 0.5).float())
    dr = dropped.sum(dim=0) / (av.sum(dim=0) + 1e-6)
    return dr.detach().cpu().numpy()


# =========================
# Validation: loss + metrics
# =========================
@torch.no_grad()
def validate_full(
    model: FlexCare,
    val_loaders: list[DataLoader],
    device: torch.device,
    task_list: list[str],
    max_batches: int = 0,
):
    model.eval()
    ce_crit = nn.CrossEntropyLoss()

    total_loss_sum = 0.0
    total_n = 0
    per_task = []

    for t, task_now in enumerate(task_list):
        outGT_full = torch.FloatTensor().to(device)
        outPRED_full = torch.FloatTensor().to(device)

        n_batches = 0
        for data in val_loaders[t]:
            ehr, ehr_length, mask_ehr, cxr, mask_cxr, note, mask_note, label, task_index = data
            ehr = torch.from_numpy(ehr).float().to(device)
            cxr = cxr.to(device)

            mask_ehr_t = torch.from_numpy(mask_ehr).long().to(device)
            mask_cxr_t = torch.from_numpy(mask_cxr).long().to(device)
            mask_note_t = torch.from_numpy(mask_note).long().to(device)

            y_true = torch.from_numpy(label).float().to(device)
            task_index = torch.from_numpy(task_index).long().to(device)

            pack = model(
                ehr, ehr_length, mask_ehr_t,
                cxr, mask_cxr_t,
                note, mask_note_t,
                task_index,
            )
            y_full = pack[0] if isinstance(pack, (tuple, list)) else pack
            y_full = y_full.reshape(ehr.shape[0], -1)

            if task_now in ['length-of-stay', 'drg']:
                y_true_use = y_true.long().view(-1)
                loss_val = ce_crit(y_full, y_true_use)
            else:
                y_true_use = y_true
                loss_val = bce_loss_ignore_neg(y_full, y_true_use, reduction="mean")

            bsz = ehr.size(0)
            total_loss_sum += float(loss_val.item()) * bsz
            total_n += bsz

            if task_now in ['length-of-stay', 'drg']:
                _, y_cls = torch.max(y_full, dim=1)
                outPRED_full = torch.cat((outPRED_full, y_cls), 0)
                outGT_full = torch.cat((outGT_full, y_true_use), 0)
            else:
                outPRED_full = torch.cat((outPRED_full, y_full), 0)
                outGT_full = torch.cat((outGT_full, y_true_use), 0)

            n_batches += 1
            if max_batches > 0 and n_batches >= max_batches:
                break

        yt_full, yp_full = _ensure_metric_shapes(task_now, outGT_full, outPRED_full)
        auc_full, aupr_full = my_metrics(yt_full, yp_full, task_now)
        per_task.append((task_now, float(auc_full), float(aupr_full)))

    avg_val_loss = total_loss_sum / max(1, total_n)
    return float(avg_val_loss), per_task


# =========================
# Safe checkpoint save/load
# =========================
def save_checkpoint(model: nn.Module, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint_filtered(model: nn.Module, path: str, device: torch.device):
    state_dict = torch.load(path, map_location=device)
    model_state = model.state_dict()
    filtered = {}
    for k, v in state_dict.items():
        if k not in model_state:
            print(f"[LOAD] Skipping unexpected key: {k}")
            continue
        if v.shape != model_state[k].shape:
            print(f"[LOAD] Shape mismatch for {k}: ckpt {tuple(v.shape)} vs model {tuple(model_state[k].shape)}; skipping")
            continue
        filtered[k] = v
    model_state.update(filtered)
    model.load_state_dict(model_state)


# =========================
# Main
# =========================
def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = str(getattr(args, 'device', 0))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if (getattr(args, 'device', 'cpu') != "cpu" and torch.cuda.is_available()) else "cpu")

    task_list = args.task.split(',')
    num_workers = int(getattr(args, "num_workers", 0))

    # ---- TensorBoard
    os.makedirs(args.tb_log_dir, exist_ok=True)
    tb_run = f"fixed_ko_seed{args.seed}_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    writer = SummaryWriter(log_dir=os.path.join(args.tb_log_dir, tb_run))
    print(f"[TB] {os.path.join(args.tb_log_dir, tb_run)}")

    # ---- Data
    discretizer = Discretizer(
        timestep=float(args.timestep),
        store_masks=True,
        impute_strategy='previous',
        start_time='zero',
    )

    def cont_channels_from_template():
        p = f'{args.ehr_path}/10002430_episode1_timeseries.csv'
        with open(p, "r") as tsfile:
            header = tsfile.readline().strip().split(',')
        return [i for (i, x) in enumerate(header) if x.find("->") == -1 and x != "Hours"]

    mutli_train_dl, mutli_val_dl, mutli_test_dl = [], [], []
    for task in task_list:
        normalizer = Normalizer(fields=cont_channels_from_template())
        norm_path = pick_normalizer_state(args, task)
        normalizer.load_params(norm_path)
        print(f"[NORM] task={task} -> {norm_path}")

        train_ds, val_ds, test_ds = get_multimodal_datasets(discretizer, normalizer, args, task)

        mutli_train_dl.append(
            DataLoader(train_ds, args.batch_size, shuffle=True, collate_fn=my_collate,
                       pin_memory=True, num_workers=num_workers, drop_last=True)
        )
        mutli_val_dl.append(
            DataLoader(val_ds, args.batch_size, shuffle=False, collate_fn=my_collate,
                       pin_memory=True, num_workers=num_workers, drop_last=False)
        )
        mutli_test_dl.append(
            DataLoader(test_ds, args.batch_size, shuffle=False, collate_fn=my_collate,
                       pin_memory=True, num_workers=num_workers, drop_last=False)
        )

    # ---- Model
    model = FlexCare(
        hidden_dim=args.hidden_dim,
        layers=4,
        expert_k=2,
        expert_total=10,
        device=device,
        normalize_before_placeholder=True,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    ortho_coeff = 0.01
    ce_crit = nn.CrossEntropyLoss()

    # ---- Fixed KO probabilities
    if float(getattr(args, "p_all_keep", -1.0)) >= 0.0:
        r = base_knockout_rate(num_mods=3, p_all_keep=float(args.p_all_keep))
        drop_probs_np = np.array([r, r, r], dtype=np.float64)
        print(f"[KO] shared fixed drop via p_all_keep={args.p_all_keep} -> r={r:.4f}")
    else:
        drop_probs_np = np.array([float(args.drop_ehr), float(args.drop_cxr), float(args.drop_note)], dtype=np.float64)
        print(f"[KO] per-modality fixed drop_probs={np.round(drop_probs_np, 4)}")

    drop_probs_np = np.clip(drop_probs_np, 0.0, 0.95)
    drop_probs = torch.tensor(drop_probs_np, dtype=torch.float32, device=device)

    # KO RNG
    ko_gen = torch.Generator(device=device)
    ko_gen.manual_seed(int(args.seed) + int(getattr(args, "fixed_ko_seed_offset", 0)))

    # ---- Early stopping + best save
    best_val_loss = float("inf")
    best_epoch = 0
    no_improve = 0

    os.makedirs(args.save_dir, exist_ok=True)
    best_ckpt_path = os.path.join(args.save_dir, args.save_name)

    # =========================
    # Train
    # =========================
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        total_loss_sum = 0.0
        total_n = 0

        rd_sum = np.zeros(3, dtype=np.float64)
        rd_cnt = 0

        for t, task_now in enumerate(task_list):
            for data in mutli_train_dl[t]:
                optimizer.zero_grad(set_to_none=True)

                ehr, ehr_length, mask_ehr, cxr, mask_cxr, note, mask_note, label, task_index = data
                ehr = torch.from_numpy(ehr).float().to(device)
                cxr = cxr.to(device)
                y_true = torch.from_numpy(label).float().to(device)
                task_index = torch.from_numpy(task_index).long().to(device)

                avail = torch.stack([
                    torch.from_numpy(mask_ehr),
                    torch.from_numpy(mask_cxr),
                    torch.from_numpy(mask_note),
                ], dim=1).long().to(device)  # (B,3)

                keep = sample_keep_mask(avail, drop_probs, generator=ko_gen)
                if bool(getattr(args, "ensure_one_kept", False)):
                    keep = enforce_one_kept(avail, keep, generator=ko_gen)

                rd_sum += realized_drop_rates(avail, keep)
                rd_cnt += 1

                keep_masks = {
                    "ehr": keep[:, 0].long(),
                    "cxr": keep[:, 1].long(),
                    "note": keep[:, 2].long(),
                }

                out = model(
                    ehr, ehr_length, avail[:, 0],
                    cxr, avail[:, 1],
                    note, avail[:, 2],
                    task_index,
                    keep_masks=keep_masks,
                )

                if isinstance(out, (tuple, list)):
                    y_pred, ortho_loss = out[0], out[1]
                else:
                    y_pred, ortho_loss = out, torch.tensor(0.0, device=device)

                y_pred = y_pred.reshape(ehr.shape[0], -1)

                if task_now in ['length-of-stay', 'drg']:
                    core_loss = ce_crit(y_pred, y_true.long().view(-1))
                else:
                    core_loss = bce_loss_ignore_neg(y_pred, y_true, reduction="mean")

                total_loss = core_loss + ortho_coeff * ortho_loss
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                bsz = ehr.size(0)
                total_loss_sum += float(total_loss.item()) * bsz
                total_n += bsz

        train_loss = total_loss_sum / max(1, total_n)
        writer.add_scalar("Loss/train", train_loss, epoch)

        realized = rd_sum / max(1, rd_cnt)
        writer.add_scalar("KO/realized_drop_ehr", float(realized[0]), epoch)
        writer.add_scalar("KO/realized_drop_cxr", float(realized[1]), epoch)
        writer.add_scalar("KO/realized_drop_note", float(realized[2]), epoch)

        # =========================
        # Validate
        # =========================
        val_loss, per_task = validate_full(
            model=model,
            val_loaders=mutli_val_dl,
            device=device,
            task_list=task_list,
            max_batches=int(getattr(args, "max_val_batches", 0)),
        )
        writer.add_scalar("Loss/val", val_loss, epoch)

        for (task_now, auc, aupr) in per_task:
            writer.add_scalar(f"Val/{task_now}/AUC", auc, epoch)
            writer.add_scalar(f"Val/{task_now}/AUPR", aupr, epoch)

        print(f"[Ep {epoch:03d}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"fixed_drop={np.round(drop_probs_np, 3)} realized_drop={np.round(realized, 3)}")
        for (task_now, auc, aupr) in per_task:
            print(f"   [VAL] {task_now:20s} AUC={auc:.4f} AUPR={aupr:.4f}")

        # =========================
        # Early stopping + best save
        # =========================
        if val_loss < (best_val_loss - float(args.early_stop_min_delta)):
            best_val_loss = val_loss
            best_epoch = epoch
            no_improve = 0
            save_checkpoint(model, best_ckpt_path)
            print(f"[CKPT] Saved best -> {best_ckpt_path} (epoch={epoch}, val_loss={best_val_loss:.4f})")
        else:
            no_improve += 1
            print(f"[EARLY] no improvement: {no_improve}/{int(args.early_stop_patience)}")

        if no_improve >= int(args.early_stop_patience):
            print(f"[EARLY] Patience reached. Stopping at epoch {epoch}. "
                  f"Best epoch={best_epoch} best_val_loss={best_val_loss:.4f}")
            break

    writer.close()

    # =========================
    # Test with best checkpoint
    # =========================
    if osp.exists(best_ckpt_path):
        print(f"[TEST] Loading best checkpoint: {best_ckpt_path}")
        load_checkpoint_filtered(model, best_ckpt_path, device)
        model.eval()

        with torch.no_grad():
            for t, task_now in enumerate(task_list):
                outGT_full = torch.FloatTensor().to(device)
                outPRED_full = torch.FloatTensor().to(device)

                for data in mutli_test_dl[t]:
                    ehr, ehr_length, mask_ehr, cxr, mask_cxr, note, mask_note, label, task_index = data
                    ehr = torch.from_numpy(ehr).float().to(device)
                    cxr = cxr.to(device)

                    mask_ehr_t = torch.from_numpy(mask_ehr).long().to(device)
                    mask_cxr_t = torch.from_numpy(mask_cxr).long().to(device)
                    mask_note_t = torch.from_numpy(mask_note).long().to(device)

                    y_true = torch.from_numpy(label).float().to(device)
                    task_index = torch.from_numpy(task_index).long().to(device)

                    pack = model(
                        ehr, ehr_length, mask_ehr_t,
                        cxr, mask_cxr_t,
                        note, mask_note_t,
                        task_index,
                    )
                    y_pred = pack[0] if isinstance(pack, (tuple, list)) else pack
                    y_pred = y_pred.reshape(ehr.shape[0], -1)

                    if task_now in ['length-of-stay', 'drg']:
                        y_true_use = y_true.long().view(-1)
                        _, y_cls = torch.max(y_pred, dim=1)
                        outPRED_full = torch.cat((outPRED_full, y_cls), 0)
                        outGT_full = torch.cat((outGT_full, y_true_use), 0)
                    else:
                        y_true_use = y_true
                        outPRED_full = torch.cat((outPRED_full, y_pred), 0)
                        outGT_full = torch.cat((outGT_full, y_true_use), 0)

                yt_full, yp_full = _ensure_metric_shapes(task_now, outGT_full, outPRED_full)
                auc_full, aupr_full = my_metrics(yt_full, yp_full, task_now)
                print(f"[TEST] Task={task_now:20s} FULL AUC={auc_full:.4f} AUPR={aupr_full:.4f}")
    else:
        print(f"[TEST] Best checkpoint not found at {best_ckpt_path}; skipping test.")


if __name__ == "__main__":
    main()

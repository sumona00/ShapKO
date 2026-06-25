from __future__ import annotations

import os
import math
import csv
import datetime
import inspect
from os import path as osp

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import Discretizer, Normalizer, my_metrics
from dataset.dataloader import get_multimodal_datasets
from mymodel.model_knockout import FlexCare
from arguments import args_parser

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize


# =========================
# Normalizer selection (same idea as train)
# =========================
def _ts_tag_from_args(args) -> str:
    ts = float(getattr(args, "timestep", 1.0))
    return f"{ts:.1f}"


def pick_normalizer_state(args, task: str) -> str:
    cli_path = getattr(args, "normalizer_state", None)
    if isinstance(cli_path, str) and len(cli_path) > 0:
        if osp.exists(cli_path):
            return cli_path
        raise FileNotFoundError(f"--normalizer_state provided but not found: {cli_path}")

    ts_tag = _ts_tag_from_args(args)

    if task == "decompensation":
        suffix = str(getattr(args, "normalizer_decomp_suffix",
                             "input_str_previous.n1e5.start_time_zero.normalizer"))
        fname = f"decomp_ts{ts_tag}.{suffix}"
    else:
        suffix = str(getattr(args, "normalizer_ph_suffix",
                             "input_str_previous.start_time_zero.normalizer"))
        fname = f"ph_ts{ts_tag}.{suffix}"

    tried = []

    norm_dir = getattr(args, "normalizer_dir", "")
    if isinstance(norm_dir, str) and len(norm_dir) > 0:
        cand = osp.join(norm_dir, fname)
        tried.append(cand)
        if osp.exists(cand):
            return cand

    here = osp.dirname(__file__)
    cand_repo = osp.join(here, "normalizers", fname)
    tried.append(cand_repo)
    if osp.exists(cand_repo):
        return cand_repo

    cand_cluster = osp.join("/midtier/sablab/scratch/gay9002/FlexCare/normalizers", fname)
    tried.append(cand_cluster)
    if osp.exists(cand_cluster):
        return cand_cluster

    raise FileNotFoundError(
        f"Could not find normalizer file for task='{task}' timestep={ts_tag}.\n"
        f"Expected filename: {fname}\n"
        f"Tried:\n  - " + "\n  - ".join(tried) + "\n"
        f"Fix: pass --normalizer_state /full/path/to/<file>.normalizer"
    )


# =========================
# Collate (same as train)
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
# Metrics: robust multiclass AUC/AUPR + ACC/F1/SENS/SPEC
# =========================
def multiclass_auc_aupr(yt: torch.Tensor, yp_prob: torch.Tensor):
    yt_np = yt.detach().cpu().numpy()
    yp_np = yp_prob.detach().cpu().numpy()
    n_classes = yp_np.shape[1]
    classes = np.arange(n_classes)

    # AUC (macro, OVR; skip degenerate)
    auc_list = []
    for c in classes:
        y_true_c = (yt_np == c).astype(int)
        if len(np.unique(y_true_c)) < 2:
            continue
        try:
            auc_list.append(roc_auc_score(y_true_c, yp_np[:, c]))
        except Exception:
            continue
    auc = float(np.mean(auc_list)) if len(auc_list) > 0 else float("nan")

    # AUPR (macro; fallback manual)
    try:
        y_true_oh = label_binarize(yt_np, classes=classes)
        aupr = float(average_precision_score(y_true_oh, yp_np, average="macro"))
    except Exception:
        ap_list = []
        for c in classes:
            y_true_c = (yt_np == c).astype(int)
            if len(np.unique(y_true_c)) < 2:
                continue
            try:
                ap_list.append(average_precision_score(y_true_c, yp_np[:, c]))
            except Exception:
                continue
        aupr = float(np.mean(ap_list)) if len(ap_list) > 0 else float("nan")

    return auc, aupr


def _ensure_metric_shapes(task_now: str, yt: torch.Tensor, yp: torch.Tensor):
    ce_tasks = {'length-of-stay', 'drg'}
    if task_now in ce_tasks:
        return yt.long().view(-1), yp.float()
    if yt.dim() == 1:
        yt = yt.unsqueeze(1)
    if yp.dim() == 1:
        yp = yp.unsqueeze(1)
    return yt, yp


def _safe_div(num, den):
    num = float(num); den = float(den)
    if den <= 0:
        return float("nan")
    return num / den


def compute_binary_multilabel_metrics(yt: torch.Tensor, yp: torch.Tensor, threshold: float = 0.5):
    # micro-average over valid entries (yt>=0)
    mask = (yt >= 0)
    if mask.sum() == 0:
        return (float("nan"),) * 4

    y_true = yt[mask].float()
    y_prob = yp[mask].float()
    y_pred = (y_prob >= threshold).float()

    tp = (y_pred.eq(1) & y_true.eq(1)).sum().item()
    tn = (y_pred.eq(0) & y_true.eq(0)).sum().item()
    fp = (y_pred.eq(1) & y_true.eq(0)).sum().item()
    fn = (y_pred.eq(0) & y_true.eq(1)).sum().item()

    total = tp + tn + fp + fn
    acc = _safe_div(tp + tn, total)

    prec = _safe_div(tp, tp + fp)
    sens = _safe_div(tp, tp + fn)
    spec = _safe_div(tn, tn + fp)

    if math.isnan(prec) or math.isnan(sens) or (prec + sens) == 0:
        f1 = float("nan")
    else:
        f1 = 2.0 * prec * sens / (prec + sens)

    return acc, f1, sens, spec


def compute_multiclass_metrics(yt: torch.Tensor, yp_labels: torch.Tensor):
    yt = yt.view(-1).long()
    yp_labels = yp_labels.view(-1).long()

    correct = (yt == yp_labels).sum().item()
    total = yt.numel()
    acc = _safe_div(correct, total)

    classes = torch.unique(yt)
    classes = classes[classes >= 0]
    if classes.numel() == 0:
        return (float("nan"),) * 4

    f1_list, sens_list, spec_list = [], [], []
    for c_t in classes:
        c = int(c_t.item())
        true_pos = (yt == c)
        true_neg = (yt != c)
        pred_pos = (yp_labels == c)
        pred_neg = (yp_labels != c)

        tp = (true_pos & pred_pos).sum().item()
        fn = (true_pos & pred_neg).sum().item()
        fp = (true_neg & pred_pos).sum().item()
        tn = (true_neg & pred_neg).sum().item()

        prec_c = _safe_div(tp, tp + fp)
        sens_c = _safe_div(tp, tp + fn)
        spec_c = _safe_div(tn, tn + fp)

        if math.isnan(prec_c) or math.isnan(sens_c) or (prec_c + sens_c) == 0:
            f1_c = float("nan")
        else:
            f1_c = 2.0 * prec_c * sens_c / (prec_c + sens_c)

        if not math.isnan(f1_c):   f1_list.append(f1_c)
        if not math.isnan(sens_c): sens_list.append(sens_c)
        if not math.isnan(spec_c): spec_list.append(spec_c)

    f1 = float(np.mean(f1_list)) if len(f1_list) > 0 else float("nan")
    sens = float(np.mean(sens_list)) if len(sens_list) > 0 else float("nan")
    spec = float(np.mean(spec_list)) if len(spec_list) > 0 else float("nan")
    return acc, f1, sens, spec


def compute_all_metrics(task_now: str, yt: torch.Tensor, yp: torch.Tensor):
    ce_tasks = {'length-of-stay', 'drg'}
    if task_now in ce_tasks:
        yp_labels = yp.argmax(dim=1) if yp.dim() > 1 else yp
        return compute_multiclass_metrics(yt, yp_labels)
    return compute_binary_multilabel_metrics(yt, yp)


# =========================
# Checkpoint load (filtered)
# =========================
def load_checkpoint_filtered(model: nn.Module, ckpt_path: str, device: torch.device):
    state_dict = torch.load(ckpt_path, map_location=device)
    model_state = model.state_dict()
    filtered = {}
    skipped = 0
    for k, v in state_dict.items():
        if k not in model_state:
            skipped += 1
            continue
        if tuple(v.shape) != tuple(model_state[k].shape):
            skipped += 1
            continue
        filtered[k] = v
    model_state.update(filtered)
    model.load_state_dict(model_state)
    if skipped > 0:
        print(f"[LOAD] loaded={len(filtered)} keys, skipped={skipped} keys (likely hidden_dim mismatch if many skipped).")


# =========================
# Modality combos (keep triplets)
# =========================
def modality_combos():
    # name -> keep (ehr,cxr,note)
    return [
        ("FULL",     (1, 1, 1)),
        ("EHR",      (1, 0, 0)),
        ("CXR",      (0, 1, 0)),
        ("NOTE",     (0, 0, 1)),
        ("EHR+CXR",  (1, 1, 0)),
        ("EHR+NOTE", (1, 0, 1)),
        ("CXR+NOTE", (0, 1, 1)),
    ]


def forward_supports_keep_masks(model: nn.Module) -> bool:
    try:
        sig = inspect.signature(model.forward)
        return "keep_masks" in sig.parameters
    except Exception:
        # if signature introspection fails, assume it supports (your KO model does)
        return True


@torch.no_grad()
def eval_one_task_combo(
    model: nn.Module,
    dl: DataLoader,
    device: torch.device,
    task_now: str,
    keep_triplet: tuple[int, int, int],
    supports_keep_masks: bool,
):
    ce_tasks = {'length-of-stay', 'drg'}

    outGT = torch.FloatTensor().to(device)
    outPRED = torch.FloatTensor().to(device)

    # For reporting how many rows actually contributed (combo-available rows)
    n_rows_used = 0

    for data in dl:
        ehr, ehr_length, mask_ehr, cxr, mask_cxr, note, mask_note, label, task_index = data

        ehr_t = torch.from_numpy(ehr).float().to(device)
        cxr_t = cxr.to(device)

        avail_ehr  = torch.from_numpy(mask_ehr).long().to(device)
        avail_cxr  = torch.from_numpy(mask_cxr).long().to(device)
        avail_note = torch.from_numpy(mask_note).long().to(device)

        y_true = torch.from_numpy(label).float().to(device)
        task_index_t = torch.from_numpy(task_index).long().to(device)

        # requested modalities for this combo (constants per batch)
        req_ehr  = int(keep_triplet[0])
        req_cxr  = int(keep_triplet[1])
        req_note = int(keep_triplet[2])

        # keep_masks implements the combo but never keeps unavailable modalities
        keep_masks = {
            "ehr":  (avail_ehr  * req_ehr).long(),
            "cxr":  (avail_cxr  * req_cxr).long(),
            "note": (avail_note * req_note).long(),
        }

        # IMPORTANT: only score rows where at least one *requested* modality exists
        # e.g., for CXR-only: score only rows with avail_cxr==1
        kept_row = (keep_masks["ehr"] + keep_masks["cxr"] + keep_masks["note"]) > 0  # (B,)
        if kept_row.sum().item() == 0:
            continue  # nothing in this batch is eligible for this combo

        if supports_keep_masks:
            pack = model(
                ehr_t, ehr_length, avail_ehr,
                cxr_t, avail_cxr,
                note, avail_note,
                task_index_t,
                keep_masks=keep_masks,
            )
        else:
            pack = model(
                ehr_t, ehr_length, avail_ehr,
                cxr_t, avail_cxr,
                note, avail_note,
                task_index_t,
            )

        y_out = pack[0] if isinstance(pack, (tuple, list)) else pack
        y_out = y_out.reshape(ehr_t.shape[0], -1)

        # Filter to eligible rows ONLY
        y_out = y_out[kept_row]
        y_true_kept = y_true[kept_row]
        n_rows_used += int(kept_row.sum().item())

        if task_now in ce_tasks:
            # CE: logits -> probs; ignore -1 labels
            y_true_ce = y_true_kept.long().view(-1)
            valid = (y_true_ce != -1)
            if valid.any():
                y_prob = torch.softmax(y_out[valid], dim=1)
                outPRED = torch.cat((outPRED, y_prob), 0)
                outGT   = torch.cat((outGT,   y_true_ce[valid]), 0)
        else:
            # BCE/multilabel
            outPRED = torch.cat((outPRED, y_out), 0)
            outGT   = torch.cat((outGT,   y_true_kept), 0)

    if outGT.numel() == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), 0

    yt, yp = _ensure_metric_shapes(task_now, outGT, outPRED)

    # AUC/AUPR
    if task_now in ce_tasks:
        auc, aupr = multiclass_auc_aupr(yt, yp)
        n_valid = int(yt.shape[0])
    else:
        auc, aupr = my_metrics(yt, yp, task_now)
        # valid entries are those with yt>=0, after row filtering
        n_valid = int((yt >= 0).sum().item())

    # ACC/F1/SENS/SPEC
    acc, f1, sens, spec = compute_all_metrics(task_now, yt, yp)

    return float(auc), float(aupr), float(acc), float(f1), float(sens), float(spec), int(n_valid)


# =========================
# Main
# =========================
def main():
    parser = args_parser()

    # NOTE: do NOT add --num_workers here; args_parser already defines it -> avoids argparse conflict
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="Checkpoint path (e.g., checkpoints/shapley_ko_flexcare_best_seed40_*.pt)")
    parser.add_argument("--normalizer_dir", type=str, default="",
                        help="Optional directory containing normalizer files.")
    parser.add_argument("--combos", type=str, default="ALL",
                        help="ALL or comma-separated subset of FULL,EHR,CXR,NOTE,EHR+CXR,EHR+NOTE,CXR+NOTE")
    parser.add_argument("--report_csv", type=str, default="",
                        help="Optional output CSV path; default results/shapley_ko_combo_report_*.csv")

    args, _ = parser.parse_known_args()
    task_list = args.task.split(",")

    # device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(getattr(args, "device", 0))
    device = torch.device("cuda" if (getattr(args, "device", "cpu") != "cpu" and torch.cuda.is_available()) else "cpu")
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    # ---- data ----
    discretizer = Discretizer(
        timestep=float(args.timestep),
        store_masks=True,
        impute_strategy="previous",
        start_time="zero",
    )

    def cont_channels_from_template():
        p = f"{args.ehr_path}/10002430_episode1_timeseries.csv"
        with open(p, "r") as tsfile:
            header = tsfile.readline().strip().split(",")
        return [i for (i, x) in enumerate(header) if x.find("->") == -1 and x != "Hours"]

    test_loaders = []
    for task in task_list:
        normalizer = Normalizer(fields=cont_channels_from_template())
        norm_path = pick_normalizer_state(args, task)
        normalizer.load_params(norm_path)
        print(f"[NORM] task={task} -> {norm_path}")

        _, _, test_ds = get_multimodal_datasets(discretizer, normalizer, args, task)
        dl_test = DataLoader(
            test_ds,
            batch_size=int(args.batch_size),
            shuffle=False,
            collate_fn=my_collate,
            pin_memory=True,
            num_workers=int(getattr(args, "num_workers", 0)),
            drop_last=False,
        )
        test_loaders.append(dl_test)
        print(f"[DATA][TEST] Task={task} | test={len(test_ds)}")

    # ---- model (MUST match training hyperparams) ----
    model = FlexCare(
        hidden_dim=int(args.hidden_dim),
        layers=4,
        expert_k=2,
        expert_total=10,
        device=device,
        normalize_before_placeholder=True,
    ).to(device)

    print(f"[LOAD] Loading checkpoint from {args.ckpt_path}")
    load_checkpoint_filtered(model, args.ckpt_path, device)
    model.eval()

    supports_keep_masks = forward_supports_keep_masks(model)
    print(f"[MODEL] forward supports keep_masks = {supports_keep_masks}")

    # combos selection
    all_combos = dict(modality_combos())
    if args.combos.strip().upper() == "ALL":
        run_keys = list(all_combos.keys())
    else:
        run_keys = [k.strip() for k in args.combos.split(",") if k.strip()]
    for k in run_keys:
        if k not in all_combos:
            raise ValueError(f"Unknown combo '{k}'. Allowed: {list(all_combos.keys())}")

    # ---- eval ----
    rows = []
    with torch.no_grad():
        for combo_key in run_keys:
            keep_triplet = all_combos[combo_key]
            print(f"\n======================\n[COMBO] {combo_key} keep={keep_triplet}\n======================")

            for t_idx, task_now in enumerate(task_list):
                auc, aupr, acc, f1, sens, spec, n_valid = eval_one_task_combo(
                    model=model,
                    dl=test_loaders[t_idx],
                    device=device,
                    task_now=task_now,
                    keep_triplet=keep_triplet,
                    supports_keep_masks=supports_keep_masks,
                )
                print(
                    f"[TEST] Task={task_now:20s} Combo={combo_key:9s} "
                    f"AUC={auc:.4f}  AUPR={aupr:.4f}  "
                    f"ACC={acc:.4f}  F1={f1:.4f}  SENS={sens:.4f}  SPEC={spec:.4f}  "
                    f"n_valid={n_valid}"
                )
                rows.append({
                    "task": task_now,
                    "combo": combo_key,
                    "AUC": float(auc),
                    "AUPR": float(aupr),
                    "ACC": float(acc),
                    "F1": float(f1),
                    "SENS": float(sens),
                    "SPEC": float(spec),
                    "n_valid": int(n_valid),
                })

    # ---- save CSV ----
    os.makedirs("results", exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = args.report_csv.strip()
    if not out_csv:
        out_csv = osp.join("results", f"shapley_ko_combo_report_seed{args.seed}_{stamp}.csv")

    fieldnames = ["task", "combo", "AUC", "AUPR", "ACC", "F1", "SENS", "SPEC", "n_valid"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"\n[TEST] Saved report to {out_csv}")


if __name__ == "__main__":
    main()

"""
SimMLM (DMoME + MoFe) for multimodal survival prediction.

Adapts Li et al., 2025 (https://arxiv.org/abs/2507.19264) to a 4-modality
survival setting (path, rad, demo, omic).

1) Dynamic Mixture of Modality Experts (DMoME):
   - One "expert" hazard head per modality (linear on top of the existing
     fc1..fc4 projections).
   - A small gating network maps the binary modality code (B, 4) to gating
     logits (B, 4); missing modalities are masked to -inf so their softmax
     weight is 0. The final hazard is the weighted sum of expert hazards.

2) More vs. Fewer (MoFe) ranking loss:
   - Per training batch, we forward with full modalities (x+) and with the
     dropped modalities provided by the dataloader (x-). The Cox losses
     are L+ and L-. We add max(L+ - L-, 0) so the network is encouraged
     to perform at least as well with more modalities as with fewer.

The setup mirrors test_base.py: SWA, 30 epochs, val C-index for selection.
"""

import os
import pickle
import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, SWALR

from data_loader import PathgraphomicDatasetLoader, PathgraphomicFastDatasetLoader
from options import parse_args
from networks import define_net, define_optimizer, define_scheduler, define_reg
from utils_all import complete_incomplete_data_selection, CIndex_lifeline


NUM_MOD = 4


# --------------------------------------------------------------------------
def CoxLoss(survtime, censor, hazard_pred, device):
    n = len(survtime)
    R = np.zeros([n, n], dtype=int)
    for i in range(n):
        for j in range(n):
            R[i, j] = survtime[j] >= survtime[i]
    R = torch.FloatTensor(R).to(device)
    theta = hazard_pred.reshape(-1)
    exp_theta = torch.exp(theta)
    censor = censor.to(device)
    return -torch.mean(
        (theta - torch.log(torch.sum(exp_theta * R, dim=1) + 1e-8)) * censor
    )


# --------------------------------------------------------------------------
class SimMLMModel(nn.Module):
    """DMoME on top of the per-modality fc1..fc4 projections."""

    def __init__(self, opt, k):
        super().__init__()
        self.opt = opt
        bb = define_net(opt, k)
        # define_net wraps in nn.DataParallel when gpu_ids is non-empty;
        # we need direct module access for fc1..fc4 etc.
        if isinstance(bb, nn.DataParallel):
            bb = bb.module
        self.backbone = bb
        H = opt.mmhid

        # Per-modality experts: simple linear hazard heads on each
        # projected representation.
        self.expert_heads = nn.ModuleList([nn.Linear(H, opt.label_dim) for _ in range(NUM_MOD)])

        # Gating network: takes the binary modality code (4) -> 4 gating logits.
        self.gate = nn.Sequential(
            nn.Linear(NUM_MOD, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, NUM_MOD),
        )

        # Reuse output range/shift from the backbone for Sigmoid scaling.
        self.output_range = self.backbone.output_range
        self.output_shift = self.backbone.output_shift
        self.act = self.backbone.act

    def encode_modalities(self, kwargs):
        bb = self.backbone
        if self.opt.use_embedding:
            x_path = kwargs['x_path']
            x_rad = kwargs['x_rad']
            x_demo = kwargs['x_demo']
            x_omic = kwargs['x_omic']
        else:
            x_path, _ = bb.path_net(x_path=kwargs['x_path'])
            x_rad, _ = bb.rad_net(x_rad=kwargs['x_rad'], x_radiomics=kwargs['x_radiomics'])
            x_demo, _ = bb.demo_net(x_demo=kwargs['x_demo'])
            x_omic, _ = bb.omic_net(x_omic=kwargs['x_omic'])

        z_path = F.relu(bb.fc1(x_path.view(x_path.size(0), -1)))
        z_rad = F.relu(bb.fc2(x_rad.view(x_rad.size(0), -1)))
        z_demo = F.relu(bb.fc3(x_demo.view(x_demo.size(0), -1)))
        z_omic = F.relu(bb.fc4(x_omic.view(x_omic.size(0), -1)))
        return [z_path, z_rad, z_demo, z_omic]

    def _gate_and_combine(self, logits_per_mod, keep):
        gate_logits = self.gate(keep)
        # Mask missing modalities to -inf so their softmax weight is 0
        gate_logits = gate_logits.masked_fill(keep < 0.5, float('-inf'))
        weights = F.softmax(gate_logits, dim=-1)  # (B, M)
        weights = torch.nan_to_num(weights, nan=0.0)  # safety for all-zero rows

        hazard = (weights * logits_per_mod).sum(dim=-1, keepdim=True)
        if self.act is not None:
            hazard = self.act(hazard)
            if isinstance(self.act, nn.Sigmoid):
                hazard = hazard * self.output_range + self.output_shift
        return hazard, weights

    def _forward_with_keep(self, kwargs, keep):
        z_list = self.encode_modalities(kwargs)
        logits_per_mod = torch.cat(
            [head(z) for head, z in zip(self.expert_heads, z_list)],
            dim=1)  # (B, M)
        hazard, weights = self._gate_and_combine(logits_per_mod, keep)
        return hazard, weights, logits_per_mod

    def forward(self, **kwargs):
        keep = kwargs['x_keep_masks']
        hazard, weights, logits_per_mod = self._forward_with_keep(kwargs, keep)
        return {"weights": weights, "expert_logits": logits_per_mod}, hazard

    def forward_pair(self, **kwargs):
        """Returns (hazard_full, hazard_partial) for the same batch.
        Encodes modalities and computes per-expert logits ONCE; only the
        gating/combination step depends on `keep`."""
        keep_partial = kwargs['x_keep_masks']
        keep_full = torch.ones_like(keep_partial)
        z_list = self.encode_modalities(kwargs)
        logits_per_mod = torch.cat(
            [head(z) for head, z in zip(self.expert_heads, z_list)],
            dim=1)  # (B, M)
        h_full, _ = self._gate_and_combine(logits_per_mod, keep_full)
        h_part, _ = self._gate_and_combine(logits_per_mod, keep_partial)
        return h_full, h_part


# --------------------------------------------------------------------------
@torch.no_grad()
def eval_subset(opt, model, data, mask, device, force_keep):
    model.eval()
    loader = torch.utils.data.DataLoader(
        PathgraphomicFastDatasetLoader(opt, data, mask, split="test"),
        batch_size=opt.batch_test_size, shuffle=False)
    risks, censors, surv = [], [], []
    for batch in loader:
        _, x_path, x_omic, x_rad, x_demo, x_radiomics, censor, survtime, x_masks, _ = batch
        bs = x_path.size(0)
        keep = force_keep.unsqueeze(0).repeat(bs, 1).to(device)
        _, pred = model(
            x_path=x_path.to(device), x_omic=x_omic.to(device),
            x_rad=x_rad.to(device), x_demo=x_demo.to(device),
            x_radiomics=x_radiomics.to(device),
            x_masks={m: v.to(device) for m, v in x_masks.items()},
            x_keep_masks=keep)
        risks.extend(pred.detach().cpu().numpy().flatten())
        censors.extend(censor.numpy().flatten())
        surv.extend(survtime.numpy().flatten())
    cidx = CIndex_lifeline(np.array(risks), np.array(censors), np.array(surv))
    loss = CoxLoss(torch.tensor(surv, device=device),
                   torch.tensor(censors, device=device),
                   torch.tensor(risks, device=device, dtype=torch.float32),
                   device)
    return loss.item(), cidx


@torch.no_grad()
def eval_val_full(opt, model, data, mask, device):
    model.eval()
    loader = torch.utils.data.DataLoader(
        PathgraphomicFastDatasetLoader(opt, data, mask, split="val"),
        batch_size=opt.batch_test_size, shuffle=False)
    risks, censors, surv = [], [], []
    for batch in loader:
        _, x_path, x_omic, x_rad, x_demo, x_radiomics, censor, survtime, x_masks, _ = batch
        bs = x_path.size(0)
        keep = torch.ones((bs, 4), dtype=torch.float32, device=device)
        _, pred = model(
            x_path=x_path.to(device), x_omic=x_omic.to(device),
            x_rad=x_rad.to(device), x_demo=x_demo.to(device),
            x_radiomics=x_radiomics.to(device),
            x_masks={m: v.to(device) for m, v in x_masks.items()},
            x_keep_masks=keep)
        risks.extend(pred.detach().cpu().numpy().flatten())
        censors.extend(censor.numpy().flatten())
        surv.extend(survtime.numpy().flatten())
    cidx = CIndex_lifeline(np.array(risks), np.array(censors), np.array(surv))
    loss = CoxLoss(torch.tensor(surv, device=device),
                   torch.tensor(censors, device=device),
                   torch.tensor(risks, device=device, dtype=torch.float32),
                   device)
    return loss.item(), cidx


# --------------------------------------------------------------------------
def train_simmlm(opt, data, mask, device, k, mofe_lambda=0.1):
    model = SimMLMModel(opt, k).to(device)
    optimizer = define_optimizer(opt, model)

    opt.niter = 10
    opt.niter_decay = 20
    scheduler = define_scheduler(opt, optimizer)

    swa_model = AveragedModel(model)
    swa_start = int((opt.niter + opt.niter_decay) * 0.75)
    swa_scheduler = SWALR(optimizer, swa_lr=opt.lr * 0.1)

    train_loader = torch.utils.data.DataLoader(
        PathgraphomicDatasetLoader(opt, data, mask, split="train"),
        batch_size=opt.batch_size, shuffle=True)

    history = {"epoch": [], "val_cindex": [], "val_loss": []}
    best_cindex = -1.0

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        model.train()
        for batch in train_loader:
            _, x_path, x_omic, x_rad, x_demo, x_radiomics, censor, survtime, x_masks, x_keep = batch
            x_keep = x_keep.to(device).float()

            kwargs = dict(
                x_path=x_path.to(device), x_omic=x_omic.to(device),
                x_rad=x_rad.to(device), x_demo=x_demo.to(device),
                x_radiomics=x_radiomics.to(device),
                x_masks={m: v.to(device) for m, v in x_masks.items()},
                x_keep_masks=x_keep)

            h_full, h_part = model.forward_pair(**kwargs)
            l_full = CoxLoss(survtime, censor, h_full, device)
            l_part = CoxLoss(survtime, censor, h_part, device)
            l_mofe = torch.clamp(l_full - l_part, min=0.0)

            loss = opt.lambda_cox * (l_full + l_part) + mofe_lambda * l_mofe \
                   + define_reg(opt, model)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch > swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

        eval_model = swa_model if epoch > swa_start else model
        v_loss, c_val = eval_val_full(opt, eval_model, data, mask, device)
        history["epoch"].append(epoch)
        history["val_cindex"].append(c_val)
        history["val_loss"].append(v_loss)

        if c_val > best_cindex:
            best_cindex = c_val
            os.makedirs(opt.checkpoints_dir, exist_ok=True)
            torch.save(eval_model.state_dict(),
                       os.path.join(opt.checkpoints_dir, f"simmlm_fold_{k}.pt"))

    with open(os.path.join(opt.checkpoints_dir, f"simmlm_history_fold_{k}.pkl"), "wb") as f:
        pickle.dump(history, f)
    return swa_model, best_cindex


# --------------------------------------------------------------------------
def evaluate_all_subsets(opt, data_cv, mask_cv, device, fold_range):
    modalities = ["Pathology", "Radiology", "Demographics", "Genomics"]
    d = len(modalities)
    rows = []
    for k in fold_range:
        ckpt = os.path.join(opt.checkpoints_dir, f"simmlm_fold_{k}.pt")
        if not os.path.exists(ckpt):
            print(f"[skip] fold {k}: no checkpoint")
            continue
        model = SimMLMModel(opt, k=k).to(device)
        state = torch.load(ckpt, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        clean = {}
        for kk, vv in state.items():
            if kk == "n_averaged":
                continue
            clean[kk[len("module."):] if kk.startswith("module.") else kk] = vv
        miss, unexp = model.load_state_dict(clean, strict=False)
        if miss: print(f"  missing: {miss[:5]}")
        if unexp: print(f"  unexpected: {unexp[:5]}")
        model.eval()

        for r in range(1, d + 1):
            for combo in itertools.combinations(range(d), r):
                label = "+".join([modalities[i] for i in combo])
                force = torch.zeros(d, device=device)
                for idx in combo:
                    force[idx] = 1
                loss, ci = eval_subset(opt, model, data_cv[k], mask_cv[k], device, force)
                rows.append({"Fold": k, "Subset": label, "C-Index": ci, "Loss": loss})

    df = pd.DataFrame(rows)
    if df.empty:
        print("No SimMLM results.")
        return None
    summary = df.groupby("Subset")["C-Index"].agg(["mean", "std"]).reset_index()
    summary["Result"] = summary.apply(lambda x: f"{x['mean']:.4f} ± {x['std']:.3f}", axis=1)
    print("\n--- SimMLM Aggregated Results ---")
    print(summary[["Subset", "Result"]].to_string(index=False))

    df.to_csv(os.path.join(opt.checkpoints_dir, "simmlm_anysubset_results.csv"), index=False)
    summary.to_csv(os.path.join(opt.checkpoints_dir, "simmlm_anysubset_summary.csv"), index=False)
    return summary


# --------------------------------------------------------------------------
if __name__ == "__main__":
    opt = parse_args()
    opt.rad_dir = os.path.join(opt.dataroot, "radiology")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_cv = pickle.load(open(os.path.join(opt.dataroot, "gbmlgg15cv_embedding.pkl"), "rb"))
    mask_cv = pickle.load(open(os.path.join(opt.dataroot, "mask_gbmlgg15cv.pkl"), "rb"))

    data_cv_comp, mask_cv_comp = complete_incomplete_data_selection(
        opt, data_cv["cv_splits"], mask_cv["cv_splits"],
        os.path.join(opt.dataroot, "img_availability.csv"),
        opt.required_modality)

    fold_range = [i + 1 for i in range(15)]

    for k in fold_range:
        ckpt = os.path.join(opt.checkpoints_dir, f"simmlm_fold_{k}.pt")
        if os.path.exists(ckpt):
            print(f"--- SimMLM fold {k}: checkpoint exists, skipping training ---")
            continue
        print(f"\n--- SimMLM training: fold {k}/15 ---")
        train_simmlm(opt, data_cv_comp[k], mask_cv_comp[k], device, k)

    print("\n=== SimMLM evaluation across all modality subsets ===")
    evaluate_all_subsets(opt, data_cv_comp, mask_cv_comp, device, fold_range)

"""
PMR (Prototypical Modal Rebalance) for multimodal survival prediction.

Adapts Fan et al., 2022 (https://arxiv.org/abs/2211.07089) to a 4-modality
survival setting (path, rad, demo, omic). The original PMR was proposed
for classification; we adapt it by treating the binary censoring/event
indicator as the class label (0 = censored, 1 = event), and using
modality representations from the existing fc1..fc4 projections.

Two key ingredients on top of the standard joint Cox training:

1) Prototypical CE (PCE) loss: per-modality, distance-based softmax over
   class prototypes (centroids of training representations per class).
   PCE acts as an internal driver for the slow-learning modality and
   does not depend on the fused logit.

2) Prototypical Entropy Regularization (PER): an entropy term on the
   prototype-distance distribution of the dominant modality, which
   prevents premature convergence and lets the slower modality catch up.
   Applied only in the early epochs.

The dominant/slow modalities are identified per-step via a prototype-based
imbalance ratio (per-modality probability of the true class summed over
the batch); the larger one is the dominant modality for that step.
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
NUM_CLASS = 2  # censor / event


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
class PMRModel(nn.Module):
    """Wraps Multimodal_fusion and exposes per-modality projected features."""

    def __init__(self, opt, k):
        super().__init__()
        self.opt = opt
        bb = define_net(opt, k)
        # define_net wraps in nn.DataParallel when gpu_ids is non-empty;
        # we need direct module access for fc1..fc4 etc.
        if isinstance(bb, nn.DataParallel):
            bb = bb.module
        self.backbone = bb
        self.hidden_dim = opt.mmhid

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

    def fuse_and_predict(self, z_list, keep_mask):
        bb = self.backbone
        z_stack = torch.stack(z_list, dim=1)  # (B, 4, H)
        keep_b = keep_mask[..., None].float()
        denom = keep_b.sum(dim=1).clamp_min(1e-8)
        fused = (z_stack * keep_b).sum(dim=1) / denom
        feats = bb.fuse_fc(fused)
        hazard = bb.classifier(feats)
        if bb.act is not None:
            hazard = bb.act(hazard)
            if isinstance(bb.act, nn.Sigmoid):
                hazard = hazard * bb.output_range + bb.output_shift
        return fused, hazard

    def forward(self, **kwargs):
        z_list = self.encode_modalities(kwargs)
        _, hazard = self.fuse_and_predict(z_list, kwargs['x_keep_masks'])
        return {"z_list": z_list}, hazard


# --------------------------------------------------------------------------
# Prototype bank (per modality, per class)
# --------------------------------------------------------------------------
class PrototypeBank:
    def __init__(self, num_mod, num_class, hidden_dim, device, momentum=0.9):
        self.num_mod = num_mod
        self.num_class = num_class
        self.dim = hidden_dim
        self.device = device
        self.momentum = momentum
        # (M, C, H)
        self.protos = torch.zeros(num_mod, num_class, hidden_dim, device=device)
        self.initialized = False

    @torch.no_grad()
    def recompute_from_loader(self, model, loader, device):
        """Pass over the loader, compute centroids per (modality, class)."""
        sums = torch.zeros(self.num_mod, self.num_class, self.dim, device=device)
        counts = torch.zeros(self.num_mod, self.num_class, device=device)
        model.eval()
        for batch in loader:
            _, x_path, x_omic, x_rad, x_demo, x_radiomics, censor, _, x_masks, x_keep = batch
            kwargs = dict(
                x_path=x_path.to(device), x_omic=x_omic.to(device),
                x_rad=x_rad.to(device), x_demo=x_demo.to(device),
                x_radiomics=x_radiomics.to(device),
                x_masks={m: v.to(device) for m, v in x_masks.items()},
                x_keep_masks=x_keep.to(device).float())
            z_list = model.encode_modalities(kwargs)
            keep = kwargs['x_keep_masks']  # (B,4)
            y = censor.long().to(device)  # 0 or 1
            for m, z in enumerate(z_list):
                avail = keep[:, m] > 0.5
                if avail.sum() == 0:
                    continue
                z_a = z[avail]
                y_a = y[avail]
                for c in range(self.num_class):
                    mask_c = (y_a == c)
                    if mask_c.sum() > 0:
                        sums[m, c] += z_a[mask_c].sum(dim=0)
                        counts[m, c] += mask_c.sum().float()
        new_protos = sums / counts.clamp_min(1.0)[..., None]
        if self.initialized:
            self.protos = self.momentum * self.protos + (1 - self.momentum) * new_protos
        else:
            self.protos = new_protos
            self.initialized = True
        model.train()

    def proto_logits(self, z, m_idx):
        """Return (B, C) negative-distance logits for modality m."""
        # z: (B, H), protos[m]: (C, H)
        diff = z[:, None, :] - self.protos[m_idx][None, :, :]  # (B, C, H)
        d = (diff * diff).sum(dim=-1)  # squared euclidean
        return -d

    def proto_probs(self, z, m_idx):
        return F.softmax(self.proto_logits(z, m_idx), dim=-1)


# --------------------------------------------------------------------------
# Eval engines
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
# Training
# --------------------------------------------------------------------------
def train_pmr(opt, data, mask, device, k,
              alpha=1.0, mu=0.01, regularize_epochs_frac=0.5):
    model = PMRModel(opt, k).to(device)
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

    bank = PrototypeBank(NUM_MOD, NUM_CLASS, opt.mmhid, device, momentum=0.9)

    history = {"epoch": [], "val_cindex": [], "val_loss": []}
    best_cindex = -1.0
    total_epochs = opt.niter + opt.niter_decay
    per_epoch_cutoff = int(total_epochs * regularize_epochs_frac)

    for epoch in range(opt.epoch_count, total_epochs + 1):
        # Recompute prototypes at the start of each epoch
        bank.recompute_from_loader(model, train_loader, device)

        model.train()
        for batch in train_loader:
            _, x_path, x_omic, x_rad, x_demo, x_radiomics, censor, survtime, x_masks, x_keep = batch
            x_keep = x_keep.to(device).float()
            y = censor.long().to(device)

            kwargs = dict(
                x_path=x_path.to(device), x_omic=x_omic.to(device),
                x_rad=x_rad.to(device), x_demo=x_demo.to(device),
                x_radiomics=x_radiomics.to(device),
                x_masks={m: v.to(device) for m, v in x_masks.items()},
                x_keep_masks=x_keep)

            z_list = model.encode_modalities(kwargs)
            _, hazard = model.fuse_and_predict(z_list, x_keep)
            cox = CoxLoss(survtime, censor, hazard, device)

            # PCE per modality (only over samples with that modality available)
            pce_per_mod = []
            true_class_prob = []  # batch-summed prob of correct class per modality
            for m, z in enumerate(z_list):
                avail = x_keep[:, m] > 0.5
                if avail.sum() == 0:
                    pce_per_mod.append(torch.tensor(0.0, device=device))
                    true_class_prob.append(torch.tensor(0.0, device=device))
                    continue
                logits = bank.proto_logits(z[avail], m)
                pce = F.cross_entropy(logits, y[avail])
                pce_per_mod.append(pce)

                with torch.no_grad():
                    p = F.softmax(logits, dim=-1)
                    tcp = p.gather(1, y[avail].unsqueeze(1)).sum()
                true_class_prob.append(tcp)

            tcp_tensor = torch.stack(true_class_prob)  # (M,)

            # Imbalance: weight each modality's PCE by clip(0, max - tcp_m, 1) / ref
            # Slow modality has small tcp -> larger weight; dominant modality -> 0 weight.
            tcp_max = tcp_tensor.max().clamp_min(1e-6)
            mod_weights = torch.clamp(tcp_max - tcp_tensor, min=0.0) / tcp_max  # in [0,1]
            pce_total = sum(w * p for w, p in zip(mod_weights, pce_per_mod))

            # PER on dominant modality (max tcp), only for early epochs
            per_total = torch.tensor(0.0, device=device)
            if epoch <= per_epoch_cutoff:
                dom_idx = int(torch.argmax(tcp_tensor).item())
                avail = x_keep[:, dom_idx] > 0.5
                if avail.sum() > 0:
                    p_dom = F.softmax(bank.proto_logits(z_list[dom_idx][avail], dom_idx), dim=-1)
                    ent = -(p_dom * torch.log(p_dom + 1e-8)).sum(dim=-1).mean()
                    per_total = ent

            loss = (opt.lambda_cox * cox
                    + alpha * pce_total
                    - mu * per_total
                    + define_reg(opt, model))

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
                       os.path.join(opt.checkpoints_dir, f"pmr_fold_{k}.pt"))

    with open(os.path.join(opt.checkpoints_dir, f"pmr_history_fold_{k}.pkl"), "wb") as f:
        pickle.dump(history, f)
    return swa_model, best_cindex


# --------------------------------------------------------------------------
def evaluate_all_subsets(opt, data_cv, mask_cv, device, fold_range):
    modalities = ["Pathology", "Radiology", "Demographics", "Genomics"]
    d = len(modalities)
    rows = []
    for k in fold_range:
        ckpt = os.path.join(opt.checkpoints_dir, f"pmr_fold_{k}.pt")
        if not os.path.exists(ckpt):
            print(f"[skip] fold {k}: no checkpoint")
            continue
        model = PMRModel(opt, k=k).to(device)
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
        print("No PMR results.")
        return None
    summary = df.groupby("Subset")["C-Index"].agg(["mean", "std"]).reset_index()
    summary["Result"] = summary.apply(lambda x: f"{x['mean']:.4f} ± {x['std']:.3f}", axis=1)
    print("\n--- PMR Aggregated Results ---")
    print(summary[["Subset", "Result"]].to_string(index=False))

    df.to_csv(os.path.join(opt.checkpoints_dir, "pmr_anysubset_results.csv"), index=False)
    summary.to_csv(os.path.join(opt.checkpoints_dir, "pmr_anysubset_summary.csv"), index=False)
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
        ckpt = os.path.join(opt.checkpoints_dir, f"pmr_fold_{k}.pt")
        if os.path.exists(ckpt):
            print(f"--- PMR fold {k}: checkpoint exists, skipping training ---")
            continue
        print(f"\n--- PMR training: fold {k}/15 ---")
        train_pmr(opt, data_cv_comp[k], mask_cv_comp[k], device, k)

    print("\n=== PMR evaluation across all modality subsets ===")
    evaluate_all_subsets(opt, data_cv_comp, mask_cv_comp, device, fold_range)

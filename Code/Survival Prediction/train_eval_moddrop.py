"""
ModDrop++ for multimodal survival prediction.

Adapts ModDrop++ (Liu et al., MICCAI 2022) to a 4-modality survival
setting (path, rad, demo, omic). Two upgrades on top of the existing
Multimodal_fusion baseline:

1) Dynamic head with filter scaling: a tiny MLP maps the binary modality
   code (B, 4) to per-modality, per-channel scale factors that modulate
   each modality's encoded features. This lets the network adapt to
   different missing conditions without changing the backbone.

2) Intra-subject co-training: for each batch we run TWO forward passes
   on the same samples - one with all modalities present (full), one
   with the actual (possibly dropped) modality mask. We then add a
   feature-similarity loss between the two fused features so the
   missing-modality features stay close to the full-modality ones.

Both passes contribute a Cox loss; the similarity term complements it.
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


# --------------------------------------------------------------------------
# Cox loss
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
# ModDrop++ wrapper around Multimodal_fusion
# --------------------------------------------------------------------------
class ModDropPlusPlus(nn.Module):
    """Wrap the existing Multimodal_fusion with a dynamic head and expose
    intermediate fused features for the co-training similarity loss."""

    NUM_MOD = 4  # path, rad, demo, omic

    def __init__(self, opt, k):
        super().__init__()
        self.opt = opt
        bb = define_net(opt, k)
        # define_net wraps in nn.DataParallel when gpu_ids is non-empty;
        # we need direct module access for fc1..fc4 etc.
        if isinstance(bb, nn.DataParallel):
            bb = bb.module
        self.backbone = bb
        # Dimensionality of fused features just before fuse_fc / classifier
        self.hidden_dim = opt.mmhid

        # Dynamic head: produces per-modality, per-channel scaling factors.
        # Input: modality code (B, 4); output: (B, 4 * hidden_dim) -> reshape (B, 4, hidden_dim)
        self.dynamic_head = nn.Sequential(
            nn.Linear(self.NUM_MOD, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, self.NUM_MOD * self.hidden_dim),
        )
        # Initialize so the initial scaling is ~1 (identity-like behavior).
        nn.init.zeros_(self.dynamic_head[-1].weight)
        nn.init.zeros_(self.dynamic_head[-1].bias)

    # --- Internal: replicate the per-modality encoding from Multimodal_fusion
    def _encode(self, kwargs):
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
        return torch.stack([z_path, z_rad, z_demo, z_omic], dim=1)  # (B, 4, H)

    def _forward_with_keep(self, kwargs, keep):
        """Forward pass conditioned on a given keep-mask `keep` (B, 4).
        Returns (fused_pre, hazard) so caller can use the fused features."""
        bb = self.backbone
        z = self._encode(kwargs)  # (B, 4, H)

        # Dynamic filter scaling: scales = 1 + delta(m); delta starts at 0.
        delta = self.dynamic_head(keep).view(z.size(0), self.NUM_MOD, self.hidden_dim)
        scales = 1.0 + delta
        z = z * scales

        # Masked average using the same keep mask
        keep_b = keep[..., None].float()
        denom = keep_b.sum(dim=1).clamp_min(1e-8)
        fused = (z * keep_b).sum(dim=1) / denom

        feats = bb.fuse_fc(fused)
        hazard = bb.classifier(feats)
        if bb.act is not None:
            hazard = bb.act(hazard)
            if isinstance(bb.act, nn.Sigmoid):
                hazard = hazard * bb.output_range + bb.output_shift
        return fused, hazard

    def forward(self, **kwargs):
        keep = kwargs['x_keep_masks']
        fused, hazard = self._forward_with_keep(kwargs, keep)
        # Match baseline contract: (extras_dict, hazard)
        return {"fused": fused}, hazard

    def forward_full_and_drop(self, **kwargs):
        """Used during training: returns (fused_full, hazard_full, fused_drop, hazard_drop)."""
        keep_drop = kwargs['x_keep_masks']
        keep_full = torch.ones_like(keep_drop)
        fused_full, hazard_full = self._forward_with_keep(kwargs, keep_full)
        fused_drop, hazard_drop = self._forward_with_keep(kwargs, keep_drop)
        return fused_full, hazard_full, fused_drop, hazard_drop


def feature_similarity_loss(f_full, f_drop):
    """Cosine-based similarity loss in feature space.
    Lower is better when features align in direction; we return 1-cos so
    minimizing the loss maximizes similarity."""
    f_full_n = F.normalize(f_full, dim=-1)
    f_drop_n = F.normalize(f_drop, dim=-1)
    cos = (f_full_n * f_drop_n).sum(dim=-1)
    return (1.0 - cos).mean()


# --------------------------------------------------------------------------
# Eval engine (single forced subset)
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
# Training loop
# --------------------------------------------------------------------------
def train_moddrop(opt, data, mask, device, k, sim_weight=0.05):
    model = ModDropPlusPlus(opt, k).to(device)
    optimizer = define_optimizer(opt, model)

    opt.niter = 10
    opt.niter_decay = 20
    scheduler = define_scheduler(opt, optimizer)

    swa_model = AveragedModel(model)
    swa_start = int((opt.niter + opt.niter_decay) * 0.75)
    swa_scheduler = SWALR(optimizer, swa_lr=opt.lr * 0.1)

    best_cindex = -1.0
    train_loader = torch.utils.data.DataLoader(
        PathgraphomicDatasetLoader(opt, data, mask, split="train"),
        batch_size=opt.batch_size, shuffle=True)

    history = {"epoch": [], "val_cindex": [], "val_loss": []}

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

            f_full, h_full, f_drop, h_drop = model.forward_full_and_drop(**kwargs)

            loss_full = CoxLoss(survtime, censor, h_full, device)
            loss_drop = CoxLoss(survtime, censor, h_drop, device)
            loss_sim = feature_similarity_loss(f_full.detach(), f_drop)

            loss = opt.lambda_cox * (loss_full + loss_drop) + sim_weight * loss_sim \
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
                       os.path.join(opt.checkpoints_dir, f"moddrop_fold_{k}.pt"))

    with open(os.path.join(opt.checkpoints_dir, f"moddrop_history_fold_{k}.pkl"), "wb") as f:
        pickle.dump(history, f)
    return swa_model, best_cindex


# --------------------------------------------------------------------------
# Per-fold evaluation across all subsets
# --------------------------------------------------------------------------
def evaluate_all_subsets(opt, data_cv, mask_cv, device, fold_range):
    modalities = ["Pathology", "Radiology", "Demographics", "Genomics"]
    d = len(modalities)
    rows = []

    for k in fold_range:
        ckpt_path = os.path.join(opt.checkpoints_dir, f"moddrop_fold_{k}.pt")
        if not os.path.exists(ckpt_path):
            print(f"[skip] fold {k}: no checkpoint")
            continue
        model = ModDropPlusPlus(opt, k=k).to(device)
        # SWA-saved state dict may have AveragedModel internals; load with strict=False
        state = torch.load(ckpt_path, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        # Strip AveragedModel "module." prefix if present
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
        print("No moddrop results.")
        return None

    summary = df.groupby("Subset")["C-Index"].agg(["mean", "std"]).reset_index()
    summary["Result"] = summary.apply(lambda x: f"{x['mean']:.4f} ± {x['std']:.3f}", axis=1)
    print("\n--- ModDrop++ Aggregated Results ---")
    print(summary[["Subset", "Result"]].to_string(index=False))

    df.to_csv(os.path.join(opt.checkpoints_dir, "moddrop_anysubset_results.csv"), index=False)
    summary.to_csv(os.path.join(opt.checkpoints_dir, "moddrop_anysubset_summary.csv"), index=False)
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
        ckpt = os.path.join(opt.checkpoints_dir, f"moddrop_fold_{k}.pt")
        if os.path.exists(ckpt):
            print(f"--- ModDrop++ fold {k}: checkpoint exists, skipping training ---")
            continue
        print(f"\n--- ModDrop++ training: fold {k}/15 ---")
        train_moddrop(opt, data_cv_comp[k], mask_cv_comp[k], device, k)

    print("\n=== ModDrop++ evaluation across all modality subsets ===")
    evaluate_all_subsets(opt, data_cv_comp, mask_cv_comp, device, fold_range)

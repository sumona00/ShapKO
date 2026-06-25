# train_test.py
from __future__ import annotations

import os
import pickle
import random
import argparse
import logging
from collections import OrderedDict
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from tqdm import tqdm

from data_loader import PathgraphomicFastDatasetLoader, PathgraphomicDatasetLoader
from networks import define_net, define_reg, define_optimizer, define_scheduler
from utils import CoxLoss, CIndex_lifeline, cox_log_rank, accuracy_cox, count_parameters
from utils import complete_incomplete_data_selection

import numpy as numpy_pkg
import torch.serialization
torch.serialization.add_safe_globals([numpy_pkg._core.multiarray._reconstruct])


# -------------------------
# Loader / checkpoint helpers
# -------------------------
def strip_module_prefix(state_dict):
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    new_sd = OrderedDict()
    for k, v in state_dict.items():
        new_sd[k.replace("module.", "", 1)] = v
    return new_sd


def load_checkpoint_trusted(path, device):
    return torch.load(path, map_location=device, weights_only=False)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(0)


def _init_fn(worker_id):
    np.random.seed(int(2019))


# ============================================================
# OGM-GE (Online Gradient Modulation + Gradient Enhancement)
# ============================================================
def _collect_branch_params(model: nn.Module, branch_keys: List[str]) -> Dict[str, List[nn.Parameter]]:
    out: Dict[str, List[nn.Parameter]] = {k: [] for k in branch_keys}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        for k in branch_keys:
            if k in name:
                out[k].append(p)
                break
    return out


def _grad_l2_norm(params: List[nn.Parameter], eps: float = 1e-12) -> float:
    s = 0.0
    for p in params:
        if p.grad is None:
            continue
        g = p.grad.detach()
        s += float(torch.sum(g * g).item())
    return float((s + eps) ** 0.5)


@torch.no_grad()
def apply_ogm_ge(
    model: nn.Module,
    branch_keys: List[str],
    alpha: float = 0.5,
    beta: float = 0.5,
    eps: float = 1e-8,
    ratio_clip: float = 10.0,
) -> Dict[str, float]:
    """
    Practical OGM-GE:
    - compute grad norm per branch (by parameter-name substring match)
    - dominant branch gets suppressed: scale = (1 - alpha)
    - others get enhanced: scale = 1 + beta * (ratio - 1), ratio = dom_norm / (norm + eps), clipped
    """
    branch_params = _collect_branch_params(model, branch_keys)
    norms = {k: _grad_l2_norm(branch_params[k], eps=eps) for k in branch_keys}

    dom_key = max(norms.keys(), key=lambda k: norms[k])
    dom_norm = norms[dom_key]

    if not np.isfinite(dom_norm) or dom_norm <= 0.0:
        return {f"gnorm/{k}": float(norms[k]) for k in norms}

    for k in branch_keys:
        params = branch_params[k]
        if len(params) == 0:
            continue

        if k == dom_key:
            scale = max(0.0, 1.0 - float(alpha))
        else:
            denom = float(norms[k]) + float(eps)
            ratio = float(dom_norm) / denom
            ratio = max(1.0, min(float(ratio_clip), ratio))
            scale = 1.0 + float(beta) * (ratio - 1.0)

        for p in params:
            if p.grad is None:
                continue
            p.grad.mul_(scale)

    log = {f"gnorm/{k}": float(norms[k]) for k in norms}
    log["ogmge/dom_norm"] = float(dom_norm)
    log["ogmge/dom_key_idx"] = float(branch_keys.index(dom_key))
    return log


# ============================================================
# Core train/test
# ============================================================
def loss_add(opt, output):
    add_loss = 0
    if getattr(opt, "recon", False) is True:
        add_loss = add_loss + output["recon_loss"] * opt.recon_loss_weight
    return add_loss


def train(opt, data, mask, device, k):
    model = define_net(opt, k)
    optimizer = define_optimizer(opt, model)
    scheduler = define_scheduler(opt, optimizer)

    print(model)
    print("Number of Trainable Parameters: %d" % count_parameters(model))
    print("Activation Type:", opt.act_type)
    print("Optimizer Type:", opt.optimizer_type)
    print("Regularization Type:", opt.reg_type)

    use_patch, roi_dir = ("_patch_", "all_st_patches_512") if opt.use_patch is True else ("_", "all_st")

    custom_data_loader = PathgraphomicDatasetLoader(opt, data, mask, split="train", mode=opt.mode)
    train_loader = torch.utils.data.DataLoader(
        dataset=custom_data_loader,
        batch_size=opt.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        worker_init_fn=seed_worker,
        generator=g,
    )

    metric_logger = {
        "train": {"loss": [], "pvalue": [], "cindex": [], "surv_acc": [], "grad_acc": []},
        "test": {"loss": [], "pvalue": [], "cindex": [], "surv_acc": [], "grad_acc": []},
    }

    num_epoch_no_improvement = 0
    patience_epoch = 10
    epochs_cnt = 0
    start_early_stop = 0
    best_eval_metric_value = -float("inf")

    # -------------------------
    # OGM-GE config from opt
    # -------------------------
    use_ogm_ge_flag = bool(getattr(opt, "use_ogm_ge", False))
    branch_keys = [s.strip() for s in str(getattr(opt, "ogm_ge_branches", "")).split(",") if s.strip()]
    ogm_alpha = float(getattr(opt, "ogm_ge_alpha", 0.5))
    ogm_beta = float(getattr(opt, "ogm_ge_beta", 0.5))
    ogm_eps = float(getattr(opt, "ogm_ge_eps", 1e-8))
    ogm_ratio_clip = float(getattr(opt, "ogm_ge_ratio_clip", 10.0))
    ogm_every = int(getattr(opt, "ogm_ge_every", 1))
    ogm_warmup_epochs = int(getattr(opt, "ogm_ge_warmup_epochs", 0))

    global_step = 0

    for epoch in tqdm(range(opt.epoch_count, opt.niter + opt.niter_decay + 1)):
        epochs_cnt += 1
        opt.epoch_count_training = epochs_cnt
        print("epoch:", epochs_cnt, "num_epoch_no_improvement:", num_epoch_no_improvement)

        model = model.to(device)
        model.train()

        loss_epoch = 0.0

        for batch_idx, (
            x_name,
            x_path,
            x_omic,
            x_rad,
            x_demo,
            x_radiomics,
            censor,
            survtime,
            x_masks,
            x_keep_masks,
        ) in enumerate(train_loader):

            censor = censor.to(device) if "surv" in opt.task else censor

            # masks to device
            x_masks["mask_path"] = x_masks["mask_path"].to(device)
            x_masks["mask_rad"] = x_masks["mask_rad"].to(device)
            x_masks["mask_omic"] = x_masks["mask_omic"].to(device)
            x_masks["mask_demo"] = x_masks["mask_demo"].to(device)

            # forward
            output, pred = model(
                x_path=x_path.to(device),
                x_omic=x_omic.to(device),
                x_rad=x_rad.to(device),
                x_demo=x_demo.to(device),
                x_radiomics=x_radiomics.to(device),
                x_masks=x_masks,
                x_keep_masks=x_keep_masks.to(device),
            )

            # loss
            loss_cox = CoxLoss(survtime, censor, pred, device) if opt.task == "surv" else 0
            loss_reg = define_reg(opt, model)
            loss = opt.lambda_cox * loss_cox + opt.lambda_reg * loss_reg + loss_add(opt, output)
            loss_epoch += float(loss.detach().item())

            # backward
            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # -------------------------
            # OGM-GE (MUST be between backward and step)
            # -------------------------
            if use_ogm_ge_flag and len(branch_keys) > 0:
                if (epochs_cnt - 1) >= ogm_warmup_epochs and (global_step % max(1, ogm_every) == 0):
                    _ = apply_ogm_ge(
                        model=model,
                        branch_keys=branch_keys,
                        alpha=ogm_alpha,
                        beta=ogm_beta,
                        eps=ogm_eps,
                        ratio_clip=ogm_ratio_clip,
                    )

            optimizer.step()
            global_step += 1

            if opt.verbose > 0 and opt.print_every > 0 and (
                batch_idx % opt.print_every == 0 or batch_idx + 1 == len(train_loader)
            ):
                print(
                    "Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Loss {:9.4f}".format(
                        epoch + 1,
                        opt.niter + opt.niter_decay,
                        batch_idx + 1,
                        len(train_loader),
                        float(loss.item()),
                    )
                )

        scheduler.step()

        # -------------------------
        # Validation / Early stopping
        # -------------------------
        if epochs_cnt > start_early_stop and getattr(opt, "measure", False):
            loss_train, cindex_train, pvalue_train, surv_acc_train, pred_train = test(
                opt, model, data, mask, "train", device
            )
            loss_val, cindex_val, pvalue_val, surv_acc_val, pred_val = test(opt, model, data, mask, "val", device)
            loss_test, cindex_test, pvalue_test, surv_acc_test, pred_test = test(
                opt, model, data, mask, "test", device
            )

            if opt.verbose > 0 and opt.task == "surv":
                print("[Train]\tLoss: {:.4f}, C-Index: {:.4f}".format(loss_train, cindex_train))
                print("[Val]\tLoss: {:.4f}, C-Index: {:.4f}".format(loss_val, cindex_val))
                print("[Test]\tLoss: {:.4f}, C-Index: {:.4f}\n".format(loss_test, cindex_test))

        if epochs_cnt > start_early_stop + 1:
            current_eval_metric = cindex_val

            if best_eval_metric_value <= current_eval_metric:
                best_eval_metric_value = current_eval_metric
                num_epoch_no_improvement = 0

                # save checkpoint
                if len(opt.gpu_ids) > 0 and torch.cuda.is_available() and isinstance(model, torch.nn.DataParallel):
                    model_state_dict = model.module.cpu().state_dict()
                else:
                    model_state_dict = model.cpu().state_dict()

                ckpt_path = os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, f"{opt.model_name}_{k}.pt")
                torch.save(
                    {
                        "split": k,
                        "opt": opt,
                        "epoch": epochs_cnt,
                        "data": [],
                        "model_state_dict": model_state_dict,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "metrics": [],
                    },
                    ckpt_path,
                )

                # your existing pred_test dump
                pickle.dump(
                    pred_test,
                    open(
                        os.path.join(
                            opt.checkpoints_dir,
                            opt.exp_name,
                            opt.model_name,
                            f"{opt.model_name}_{k}{use_patch}pred_test1.pkl",
                        ),
                        "wb",
                    ),
                )

            else:
                num_epoch_no_improvement += 1

            if num_epoch_no_improvement == patience_epoch:
                print("Early Stopping")
                print(epochs_cnt)
                break

    return model, optimizer, metric_logger


def test(opt, model, data, mask, split, device):
    name_list = []
    model.eval()

    custom_data_loader = PathgraphomicFastDatasetLoader(opt, data, mask, split, mode=opt.mode)
    test_loader = torch.utils.data.DataLoader(
        dataset=custom_data_loader,
        batch_size=opt.batch_test_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        worker_init_fn=seed_worker,
        generator=g,
    )

    risk_pred_all, censor_all, survtime_all = np.array([]), np.array([]), np.array([])
    loss_test = 0.0

    for batch_idx, (
        x_name,
        x_path,
        x_omic,
        x_rad,
        x_demo,
        x_radiomics,
        censor,
        survtime,
        x_masks,
        x_keep_masks,
    ) in enumerate(test_loader):

        censor = censor.to(device) if "surv" in opt.task else censor

        x_masks["mask_path"] = x_masks["mask_path"].to(device)
        x_masks["mask_rad"] = x_masks["mask_rad"].to(device)
        x_masks["mask_omic"] = x_masks["mask_omic"].to(device)
        x_masks["mask_demo"] = x_masks["mask_demo"].to(device)

        output, pred = model(
            x_path=x_path.to(device),
            x_omic=x_omic.to(device),
            x_rad=x_rad.to(device),
            x_demo=x_demo.to(device),
            x_radiomics=x_radiomics.to(device),
            x_masks=x_masks,
            x_keep_masks=x_keep_masks.to(device),
        )

        loss_cox = CoxLoss(survtime, censor, pred, device)
        loss_reg = define_reg(opt, model)
        loss = opt.lambda_cox * loss_cox + opt.lambda_reg * loss_reg + loss_add(opt, output)
        loss_test += float(loss.detach().item())

        if opt.task == "surv":
            risk_pred_all = np.concatenate((risk_pred_all, pred.detach().cpu().numpy().reshape(-1)))
            censor_all = np.concatenate((censor_all, censor.detach().cpu().numpy().reshape(-1)))
            survtime_all = np.concatenate((survtime_all, survtime.detach().cpu().numpy().reshape(-1)))

        name_list = name_list + list(x_name)

    loss_test /= len(test_loader.dataset)
    cindex_test = CIndex_lifeline(risk_pred_all, censor_all, survtime_all)
    pvalue_test = cox_log_rank(risk_pred_all, censor_all, survtime_all)
    surv_acc_test = accuracy_cox(risk_pred_all, censor_all)

    pred_test = [risk_pred_all, survtime_all, censor_all, name_list]
    return loss_test, cindex_test, pvalue_test, surv_acc_test, pred_test


# ============================================================
# Missing-modality tests (kept as-is, only cleaned lightly)
# ============================================================
def test_missingModa(opt, model, data, mask, split, device):
    model.eval()
    custom_data_loader = PathgraphomicFastDatasetLoader(opt, data, mask, split, mode=opt.mode)
    test_loader = torch.utils.data.DataLoader(dataset=custom_data_loader, batch_size=1, shuffle=False, drop_last=False)
    pred_rad_demo = test_two_view_fun(test_loader, model, opt, device, "rad_demo")
    pred_path_missing_demo = test_three_view_fun(test_loader, model, opt, device, "path_missing")
    return pred_rad_demo, pred_path_missing_demo


def test_two_view_fun(test_loader, model, opt, device, view):
    risk_pred_all, censor_all, survtime_all = np.array([]), np.array([]), np.array([])
    name_list = []

    for batch_idx, (
        x_name, x_path, x_omic, x_rad, x_demo, x_radiomics, censor, survtime, x_masks, x_keep_masks
    ) in enumerate(test_loader):

        if view == "path_demo":
            x_omic = torch.zeros_like(x_omic)
            x_rad = torch.zeros_like(x_rad)
            x_radiomics = torch.zeros_like(x_radiomics)
            x_keep_masks = torch.ones_like(x_keep_masks)
            x_keep_masks[:, 3] = 0
            x_keep_masks[:, 1] = 0

        elif view == "omic_demo":
            x_path = torch.zeros_like(x_path)
            x_rad = torch.zeros_like(x_rad)
            x_radiomics = torch.zeros_like(x_radiomics)
            x_keep_masks = torch.ones_like(x_keep_masks)
            x_keep_masks[:, 1] = 0
            x_keep_masks[:, 0] = 0

        elif view == "rad_demo":
            x_path = torch.zeros_like(x_path)
            x_omic = torch.zeros_like(x_omic)
            x_keep_masks = torch.ones_like(x_keep_masks)
            x_keep_masks[:, 3] = 0
            x_keep_masks[:, 0] = 0

        elif view == "path_omic":
            x_demo = torch.zeros_like(x_demo)
            x_rad = torch.zeros_like(x_rad)
            x_radiomics = torch.zeros_like(x_radiomics)
            x_keep_masks = torch.ones_like(x_keep_masks)
            x_keep_masks[:, 2] = 0
            x_keep_masks[:, 1] = 0

        elif view == "path_rad":
            x_omic = torch.zeros_like(x_omic)
            x_demo = torch.zeros_like(x_demo)
            x_keep_masks = torch.ones_like(x_keep_masks)
            x_keep_masks[:, 3] = 0
            x_keep_masks[:, 2] = 0

        elif view == "rad_omic":
            x_path = torch.zeros_like(x_path)
            x_demo = torch.zeros_like(x_demo)
            x_keep_masks = torch.ones_like(x_keep_masks)
            x_keep_masks[:, 0] = 0
            x_keep_masks[:, 2] = 0

        censor = censor.to(device) if "surv" in opt.task else censor

        x_masks["mask_path"] = x_masks["mask_path"].to(device)
        x_masks["mask_rad"] = x_masks["mask_rad"].to(device)
        x_masks["mask_omic"] = x_masks["mask_omic"].to(device)
        x_masks["mask_demo"] = x_masks["mask_demo"].to(device)

        output, pred = model(
            x_path=x_path.to(device),
            x_omic=x_omic.to(device),
            x_rad=x_rad.to(device),
            x_demo=x_demo.to(device),
            x_radiomics=x_radiomics.to(device),
            x_masks=x_masks,
            x_keep_masks=x_keep_masks.to(device),
        )

        if opt.task == "surv":
            risk_pred_all = np.concatenate((risk_pred_all, pred.detach().cpu().numpy().reshape(-1)))
            censor_all = np.concatenate((censor_all, censor.detach().cpu().numpy().reshape(-1)))
            survtime_all = np.concatenate((survtime_all, survtime.detach().cpu().numpy().reshape(-1)))

        name_list.append(x_name)

    pred_test = [risk_pred_all, survtime_all, censor_all, name_list]
    return pred_test


def test_three_view_fun(test_loader, model, opt, device, view):
    risk_pred_all, censor_all, survtime_all = np.array([]), np.array([]), np.array([])
    name_list = []

    for batch_idx, (
        x_name, x_path, x_omic, x_rad, x_demo, x_radiomics, censor, survtime, x_masks, x_keep_masks
    ) in enumerate(test_loader):

        if view == "path_missing":
            x_path = torch.zeros_like(x_path)
            x_keep_masks = torch.ones_like(x_keep_masks)
            x_keep_masks[:, 0] = 0

        elif view == "demo_missing":
            x_demo = torch.zeros_like(x_demo)
            x_keep_masks = torch.ones_like(x_keep_masks)
            x_keep_masks[:, 2] = 0

        elif view == "omic_missing":
            x_omic = torch.zeros_like(x_omic)
            x_keep_masks = torch.ones_like(x_keep_masks)
            x_keep_masks[:, 3] = 0

        elif view == "rado_missing":
            x_rad = torch.zeros_like(x_rad)
            x_radiomics = torch.zeros_like(x_radiomics)
            x_keep_masks = torch.ones_like(x_keep_masks)
            x_keep_masks[:, 1] = 0

        censor = censor.to(device) if "surv" in opt.task else censor

        x_masks["mask_path"] = x_masks["mask_path"].to(device)
        x_masks["mask_rad"] = x_masks["mask_rad"].to(device)
        x_masks["mask_omic"] = x_masks["mask_omic"].to(device)
        x_masks["mask_demo"] = x_masks["mask_demo"].to(device)

        output, pred = model(
            x_path=x_path.to(device),
            x_omic=x_omic.to(device),
            x_rad=x_rad.to(device),
            x_demo=x_demo.to(device),
            x_radiomics=x_radiomics.to(device),
            x_masks=x_masks,
            x_keep_masks=x_keep_masks.to(device),
        )

        if opt.task == "surv":
            risk_pred_all = np.concatenate((risk_pred_all, pred.detach().cpu().numpy().reshape(-1)))
            censor_all = np.concatenate((censor_all, censor.detach().cpu().numpy().reshape(-1)))
            survtime_all = np.concatenate((survtime_all, survtime.detach().cpu().numpy().reshape(-1)))

        name_list.append(x_name)

    pred_test = [risk_pred_all, survtime_all, censor_all, name_list]
    return pred_test


# ============================================================
# test_complete_incomplete (cleaned checkpoint load)
# ============================================================
def test_complete_incomplete(opt, data_cv_path, data_cv_mask_path, fold_range=[i + 1 for i in range(15)]):
    opt.random_drop_views = False
    opt.use_patch = True

    device = torch.device(f"cuda:{opt.gpu_ids[0]}") if opt.gpu_ids else torch.device("cpu")
    print("Using device:", device)

    os.makedirs(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name), exist_ok=True)

    use_patch, roi_dir = ("_patch_", "all_st_patches_512") if opt.use_patch is True else ("_", "all_st")

    print("Loading %s" % data_cv_path)
    data_cv = pickle.load(open(data_cv_path, "rb"))
    data_cv_mask = pickle.load(open(data_cv_mask_path, "rb"))

    data_cv_splits = data_cv["cv_splits"]
    data_cv_mask_splits = data_cv_mask["cv_splits"]
    results = []

    Available_idx_file_path = os.path.join(opt.dataroot, "img_availability.csv")
    print("Available_idx_file_path:", Available_idx_file_path)
    print("data_cv_splits:", data_cv_path)
    print("data_mask_cv_splits:", data_cv_mask_path)

    data_cv_splits, mask_cv_splits = complete_incomplete_data_selection(
        opt,
        data_cv_splits=data_cv_splits,
        data_cv_mask=data_cv_mask_splits,
        available_idx_file_path=Available_idx_file_path,
    )

    for k, data in data_cv_splits.items():
        if k not in fold_range:
            continue

        print("*******************************************")
        print("************** SPLIT (%d/%d) **************" % (k, len(data_cv_splits.items())))
        print("*******************************************")

        if opt.use_embedding is True:
            data["test"]["x_omic"] = np.array(data["test"]["x_omic"]).squeeze(axis=1)
            data["test"]["x_rad"] = np.array(data["test"]["x_rad"]).squeeze(axis=1)
            data["test"]["x_demo"] = np.array(data["test"]["x_demo"]).squeeze(axis=1)

        print("# Testing DATA:", len(data["test"]["x_path"]))
        print("# Testing patient:", set(data["test"]["x_patname"]))

        ckpt_path = os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, f"{opt.model_name}_{k}.pt")

        with torch.serialization.safe_globals([argparse.Namespace]):
            model_ckpt = load_checkpoint_trusted(ckpt_path, device)

        model_state_dict = model_ckpt["model_state_dict"]
        opt.recon = False

        if hasattr(model_state_dict, "_metadata"):
            del model_state_dict._metadata

        model = define_net(opt, None)
        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        model_state_dict = strip_module_prefix(model_state_dict)

        print("Loading the model from %s" % ckpt_path)
        model.load_state_dict(model_state_dict, strict=True)

        loss_test, cindex_test, pvalue_test, surv_acc_test, pred_test = test(
            opt, model, data, mask_cv_splits[k], "test", device
        )
        pred_rad_demo, pred_test_path_miss = test_missingModa(opt, model, data, mask_cv_splits[k], "test", device)

        pickle.dump(
            pred_rad_demo,
            open(
                os.path.join(
                    opt.checkpoints_dir, opt.exp_name, opt.model_name, f"{opt.model_name}_{k}{use_patch}pred_test_rad_demo.pkl"
                ),
                "wb",
            ),
        )
        pickle.dump(
            pred_test_path_miss,
            open(
                os.path.join(
                    opt.checkpoints_dir, opt.exp_name, opt.model_name, f"{opt.model_name}_{k}{use_patch}pred_test_path_miss.pkl"
                ),
                "wb",
            ),
        )

        if opt.task == "surv":
            print("[Final] Apply model to testing set: C-Index: %.10f, P-Value: %.10e" % (cindex_test, pvalue_test))
            logging.info("[Final] Apply model to testing set: C-Index: %.10f, P-Value: %.10e" % (cindex_test, pvalue_test))
            results.append(cindex_test)

        pickle.dump(
            pred_test,
            open(
                os.path.join(
                    opt.checkpoints_dir, opt.exp_name, opt.model_name, f"{opt.model_name}_{k}{use_patch}pred_test.pkl"
                ),
                "wb",
            ),
        )

    print("Split Results:", results)
    print("Average:", np.array(results).mean())
    pickle.dump(
        results,
        open(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, f"{opt.model_name}_results.pkl"), "wb"),
    )
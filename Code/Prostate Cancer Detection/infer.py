from pathlib import Path
import torch
import SimpleITK as sitk
import numpy as np

from .dataset import MODS, read_sitk, sitk_to_np, resample_to_ref, center_crop_or_pad
from .model import ThreeModalityClassifier
from .utils import zscore

@torch.no_grad()
def predict_case(sid: str, images_root: Path, ckpt_path: Path, target_shape=(128,192,192)) -> float:
    patient = sid.split("_")[0]
    pdir = images_root / patient

    # ref t2
    ref_p = pdir / f"{sid}_{MODS[0]}.mha"
    ref_img = read_sitk(ref_p)
    ref_np = sitk_to_np(ref_img, np.float32)
    ref_np = zscore(ref_np)
    ref_np = center_crop_or_pad(ref_np, target_shape)

    vols = [torch.from_numpy(ref_np.astype(np.float32))]  # (1,Z,Y,X)

    for mod in MODS[1:]:
        p = pdir / f"{sid}_{mod}.mha"
        img = read_sitk(p)
        if (img.GetSize() != ref_img.GetSize()) or (img.GetSpacing() != ref_img.GetSpacing()) \
           or (img.GetOrigin() != ref_img.GetOrigin()) or (img.GetDirection() != ref_img.GetDirection()):
            img = resample_to_ref(img, ref_img)
        arr = zscore(sitk_to_np(img, np.float32))
        arr = center_crop_or_pad(arr, target_shape)
        vols.append(torch.from_numpy(arr.astype(np.float32)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ThreeModalityClassifier().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    x0 = vols[0][None].to(device)
    x1 = vols[1][None].to(device)
    x2 = vols[2][None].to(device)

    logit = model((x0,x1,x2))
    prob = torch.sigmoid(logit).item()
    return float(prob)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--sid", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tZ", type=int, default=128)
    p.add_argument("--tY", type=int, default=192)
    p.add_argument("--tX", type=int, default=192)
    args = p.parse_args()
    pr = predict_case(args.sid, Path(args.images), Path(args.ckpt), target_shape=(args.tZ,args.tY,args.tX))
    print(f"{args.sid} -> P(csPCa)= {pr:.4f}")

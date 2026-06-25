from pathlib import Path
import json
import torch
from torch.utils.data import DataLoader

from .dataset import build_case_list_classification, Case3DClassificationDataset
from .model import ThreeModalityClassifier
from .utils import safe_auc, safe_ap

@torch.no_grad()
def eval_fold(fold: int, images_root: Path, labels_csv: Path, splits_json: Path, ckpt_path: Path, out_path: Path, target_shape=(128,192,192)):
    cases = build_case_list_classification(images_root, labels_csv=labels_csv, strict=True, verbose=False)
    sid_to_case = {c["sid"]: c for c in cases}

    splits = json.load(open(splits_json))
    fold_split = splits[fold] if isinstance(splits, list) else (splits[str(fold)] if str(fold) in splits else splits[fold])
    val_ids = fold_split["val"]
    val_cases = [sid_to_case[s] for s in val_ids if s in sid_to_case]

    ds = Case3DClassificationDataset(val_cases, target_shape=target_shape, normalize=True, align_to_ref=True)
    ld = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ThreeModalityClassifier().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    y_true, y_score, rows = [], [], []

    for (x0, x1, x2), y, sid in ld:
        x0 = x0.to(device); x1 = x1.to(device); x2 = x2.to(device)
        logit = model((x0, x1, x2))
        prob = torch.sigmoid(logit).item()
        yt = int(y.item())
        s = str(sid[0])
        y_true.append(yt); y_score.append(prob)
        rows.append({"sid": s, "y": yt, "p": float(prob)})

    out = {"fold": fold, "auc": safe_auc(y_true, y_score), "ap": safe_ap(y_true, y_score), "cases": rows}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=2)
    print("Saved:", out_path)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--labels_csv", required=True)
    p.add_argument("--splits", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tZ", type=int, default=128)
    p.add_argument("--tY", type=int, default=192)
    p.add_argument("--tX", type=int, default=192)
    args = p.parse_args()
    eval_fold(args.fold, Path(args.images), Path(args.labels_csv), Path(args.splits), Path(args.ckpt), Path(args.out),
              target_shape=(args.tZ,args.tY,args.tX))

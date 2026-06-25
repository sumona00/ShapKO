# dataset.py
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Tuple, Union, Optional

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset

from .utils import zscore

MODS = ["t2w", "adc", "hbv"]


# -----------------------------
# I/O helpers
# -----------------------------
def read_sitk(path: Union[str, Path]) -> sitk.Image:
    return sitk.ReadImage(str(path))


def sitk_to_np(img: sitk.Image, dtype=np.float32) -> np.ndarray:
    return sitk.GetArrayFromImage(img).astype(dtype)  # (Z,Y,X)


def resample_to_ref(moving: sitk.Image, ref: sitk.Image) -> sitk.Image:
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref)
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)
    resampler.SetInterpolator(sitk.sitkLinear)
    return resampler.Execute(moving)


# -----------------------------
# Label parsing
# -----------------------------
def parse_yesno(v) -> int:
    s = str(v).strip().upper()
    if s in {"YES", "1", "TRUE"}:
        return 1
    if s in {"NO", "0", "FALSE"}:
        return 0
    raise ValueError(f"Unknown label value: {v}")


def _norm_col(c: str) -> str:
    return (
        str(c)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("-", "")
        .replace(".", "")
        .replace("__", "_")
    )


# In dataset.py -> load_labels_csv
def load_labels_csv(labels_csv: Union[str, Path]) -> Dict[str, int]:
    df = pd.read_csv(labels_csv)
    
    # NEW: Construct the SID to match the fold JSON format
    # patient_id_study_id
    df['sid_constructed'] = df['patient_id'].astype(str) + "_" + df['study_id'].astype(str)
    
    y_col = "case_csPCa" # Based on your provided CSV header
    
    out: Dict[str, int] = {}
    for _, r in df.iterrows():
        sid = str(r['sid_constructed']).strip()
        out[sid] = parse_yesno(r[y_col])
    return out


def load_labels_json(labels_json: Union[str, Path]) -> Dict[str, int]:
    import json
    d = json.load(open(labels_json, "r"))
    out: Dict[str, int] = {}
    for k, v in d.items():
        out[str(k).strip()] = parse_yesno(v)
    return out


# -----------------------------
# Discovery from filesystem (CRITICAL FIX)
# -----------------------------
def _extract_sid_from_filename(fname: str, mod: str, suffix: str) -> Optional[Tuple[str, str, str]]:
    """
    Expect: <patient_id>_<study_id>_<mod><suffix>
    Return: (sid, patient_id, study_id) where sid = patient_id_study_id
    """
    end = f"_{mod}{suffix}"
    if not fname.endswith(end):
        return None

    core = fname[: -len(end)]  # "<patient_id>_<study_id>"
    if "_" not in core:
        return None

    patient_id, study_id = core.split("_", 1)
    sid = f"{patient_id}_{study_id}"
    return sid, patient_id, study_id


def discover_cases_from_images(
    images_root: Union[str, Path],
    mods: List[str] = MODS,
    image_suffix: str = ".mha",
    verbose: bool = True,
) -> List[Dict]:
    """
    Filesystem truth:
      images_root/<patient_id>/<patient_id>_<study_id>_<mod>.mha

    We construct:
      sid = <patient_id>_<study_id>   (MATCHES splits_10fold.json)
    """
    images_root = Path(images_root)
    anchor_mod = mods[0]

    cases_by_sid: Dict[str, Dict] = {}

    patient_dirs = [p for p in images_root.iterdir() if p.is_dir()]
    for pdir in sorted(patient_dirs):
        patient_id = pdir.name

        # anchor files to enumerate studies in this patient folder
        anchor_glob = f"{patient_id}_*_{anchor_mod}{image_suffix}"
        anchors = sorted(pdir.glob(anchor_glob))

        for a in anchors:
            parsed = _extract_sid_from_filename(a.name, anchor_mod, image_suffix)
            if parsed is None:
                continue
            sid, pid_from_name, study_id = parsed

            # sanity: enforce folder name == pid in filename
            if pid_from_name != patient_id:
                # skip weirdly placed files
                continue

            scans = {m: (pdir / f"{sid}_{m}{image_suffix}") for m in mods}
            exists = {m: scans[m].exists() for m in mods}

            cases_by_sid[sid] = {
                "sid": sid,                 # patient_id_study_id (MATCH SPLITS)
                "patient_id": patient_id,   # folder name
                "study_id": study_id,       # extracted
                "scans": scans,
                "exists": exists,
            }

    discovered = list(cases_by_sid.values())

    if verbose:
        print(f"[discover_cases_from_images] Patient dirs: {len(patient_dirs)} | discovered sids: {len(discovered)}")
        if len(discovered) > 0:
            ex = discovered[0]
            print("[discover_cases_from_images] EX sid:", ex["sid"], "patient_id:", ex["patient_id"], "study_id:", ex["study_id"])
            for m in mods:
                print(" ", m, "->", ex["scans"][m], "exists=", ex["exists"][m])

    return discovered


def build_case_list_classification(
    images_root: Union[str, Path],
    labels_csv: Optional[Union[str, Path]] = None,
    labels_json: Optional[Union[str, Path]] = None,
    mods: List[str] = MODS,
    image_suffix: str = ".mha",
    strict: bool = True,
    verbose: bool = True,
) -> List[Dict]:
    """
    Builds list of cases where:
      case["sid"] == patient_id_study_id  (MUST match splits_10fold.json)
    """
    images_root = Path(images_root)

    if (labels_csv is None) == (labels_json is None):
        raise ValueError("Provide exactly one of labels_csv or labels_json")

    labels = load_labels_csv(labels_csv) if labels_csv is not None else load_labels_json(labels_json)

    discovered = discover_cases_from_images(images_root, mods=mods, image_suffix=image_suffix, verbose=verbose)

    cases: List[Dict] = []
    skipped_missing = 0
    skipped_nolabel = 0

    for c in discovered:
        sid = c["sid"]
        if sid not in labels:
            skipped_nolabel += 1
            continue
        if strict and (not all(c["exists"].values())):
            skipped_missing += 1
            continue
        c2 = dict(c)
        c2["y"] = int(labels[sid])
        cases.append(c2)

    if verbose:
        print(f"[build_case_list_classification] Loaded labels for {len(labels)} sids")
        print(f"[build_case_list_classification] Discovered {len(discovered)} cases on disk")
        print(
            f"[build_case_list_classification] Built {len(cases)} labeled cases "
            f"(skipped {skipped_missing} missing-modality, {skipped_nolabel} no-label)"
        )
        if len(cases) > 0:
            ex = cases[0]
            print("[build_case_list_classification] EX labeled sid:", ex["sid"], "y=", ex["y"])

    return cases


# -----------------------------
# Preprocess
# -----------------------------
def center_crop_or_pad(arr: np.ndarray, target_zyx: Tuple[int, int, int]) -> np.ndarray:
    """
    arr: (Z,Y,X) or (1,Z,Y,X)
    return: (1,tZ,tY,tX)
    """
    if arr.ndim == 3:
        arr = arr[None]
    C, Z, Y, X = arr.shape
    tZ, tY, tX = target_zyx
    out = np.zeros((C, tZ, tY, tX), dtype=arr.dtype)

    def _calc(in_len, out_len):
        if in_len >= out_len:
            in0 = (in_len - out_len) // 2
            out0 = 0
            sz = out_len
        else:
            in0 = 0
            out0 = (out_len - in_len) // 2
            sz = in_len
        return in0, out0, sz

    z_in, z_out, z_sz = _calc(Z, tZ)
    y_in, y_out, y_sz = _calc(Y, tY)
    x_in, x_out, x_sz = _calc(X, tX)

    out[:, z_out:z_out + z_sz, y_out:y_out + y_sz, x_out:x_out + x_sz] = \
        arr[:, z_in:z_in + z_sz, y_in:y_in + y_sz, x_in:x_in + x_sz]
    return out


# -----------------------------
# Dataset
# -----------------------------
class Case3DClassificationDataset(Dataset):
    """
    Output:
      (x_t2w, x_adc, x_hbv), y, sid
    """
    def __init__(
        self,
        cases: List[Dict],
        mods: List[str] = MODS,
        normalize: bool = True,
        align_to_ref: bool = True,
        target_shape: Tuple[int, int, int] = (128, 192, 192),
        return_sid: bool = True,
        allow_missing: bool = False,
    ):
        self.cases = cases
        self.mods = list(mods)
        self.normalize = bool(normalize)
        self.align_to_ref = bool(align_to_ref)
        self.target_shape = tuple(int(x) for x in target_shape)
        self.return_sid = bool(return_sid)
        self.allow_missing = bool(allow_missing)

        if len(self.cases) == 0:
            raise ValueError("Case3DClassificationDataset received 0 cases.")

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx: int):
        case = self.cases[idx]
        sid = case["sid"]
        y = int(case["y"])

        # ref = first modality (t2w)
        ref_path = Path(case["scans"][self.mods[0]])
        if not ref_path.exists():
            if not self.allow_missing:
                raise FileNotFoundError(f"Missing reference modality: {ref_path}")
            ref_np = np.zeros((1, *self.target_shape), np.float32)
            ref_img = None
        else:
            ref_img = read_sitk(ref_path)
            ref_np = sitk_to_np(ref_img, dtype=np.float32)
            if self.normalize:
                ref_np = zscore(ref_np)
            ref_np = center_crop_or_pad(ref_np, self.target_shape)

        vols = [torch.from_numpy(ref_np.astype(np.float32))]

        # remaining modalities
        for mod in self.mods[1:]:
            p = Path(case["scans"][mod])
            if not p.exists():
                if not self.allow_missing:
                    raise FileNotFoundError(f"Missing modality {mod}: {p}")
                v = np.zeros((1, *self.target_shape), np.float32)
                vols.append(torch.from_numpy(v))
                continue

            img = read_sitk(p)
            if self.align_to_ref and ref_img is not None:
                if (img.GetSize() != ref_img.GetSize()) or (img.GetSpacing() != ref_img.GetSpacing()) \
                   or (img.GetOrigin() != ref_img.GetOrigin()) or (img.GetDirection() != ref_img.GetDirection()):
                    img = resample_to_ref(img, ref_img)

            arr = sitk_to_np(img, dtype=np.float32)
            if self.normalize:
                arr = zscore(arr)
            arr = center_crop_or_pad(arr, self.target_shape)
            vols.append(torch.from_numpy(arr.astype(np.float32)))

        y_t = torch.tensor([float(y)], dtype=torch.float32)

        if self.return_sid:
            return tuple(vols), y_t, sid
        return tuple(vols), y_t

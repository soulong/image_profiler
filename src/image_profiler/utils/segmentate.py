"""Cellpose-SAM segmentation for microscopy images.

Cellpose-SAM uses the first three channels of the input image.
Supports:
    1. single channel → C1 = image, C2 = 0
    2. two channels  → C1 = chA, C2 = chB
    3. two merged channels → C1 = merge(chan1), C2 = merge(chan2)
"""

from __future__ import annotations

import gc
import re
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
import torch
from cellpose import models, io as cp_io
from imageio.v3 import imread
from skimage.morphology import closing
from skimage.transform import rescale
from tqdm import tqdm


def _merge_channels(
    paths: List[Path],
    method: str = "mean",
    resize_factor: float = 1.0,
) -> np.ndarray:
    """Read, optionally resize and merge a list of images.

    Parameters
    ----------
    paths : list of Path
        List of image paths.
    method : str
        Merge method: "mean", "max", or "min".
    resize_factor : float
        Resize factor for images.

    Returns
    -------
    np.ndarray
        Merged image.
    """
    imgs = [imread(p) for p in paths]
    stacked = np.stack(imgs, axis=0)
    
    if stacked.ndim == 4:
        stacked = np.mean(stacked, axis=3, keepdims=False)

    if method == "mean":
        merged = np.mean(stacked, axis=0)
    elif method == "max":
        merged = np.max(stacked, axis=0)
    elif method == "min":
        merged = np.min(stacked, axis=0)
    else:
        raise ValueError(f"Unsupported merge method: {method}")

    if resize_factor != 1.0:
        merged = rescale(
            merged,
            resize_factor,
            anti_aliasing=True,
            preserve_range=True,
        ).astype(stacked.dtype)

    return merged


def _build_cellpose_image(
    df: pd.DataFrame,
    row_idx: int,
    chan1: List[str],
    chan2: Optional[List[str]],
    merge1: str,
    merge2: str,
    resize_factor: float,
) -> np.ndarray:
    """Build a (2, H, W) image for Cellpose-SAM.

    Parameters
    ----------
    df : pd.DataFrame
        Metadata DataFrame.
    row_idx : int
        Row index.
    chan1 : list of str
        First channel group.
    chan2 : list of str, optional
        Second channel group.
    merge1 : str
        Merge method for chan1.
    merge2 : str
        Merge method for chan2.
    resize_factor : float
        Resize factor.

    Returns
    -------
    np.ndarray
        Image with shape (2, H, W).
    """
    row = df.iloc[row_idx]
    dir_path = Path(row["directory"])

    ch1_paths = [
        dir_path / row[ch] 
        for ch in chan1 
        if ch in row and pd.notna(row[ch])
    ]
    if not ch1_paths:
        raise ValueError(f"Missing images for channel group 1: {chan1}")
    c1 = _merge_channels(ch1_paths, merge1, resize_factor)

    if chan2:
        ch2_paths = [
            dir_path / row[ch] 
            for ch in chan2 
            if ch in row and pd.notna(row[ch])
        ]
        if not ch2_paths:
            raise ValueError(f"Missing images for channel group 2: {chan2}")
        c2 = _merge_channels(ch2_paths, merge2, resize_factor)
    else:
        c2 = np.zeros_like(c1)

    return np.stack([c1, c2], axis=0)


def cellpose_segment_measurement(
    dataset: Dict,
    chan1: Union[List[str], str],
    chan2: Union[List[str], str],
    merge1: str = 'mean',
    merge2: str = 'mean',   
    model_name: str = "cpsam",
    diameter: Optional[float] = None,
    normalize: Optional[Dict] = {"percentile": [0.1, 99.9]},
    resize_factor: float = 1.0,
    mask_name: str = 'cell',
    overwrite_mask: bool = False,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    gpu_batch_size: int = 16,
) -> Dict:
    """Run Cellpose-SAM on a single measurement folder.

    Parameters
    ----------
    dataset : dict
        Dataset dictionary with 'metadata' and 'intensity_colnames'.
    chan1 : list of str
        First channel group.
    chan2 : list of str, optional
        Second channel group.
    merge1 : str
        Merge method for chan1.
    merge2 : str
        Merge method for chan2.
    model_name : str
        Cellpose model name or path.
    diameter : float, optional
        Object diameter in pixels.
    normalize : dict, optional
        Normalization parameters.
    resize_factor : float
        Resize factor.
    mask_name : str
        Name for mask files.
    overwrite_mask : bool
        Overwrite existing masks.
    flow_threshold : float
        Flow error threshold.
    cellprob_threshold : float
        Cell probability threshold.
    gpu_batch_size : int
        GPU batch size.

    Returns
    -------
    dict
        Summary with keys: "success", "processed", "skipped", "failed", 
        "masks_saved", "errors".
    """
    summary = {
        "success": False,
        "processed": 0,
        "skipped": 0,
        "failed": 0,
        "masks_saved": 0,
        "errors": []
    }
    
    df = dataset.get("metadata", pd.DataFrame())
    intensity_cols = dataset.get("intensity_colnames", [])

    if isinstance(chan1, str): chan1 = [chan1]
    if isinstance(chan2, str): chan2 = [chan2]
    
    missing = [ch for ch in (chan1 + (chan2 or [])) if ch not in intensity_cols]
    if missing:
        summary["errors"].append(f"Missing channels in dataset: {missing}")
        print(f"[Cellpose] ERROR: Missing channels in dataset: {missing}")
        return summary
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"[Cellpose] Starting segmentation")
    print(f"[Cellpose] Model: {model_name}")
    print(f"[Cellpose] Device: {device}")
    print(f"[Cellpose] Channels: chan1={chan1}, chan2={chan2}")
    print(f"[Cellpose] Mask name: {mask_name}")
    print(f"[Cellpose] Total images: {len(df)}")
    print(f"[Cellpose] Overwrite existing: {overwrite_mask}")

    try:
        model = models.CellposeModel(device=device, pretrained_model=model_name)
    except Exception as e:
        summary["errors"].append(f"Failed to load model: {e}")
        print(f"[Cellpose] ERROR: Failed to load model: {e}")
        return summary

    diameter_val = None if diameter is None or diameter <= 0 else int(diameter * resize_factor)

    for idx in tqdm(range(len(df)), desc="Cellpose", unit="img"):
        row = df.iloc[idx]

        stem_ch = chan1[0]
        src_path = Path(row["directory"]) / row[stem_ch]
        
        if not src_path.exists():
            summary["skipped"] += 1
            summary["errors"].append(f"Source file not found: {src_path.name}")
            continue

        save_stem = src_path.parent / f"{src_path.stem}_cp_masks"
        
        # cellpose auto add '_cp_mask_'
        mask_path = save_stem.with_name(f"{save_stem.name}_{mask_name}.png")
        # print(mask_path)
        if mask_path.exists() and not overwrite_mask:
            summary["skipped"] += 1
            continue

        try:
            img = _build_cellpose_image(
                df, idx,
                chan1, chan2, merge1, merge2,
                resize_factor,
            )

            masks, flows, styles = model.eval(
                img,
                batch_size=gpu_batch_size,
                normalize=normalize,
                diameter=diameter_val,
                flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold,
            )

            if resize_factor != 1.0:
                masks = rescale(masks, 1.0 / resize_factor, order=0).astype(np.uint16)

            n_objects = len(np.unique(masks)) - 1
            if n_objects <= 0:
                summary["processed"] += 1
                summary["errors"].append(f"No objects detected in {src_path.name}")
                continue

            cp_io.save_masks(
                img[0],
                closing(masks),
                flows,
                file_names=str(save_stem),
                suffix=f"_{mask_name}",
            )
            
            summary["processed"] += 1
            summary["masks_saved"] += 1

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            summary["failed"] += 1
            summary["errors"].append(f"GPU OOM on {src_path.name}")
            
        except Exception as e:
            summary["failed"] += 1
            summary["errors"].append(f"Error processing {src_path.name}: {e}")

        if idx % 200 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    summary["success"] = summary["masks_saved"] > 0
    
    print(f"[Cellpose] Summary:")
    print(f"  - Total images: {len(df)}")
    print(f"  - Processed: {summary['processed']}")
    print(f"  - Skipped (existing): {summary['skipped']}")
    print(f"  - Failed: {summary['failed']}")
    print(f"  - Masks saved: {summary['masks_saved']}")
    if summary["errors"]:
        print(f"  - Errors: {len(summary['errors'])}")
    
    return summary

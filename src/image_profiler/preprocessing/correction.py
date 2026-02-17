"""BaSiC Shading Correction Pipeline.

Supports:
  - fit: train BaSiC model per channel (saved as .pkl + profiles)
  - transform: apply previously trained models
"""

from __future__ import annotations

import pickle
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from shutil import copy
from tifffile import imread, imwrite
from tqdm import tqdm

from image_profiler.preprocessing.basic.basic import BaSiC


def _basic_fit(
    image_paths: List[Path],
    n_image: int = 50,
    enable_darkfield: bool = True,
    working_size: int = 128,
) -> Optional[BaSiC]:
    """Fit BaSiC model on a set of images.

    Parameters
    ----------
    image_paths : list of Path
        List of image paths for fitting.
    n_image : int
        Number of images to use for fitting.
    enable_darkfield : bool
        Enable darkfield estimation.
    working_size : int
        Working size for BaSiC model.

    Returns
    -------
    BaSiC or None
        Fitted BaSiC model, or None if fitting failed.
    """
    if len(image_paths) > n_image:
        image_paths = random.sample(image_paths, k=n_image)

    imgs = np.stack([imread(p) for p in image_paths])
    basic = BaSiC(
        get_darkfield=enable_darkfield,
        smoothness_flatfield=1,
        smoothness_darkfield=1,
        working_size=working_size,
        max_workers=8,
    )
    basic.fit(imgs)
    
    return basic


def _basic_transform(
    image_paths: List[Path],
    model: BaSiC,
    target_dir: Optional[Path] = None,
) -> tuple:
    """Apply BaSiC model to correct images.

    Parameters
    ----------
    image_paths : list of Path
        List of image paths to correct.
    model : BaSiC
        Fitted BaSiC model.
    target_dir : Path, optional
        Output directory for corrected images.

    Returns
    -------
    tuple
        (success: bool, n_processed: int, errors: list)
    """
    errors = []
    n_processed = 0
    
    try:
        imgs = np.stack([imread(p) for p in image_paths])
        dtype_in = imgs.dtype
        corrected = model.transform(imgs)

        if dtype_in == np.uint16:
            corrected = np.clip(corrected, 0, 65535)
        elif dtype_in == np.uint8:
            corrected = np.clip(corrected, 0, 255)
        corrected = corrected.astype(dtype_in)

        for src_path, corr_img in zip(image_paths, corrected):
            try:
                if target_dir is not None:
                    rel = src_path.relative_to(src_path.parents[2])
                    dst_path = target_dir / rel
                    dst_path.parent.mkdir(parents=True, exist_ok=True)
                    imwrite(dst_path, corr_img, compression="zlib")
                else:
                    imwrite(src_path, corr_img, compression="zlib")
                n_processed += 1
            except Exception as e:
                errors.append(f"Failed to write {src_path.name}: {e}")
        
        return True, n_processed, errors
    except Exception as e:
        errors.append(f"Transform failed: {e}")
        return False, 0, errors


def fit_basic_models(
    dataset: Dict,
    channels: List[str],
    n_image: int,
    working_size: int,
    enable_darkfield: bool = False,
    output_root: Optional[Path] = None,
) -> Dict:
    """Fit BaSiC models for specified channels.

    Parameters
    ----------
    dataset : dict
        Dataset dictionary with 'metadata' and 'intensity_colnames'.
    channels : list of str
        Channel names to fit models for.
    n_image : int
        Number of images to use for fitting.
    working_size : int
        Working size for BaSiC model.
    enable_darkfield : bool
        Enable darkfield estimation.
    output_root : Path, optional
        Output root directory for models.

    Returns
    -------
    dict
        Summary with keys: "success", "channels_processed", "channels_failed", "errors".
    """
    metadata = dataset["metadata"]
    intensity_cols = dataset["intensity_colnames"]
    images_dir = Path(metadata['directory'].iloc[0])
    measurement_dir = images_dir.parent

    summary = {
        "success": False,
        "channels_processed": [],
        "channels_failed": [],
        "errors": []
    }

    valid_channels = [c for c in channels if c in intensity_cols]
    if not valid_channels:
        summary["errors"].append("No valid channels found in dataset")
        print(f"[BaSiC Fit] ERROR: No valid channels found")
        return summary

    basic_dir = (
        (output_root / measurement_dir.name / "BaSiC_model") 
        if output_root 
        else (measurement_dir / "BaSiC_model")
    )
    basic_dir.mkdir(parents=True, exist_ok=True)

    print(f"[BaSiC Fit] Starting model fitting for {len(valid_channels)} channel(s)")
    print(f"[BaSiC Fit] Output directory: {basic_dir}")

    for chan in valid_channels:
        print(f"[BaSiC Fit] Processing channel: {chan}")
        
        paths = [images_dir / f for f in metadata[chan].dropna()]
        
        if len(paths) == 0:
            summary["channels_failed"].append(chan)
            summary["errors"].append(f"No images found for channel {chan}")
            print(f"[BaSiC Fit]   ERROR: No images found for channel {chan}")
            continue
        
        try:
            model = _basic_fit(paths, n_image, enable_darkfield, working_size)
            
            if model is None:
                summary["channels_failed"].append(chan)
                summary["errors"].append(f"Model fitting failed for channel {chan}")
                print(f"[BaSiC Fit]   ERROR: Model fitting failed")
                continue

            with open(basic_dir / f"model_{chan}.pkl", "wb") as f:
                pickle.dump(model, f)
            
            imwrite(
                basic_dir / f"model_{chan}_flatfield.tiff",
                model.flatfield.astype(np.float32),
                compression="zlib"
            )
            
            if enable_darkfield:
                imwrite(
                    basic_dir / f"model_{chan}_darkfield.tiff",
                    model.darkfield.astype(np.float32),
                    compression="zlib"
                )

            test_paths = random.sample(paths, min(3, len(paths)))
            imgs = np.stack([imread(p) for p in test_paths])
            corrected = model.transform(imgs)
            
            dtype_in = imgs.dtype
            if dtype_in == np.uint16:
                corrected = np.clip(corrected, 0, 65535)
            elif dtype_in == np.uint8:
                corrected = np.clip(corrected, 0, 255)
            corrected = corrected.astype(dtype_in)
            
            for i, p in enumerate(test_paths):
                copy(p, basic_dir / p.name)
                out_dtype = np.uint16 if p.suffix == ".tiff" else np.uint8
                imwrite(
                    basic_dir / f"{p.stem}_corrected.tiff",
                    corrected[i].astype(out_dtype),
                    compression="zlib"
                )
            
            summary["channels_processed"].append(chan)
            print(f"[BaSiC Fit]   SUCCESS: Model saved for channel {chan}")
            
        except Exception as e:
            summary["channels_failed"].append(chan)
            summary["errors"].append(f"Error fitting channel {chan}: {e}")
            print(f"[BaSiC Fit]   ERROR: {e}")

    summary["success"] = len(summary["channels_processed"]) > 0
    
    print(f"\n[BaSiC Fit] Summary:")
    print(f"  - Channels processed: {len(summary['channels_processed'])}")
    print(f"  - Channels failed: {len(summary['channels_failed'])}")
    if summary["channels_processed"]:
        print(f"  - Successful: {', '.join(summary['channels_processed'])}")
    if summary["channels_failed"]:
        print(f"  - Failed: {', '.join(summary['channels_failed'])}")
    
    return summary


def transform_basic_models(
    dataset: Dict,
    channels: List[str],
    output_root: Optional[Path] = None,
) -> Dict:
    """Apply fitted BaSiC models to correct images.

    Parameters
    ----------
    dataset : dict
        Dataset dictionary with 'metadata' and 'intensity_colnames'.
    channels : list of str
        Channel names to transform.
    output_root : Path, optional
        Output root directory for corrected images.

    Returns
    -------
    dict
        Summary with keys: "success", "channels_processed", "images_corrected", 
        "channels_skipped", "errors".
    """
    metadata = dataset["metadata"]
    intensity_cols = dataset["intensity_colnames"]
    valid_channels = [c for c in channels if c in intensity_cols]
    images_dir = Path(metadata['directory'].iloc[0])
    measurement_dir = images_dir.parent

    summary = {
        "success": False,
        "channels_processed": [],
        "images_corrected": 0,
        "channels_skipped": [],
        "errors": []
    }

    model_base = (
        (output_root / measurement_dir.name) 
        if output_root 
        else measurement_dir
    )

    print(f"[BaSiC Transform] Starting image correction for {len(valid_channels)} channel(s)")

    for chan in valid_channels:
        model_path = model_base / "BaSiC_model" / f"model_{chan}.pkl"
        
        if not model_path.exists():
            summary["channels_skipped"].append(chan)
            summary["errors"].append(f"Model not found for channel {chan}: {model_path}")
            print(f"[BaSiC Transform] SKIP: No model found for channel {chan}")
            continue

        print(f"[BaSiC Transform] Processing channel: {chan}")
        
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)

            paths = [images_dir / f for f in metadata[chan].dropna()]
            total_images = len(paths)
            channel_corrected = 0
            channel_errors = []

            for batch_start in tqdm(range(0, len(paths), 50), desc=f"Correct {chan}"):
                batch = paths[batch_start:batch_start + 50]
                success, n_proc, errs = _basic_transform(batch, model, output_root)
                channel_corrected += n_proc
                channel_errors.extend(errs)
            
            summary["channels_processed"].append(chan)
            summary["images_corrected"] += channel_corrected
            summary["errors"].extend(channel_errors)
            
            print(f"[BaSiC Transform]   Corrected {channel_corrected}/{total_images} images for channel {chan}")
            
        except Exception as e:
            summary["channels_skipped"].append(chan)
            summary["errors"].append(f"Error transforming channel {chan}: {e}")
            print(f"[BaSiC Transform]   ERROR: {e}")

    if output_root:
        unprocessed = set(intensity_cols) - set(valid_channels)
        target_img_dir = output_root / measurement_dir.name / "Images"
        target_img_dir.mkdir(parents=True, exist_ok=True)
        
        for ch in unprocessed:
            for fname in metadata[ch].dropna():
                src = images_dir / fname
                dst = target_img_dir / fname
                dst.parent.mkdir(parents=True, exist_ok=True)
                if src.exists():
                    copy(src, dst)

    summary["success"] = len(summary["channels_processed"]) > 0
    
    print(f"[BaSiC Transform] Summary:")
    print(f"  - Channels processed: {len(summary['channels_processed'])}")
    print(f"  - Images corrected: {summary['images_corrected']}")
    print(f"  - Channels skipped: {len(summary['channels_skipped'])}")
    if summary["channels_processed"]:
        print(f"  - Successful: {', '.join(summary['channels_processed'])}")
    if summary["channels_skipped"]:
        print(f"  - Skipped: {', '.join(summary['channels_skipped'])}")
    
    return summary

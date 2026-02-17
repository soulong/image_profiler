"""Z-stack projection utilities for microscopy images."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tifffile import imwrite
from tqdm import tqdm

try:
    import imageio.v3 as iio
except ImportError:
    import imageio as iio


def z_project_group(
    image_paths: List[Path],
    method: str = "max"
) -> tuple:
    """Project a group of Z-stack images.

    Parameters
    ----------
    image_paths : list of Path
        List of image paths for the Z-stack.
    method : str
        Projection method: "max", "mean", or "min".

    Returns
    -------
    tuple
        (projected_image, errors) where projected_image is np.ndarray or None
    """
    errors = []
    images = []
    
    for img_path in image_paths:
        if not img_path.exists():
            errors.append(f"File not found: {img_path}")
            continue
        try:
            img = iio.imread(img_path)
            images.append(img)
        except Exception as e:
            errors.append(f"Failed to read {img_path.name}: {e}")
    
    if len(images) == 0:
        return None, errors
    
    stacked = np.stack(images, axis=0)
    
    if method == "max":
        return np.max(stacked, axis=0), errors
    elif method == "mean":
        return np.mean(stacked, axis=0).astype(images[0].dtype), errors
    elif method == "min":
        return np.min(stacked, axis=0), errors
    else:
        errors.append(f"Unknown projection method: {method}")
        return None, errors


def z_project_dataset(
    metadata: pd.DataFrame,
    intensity_colnames: List[str],
    mask_colnames: Optional[List[str]],
    method: str = "max",
    delete_originals: bool = False,
    group_cols: Optional[List[str]] = None
) -> Dict:
    """Perform Z-stack projection on dataset images.

    Groups images by metadata columns (excluding 'stack'), then performs
    projection along the Z-axis for each group.

    Parameters
    ----------
    metadata : pd.DataFrame
        Metadata DataFrame with image paths.
    intensity_colnames : list of str
        Column names for intensity images.
    mask_colnames : list of str or None
        Column names for mask images.
    method : str
        Projection method: "max", "mean", or "min".
    delete_originals : bool
        Whether to delete original Z-stack images after projection.
    group_cols : list of str, optional
        Columns to group by. If None, auto-detected from metadata.

    Returns
    -------
    dict
        Summary with keys: "success", "projected", "groups", "groups_processed",
        "groups_skipped", "deleted", "errors".
    """
    summary = {
        "success": False,
        "projected": 0,
        "groups": 0,
        "groups_processed": 0,
        "groups_skipped": 0,
        "deleted": [],
        "errors": []
    }
    
    if metadata is None or metadata.empty:
        summary["errors"].append("No metadata available")
        print("[Z-Projection] ERROR: No metadata available")
        return summary
    
    if 'stack' not in metadata.columns:
        summary["errors"].append("No 'stack' column found in metadata")
        print("[Z-Projection] ERROR: No 'stack' column found in metadata")
        return summary
    
    if group_cols is None:
        group_cols = [
            c for c in metadata.columns 
            if c not in ['stack'] 
            and c not in intensity_colnames 
            and c not in (mask_colnames or [])
            and c != 'directory'
        ]
    
    grouped = metadata.groupby(group_cols, sort=False)
    total_groups = len(grouped)
    
    print(f"[Z-Projection] Starting Z-stack projection")
    print(f"[Z-Projection] Method: {method}")
    print(f"[Z-Projection] Total groups to process: {total_groups}")
    print(f"[Z-Projection] Delete originals: {delete_originals}")
    
    for group_key, group_df in tqdm(grouped, desc="Z-projection", unit="group"):
        if len(group_df) <= 1:
            summary["groups_skipped"] += 1
            continue
        
        summary["groups"] += 1
        group_errors = []
        
        try:
            paths_to_delete = []
            
            for ch in intensity_colnames:
                if ch not in group_df.columns:
                    continue
                
                image_paths = []
                for _, row in group_df.iterrows():
                    if pd.isna(row[ch]):
                        continue
                    img_path = Path(row['directory']) / row[ch]
                    if img_path.exists():
                        image_paths.append(img_path)
                
                if len(image_paths) == 0:
                    continue
                
                projected, errs = z_project_group(image_paths, method)
                group_errors.extend(errs)
                
                if projected is None:
                    continue
                
                first_path = image_paths[0]
                new_name = f"{first_path.stem}_zp{method}{first_path.suffix}"
                save_path = first_path.parent / new_name
                
                try:
                    imwrite(save_path, projected, compression='zlib')
                    paths_to_delete.extend(image_paths[1:])
                    summary["projected"] += 1
                except Exception as e:
                    group_errors.append(f"Failed to save {save_path.name}: {e}")
            
            if mask_colnames:
                for mask_col in mask_colnames:
                    if mask_col not in group_df.columns:
                        continue
                    
                    mask_paths = []
                    for _, row in group_df.iterrows():
                        if pd.isna(row[mask_col]):
                            continue
                        mask_path = Path(row['directory']) / row[mask_col]
                        if mask_path.exists():
                            mask_paths.append(mask_path)
                    
                    if len(mask_paths) == 0:
                        continue
                    
                    projected_mask, errs = z_project_group(mask_paths, method="max")
                    group_errors.extend(errs)
                    
                    if projected_mask is None:
                        continue
                    
                    first_mask_path = mask_paths[0]
                    new_mask_name = f"{first_mask_path.stem}_zp{method}{first_mask_path.suffix}"
                    save_mask_path = first_mask_path.parent / new_mask_name
                    
                    try:
                        imwrite(
                            save_mask_path, 
                            projected_mask.astype(np.uint16), 
                            compression='zlib'
                        )
                        paths_to_delete.extend(mask_paths[1:])
                    except Exception as e:
                        group_errors.append(f"Failed to save mask {save_mask_path.name}: {e}")
            
            if delete_originals:
                for path in paths_to_delete:
                    try:
                        path.unlink()
                        summary["deleted"].append(path.name)
                    except Exception as e:
                        group_errors.append(f"Failed to delete {path}: {e}")
            
            if group_errors:
                summary["errors"].extend([f"Group {group_key}: {e}" for e in group_errors])
            
            summary["groups_processed"] += 1
        
        except Exception as e:
            summary["groups_skipped"] += 1
            summary["errors"].append(f"Error processing group {group_key}: {e}")
    
    summary["success"] = summary["projected"] > 0
    
    print(f"[Z-Projection] Summary:")
    print(f"  - Total groups: {total_groups}")
    print(f"  - Groups processed: {summary['groups_processed']}")
    print(f"  - Groups skipped: {summary['groups_skipped']}")
    print(f"  - Images projected: {summary['projected']}")
    print(f"  - Originals deleted: {len(summary['deleted'])}")
    if summary["errors"]:
        print(f"  - Errors: {len(summary['errors'])}")
    
    return summary

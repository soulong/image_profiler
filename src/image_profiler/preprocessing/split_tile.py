"""Tile Splitter for Microscopy Images.

Splits every TIFF image into non-overlapping tiles of given pixel dimensions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tifffile import imread, imwrite
from tqdm import tqdm


def split_image_into_tiles(
    image_path: Path,
    tile_w_px: int,
    tile_h_px: int,
    output_dir: Optional[Path] = None,
) -> Tuple[int, List[Path], List[str]]:
    """Split a single image into tiles of exactly tile_w_px × tile_h_px pixels.

    The last tiles on right/bottom are cropped if image size is not divisible.

    Parameters
    ----------
    image_path : Path
        Path to the image to split.
    tile_w_px : int
        Tile width in pixels.
    tile_h_px : int
        Tile height in pixels.
    output_dir : Path, optional
        Output directory for tiles. Default: same as input.

    Returns
    -------
    tuple of (int, list of Path, list of str)
        (total_tiles_created, list_of_saved_paths, errors)
    """
    errors = []
    
    try:
        img = imread(image_path)
        if img.ndim != 2:
            errors.append(f"Skipping {image_path.name}: not a 2D image (ndim={img.ndim})")
            return 0, [], errors

        if img.dtype != np.uint16:
            img = img.astype(np.uint16)

        height, width = img.shape

        if tile_w_px <= 0 or tile_h_px <= 0:
            errors.append(f"Invalid tile size: {tile_w_px}x{tile_h_px}")
            return 0, [], errors

        saved_paths = []
        tile_idx = 0

        save_dir = output_dir if output_dir else image_path.parent
        save_dir.mkdir(parents=True, exist_ok=True)

        for y_start in range(0, height, tile_h_px):
            for x_start in range(0, width, tile_w_px):
                y_end = min(y_start + tile_h_px, height)
                x_end = min(x_start + tile_w_px, width)

                tile = img[y_start:y_end, x_start:x_end]

                new_name = f"{image_path.stem}_tile{tile_idx:04d}{image_path.suffix}"
                output_path = save_dir / new_name

                imwrite(output_path, tile, compression='zlib')
                saved_paths.append(output_path)
                tile_idx += 1

        return len(saved_paths), saved_paths, errors

    except Exception as e:
        errors.append(f"Error processing {image_path.name}: {e}")
        return 0, [], errors


def tile_images_from_metadata(
    dataset: Dict,
    tile_w_px: int = 512,
    tile_h_px: int = 512,
    delete_originals: bool = False,
) -> Dict:
    """Process images from dataset metadata, split into tiles.

    Parameters
    ----------
    dataset : dict
        Dictionary with 'metadata', 'intensity_colnames', 'mask_colnames'.
    tile_w_px : int
        Tile width in pixels.
    tile_h_px : int
        Tile height in pixels.
    delete_originals : bool
        Whether to delete original images after tiling.

    Returns
    -------
    dict
        Summary with keys: "success", "processed", "tiles_created", "skipped", 
        "saved", "deleted", "errors".
    """
    metadata = dataset.get("metadata")
    intensity_cols = dataset.get("intensity_colnames", [])
    
    summary = {
        "success": False,
        "processed": 0,
        "tiles_created": 0,
        "skipped": 0,
        "saved": [],
        "deleted": [],
        "errors": []
    }
    
    if metadata is None or metadata.empty:
        summary["errors"].append("No metadata available")
        print("[Tile Split] ERROR: No metadata available")
        return summary
    
    all_image_paths = []
    for idx, row in metadata.iterrows():
        directory = Path(row.get("directory", ""))
        for col in intensity_cols:
            if col in metadata.columns:
                val = row.get(col)
                if isinstance(val, list):
                    all_image_paths.extend([directory / p for p in val if pd.notna(p)])
                elif pd.notna(val):
                    all_image_paths.append(directory / val)
    
    if not all_image_paths:
        summary["errors"].append("No image paths found in metadata")
        print("[Tile Split] ERROR: No image paths found in metadata")
        return summary
    
    all_image_paths = list(set(all_image_paths))
    total_images = len(all_image_paths)
    
    print(f"[Tile Split] Starting tiling for {total_images} image(s)")
    print(f"[Tile Split] Tile size: {tile_w_px}x{tile_h_px} pixels")
    print(f"[Tile Split] Delete originals: {delete_originals}")
    
    progress = tqdm(all_image_paths, desc="Tiling Images", unit="image")
    
    for img_path in progress:
        if not img_path.exists():
            summary["errors"].append(f"File not found: {img_path}")
            summary["skipped"] += 1
            continue
            
        progress.set_postfix({
            "file": img_path.name[:30],
            "tile": f"{tile_w_px}×{tile_h_px}"
        })
        
        n_tiles, tile_paths, errs = split_image_into_tiles(img_path, tile_w_px, tile_h_px)
        summary["errors"].extend(errs)
        
        if n_tiles == 0:
            summary["skipped"] += 1
            continue
        
        summary["processed"] += 1
        summary["tiles_created"] += n_tiles
        summary["saved"].extend([p.name for p in tile_paths])
        
        if delete_originals:
            try:
                img_path.unlink()
                summary["deleted"].append(img_path.name)
            except Exception as e:
                msg = f"Failed to delete original {img_path.name}: {e}"
                summary["errors"].append(msg)
    
    summary["success"] = summary["processed"] > 0
    
    print(f"[Tile Split] Summary:")
    print(f"  - Total images: {total_images}")
    print(f"  - Processed: {summary['processed']}")
    print(f"  - Skipped: {summary['skipped']}")
    print(f"  - Tiles created: {summary['tiles_created']}")
    print(f"  - Originals deleted: {len(summary['deleted'])}")
    if summary["errors"]:
        print(f"  - Errors: {len(summary['errors'])}")
    
    return summary


# def tile_images_in_directory(
#     directory: Path,
#     tile_w_px: int = 512,
#     tile_h_px: int = 512,
#     delete_originals: bool = False,
# ) -> Dict:
#     """Process all .tiff/.tif files in one Images directory.

#     Legacy function for CLI usage.

#     Parameters
#     ----------
#     directory : Path
#         Directory containing images.
#     tile_w_px : int
#         Tile width in pixels.
#     tile_h_px : int
#         Tile height in pixels.
#     delete_originals : bool
#         Whether to delete original images after tiling.

#     Returns
#     -------
#     dict
#         Summary of tiling operation.
#     """
#     directory = Path(directory)
    
#     summary = {
#         "success": False,
#         "processed": 0,
#         "tiles_created": 0,
#         "skipped": 0,
#         "saved": [],
#         "deleted": [],
#         "errors": []
#     }
    
#     if not directory.is_dir():
#         summary["errors"].append(f"Not a directory: {directory}")
#         print(f"[Tile Split] ERROR: Not a directory: {directory}")
#         return summary

#     tiff_files = list(directory.glob("*.tif")) + list(directory.glob("*.tiff"))
    
#     if not tiff_files:
#         summary["errors"].append("No TIFF files found")
#         print(f"[Tile Split] ERROR: No TIFF files found in {directory}")
#         return summary

#     print(f"[Tile Split] Starting tiling for {len(tiff_files)} file(s)")
#     print(f"[Tile Split] Tile size: {tile_w_px}x{tile_h_px} pixels")
#     print(f"[Tile Split] Delete originals: {delete_originals}")

#     progress = tqdm(tiff_files, desc="Tiling Images", unit="image")

#     for img_path in progress:
#         progress.set_postfix({
#             "file": img_path.name[:30],
#             "tile": f"{tile_w_px}×{tile_h_px}"
#         })

#         n_tiles, tile_paths, errs = split_image_into_tiles(img_path, tile_w_px, tile_h_px)
#         summary["errors"].extend(errs)

#         if n_tiles == 0:
#             summary["skipped"] += 1
#             continue

#         summary["processed"] += 1
#         summary["tiles_created"] += n_tiles
#         summary["saved"].extend([p.name for p in tile_paths])

#         if delete_originals:
#             try:
#                 img_path.unlink()
#                 summary["deleted"].append(img_path.name)
#             except Exception as e:
#                 msg = f"Failed to delete original {img_path.name}: {e}"
#                 summary["errors"].append(msg)

#     summary["success"] = summary["processed"] > 0
    
#     print(f"\n[Tile Split] Summary:")
#     print(f"  - Total files: {len(tiff_files)}")
#     print(f"  - Processed: {summary['processed']}")
#     print(f"  - Skipped: {summary['skipped']}")
#     print(f"  - Tiles created: {summary['tiles_created']}")
#     print(f"  - Originals deleted: {len(summary['deleted'])}")
#     if summary["errors"]:
#         print(f"  - Errors: {len(summary['errors'])}")

#     return summary

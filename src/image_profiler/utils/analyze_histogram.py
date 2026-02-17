#!/usr/bin/env python
import argparse
import json
import logging
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import imageio.v3 as iio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from scipy import stats
from skimage.util import img_as_float
from tqdm import tqdm

# Assuming these are in your PYTHONPATH as per your previous script
from image_helper import (
    images_to_dataset, 
    crop_cells, 
    setup_logger, 
    find_measurement_dirs, 
    _parse_dataset_kwargs
)



# =========================================================================
# 1. UPDATED CORE ANALYSIS (Returns list of bins per cell)
# =========================================================================
def intensity_bins_analysis(
    cell_data_list: List[Dict[str, Any]], 
    min_intensity: Optional[float] = None,
    max_intensity: Optional[float] = None,
    log_transform: bool = False,
    normalize: Optional[str] = None,
    n_bins: int = 7,
    bin_method: str = 'linear',  
    manual_edges: Optional[Union[List[float], np.ndarray]] = None,
) -> List[Dict[str, Any]]:
    results = []
    for data in cell_data_list: 
        cell_id = int(data.get('cell_id'))
        img = data.get('cell_img')
        mask = data.get('cell_mask') 
        
        if img is None or mask is None:
            continue

        # 0. Get mask covered region
        gray = img[:,:,0]
        valid_pixels = gray[mask > 0].astype(float)
        
        if len(valid_pixels) == 0:
            continue

        # 1. Clip by min/max
        if min_intensity is not None or max_intensity is not None:
            valid_pixels = np.clip(valid_pixels, min_intensity, max_intensity)
        
        # 2. Log transform
        if log_transform:
            valid_pixels = np.log(valid_pixels + 1e-8)
            
        # 3. Normalize
        if normalize == 'mean':
            norm_factor = valid_pixels.mean()
            valid_pixels /= (norm_factor if norm_factor > 1e-8 else 1.0)
        elif normalize == 'quantile':
            q01, q99 = np.percentile(valid_pixels, (0.01, 99.99))
            valid_pixels = np.clip((valid_pixels - q01) / (q99 - q01 + 1e-8), 0, 1)
        elif normalize == 'median':
            med = np.median(valid_pixels)
            valid_pixels /= (med if med > 1e-8 else 1.0)

        # 4. Define Bin Edges
        p_min, p_max = valid_pixels.min(), valid_pixels.max()
        if bin_method == 'manual' and manual_edges is not None:
            bin_edges = np.array(manual_edges)
        # elif bin_method == 'logspace':
        #     # logspace uses exponents: 10^start to 10^stop
        #     bin_edges = np.logspace(np.log10(max(p_min, 1e-8)), np.log10(max(p_max, 1e-7)), n_bins + 1)
        elif bin_method == 'geomspace':
            # geomspace uses actual start/stop values
            bin_edges = np.geomspace(max(p_min, 1e-8), max(p_max, 1e-7), n_bins + 1)
        elif bin_method == 'sqrt':
            bin_edges = np.power(np.linspace(np.sqrt(p_min), np.sqrt(p_max), n_bins + 1), 2)
        else: # linear
            bin_edges = np.linspace(p_min, p_max, n_bins + 1)
        
        # 5. Calculate Bins
        counts, _ = np.histogram(valid_pixels, bins=bin_edges)
        bin_indices = np.digitize(valid_pixels, bin_edges) - 1
        
        cell_bins = []
        for i in range(len(bin_edges) - 1):
            mask_in_bin = (bin_indices == i)
            bin_mean = float(valid_pixels[mask_in_bin].mean()) if np.any(mask_in_bin) else 0.0
            cell_bins.append({
                'bin_index': i,
                'bin_count': int(counts[i]),
                'bin_mean': round(bin_mean, 4),
                'bin_edge_low': round(float(bin_edges[i]), 4),
                'bin_edge_high': round(float(bin_edges[i+1]), 4)
            })

        results.append({
            'cell_id': cell_id,
            'total_pixels': len(valid_pixels),
            'mean_intensity': round(float(valid_pixels.mean()), 4),
            'bins': cell_bins
        })
    return results

# =========================================================================
# 2. UPDATED DATABASE EXPORT (LONG SHAPE)
# =========================================================================
def export_to_sqlite(results: List[Dict[str, Any]], metadata: Dict[str, Any], db_path: Path, table_name: str):
    """Saves results where each row is one bin measurement."""
    if not results: return
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Define Column Structure
    meta_cols = {k: "TEXT" for k in metadata.keys()}
    core_cols = {
        'cell_id': 'INTEGER',
        'total_pixels': 'INTEGER',
        'cell_mean_intensity': 'REAL',
        'bin_index': 'INTEGER',
        'bin_count': 'INTEGER',
        'bin_mean': 'REAL',
        'bin_edge_low': 'REAL',
        'bin_edge_high': 'REAL'
    }

    all_cols = {**meta_cols, **core_cols}
    col_def = ", ".join([f"{k} {v}" for k, v in all_cols.items()])
    
    # Primary Key now includes bin_index to ensure each bin is unique
    pk_cols = list(metadata.keys()) + ['cell_id', 'bin_index']
    cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({col_def}, PRIMARY KEY ({', '.join(pk_cols)}))")
    
    # Prepare Data for Batch Insert
    insert_keys = list(all_cols.keys())
    placeholders = ", ".join(["?"] * len(insert_keys))
    sql = f"INSERT OR REPLACE INTO {table_name} ({','.join(insert_keys)}) VALUES ({placeholders})"
    
    rows_to_insert = []
    for res in results:
        # Fixed image/cell metadata for this cell
        base_row = [metadata[k] for k in metadata.keys()]
        base_row += [res['cell_id'], res['total_pixels'], res['mean_intensity']]
        
        # Create a new row for every bin in the cell
        for b in res['bins']:
            bin_row = base_row + [
                b['bin_index'], 
                b['bin_count'], 
                b['bin_mean'], 
                b['bin_edge_low'], 
                b['bin_edge_high']
            ]
            rows_to_insert.append(bin_row)
        
    cursor.executemany(sql, rows_to_insert)
    conn.commit()
    conn.close()
# =========================================================================
# 3. MAIN TERMINAL EXECUTION
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Pixel Intensity Binning Analysis CLI",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("root_dir", type=str, help="Root folder containing measurement sub-folders")
    parser.add_argument("--min_intensity", type=float, default=None, help="Floor pixel intensities at this value, untransformed or normalized intensity")
    parser.add_argument("--max_intensity", type=float, default=None, help="Cap pixel intensities at this value, untransformed or normalized intensity")
    parser.add_argument("--log_transform", action="store_true")
    parser.add_argument("--normalize", choices=['mean', 'median', 'quantile', 'none'], default='none',help="Normalization applied after log transform but before binning")
    parser.add_argument("--n_bins", type=int, default=7)
    parser.add_argument("--bin_method", choices=['linear', 'sqrt', 'geomspace', 'manual'], default='linear', help="Algorithm used to define bin spacing")
    parser.add_argument("--manual_edges", nargs='+', type=float, default=[0.0, 0.2, 0.5, 0.7, 0.85, 0.92, 0.95, 1.0])
    parser.add_argument("--overwrite_db", action="store_true", help="Overwrite existing database file")
    parser.add_argument("--db_name", default="histogram.db", help="Output SQLite database name")
    parser.add_argument("--dataset_kwargs", type=str, default="{}",
                        help="JSON string with overrides for images_to_dataset(). "
                             "Examples:\n"
                             "  --dataset_kwargs '{\"image_suffix\": \".tif\"}'\n"
                             "  --dataset_kwargs '{\"image_subdir\": \"Images\"}'")
    
    args = parser.parse_args()
    user_dataset_kwargs = _parse_dataset_kwargs(args.dataset_kwargs)
    
    root = Path(args.root_dir)
    log_dir = root / "logs"
    logger = setup_logger(log_dir, name="intensity_binning")
    logger.info(f"Starting Analysis: {root}")

    measurements = find_measurement_dirs(root)
    if not measurements:
        logger.error("No measurement directories found."); sys.exit(1)

    for mdir in measurements:
        logger.info(f"Processing measurement: {mdir.name}")
    
        db_path = mdir / args.db_name
        if db_path.exists() and not args.overwrite_db:
            logger.info(f"Database {db_path} already exists → skipping measurement")
            return True
        if db_path.exists():
            db_path.unlink()  # remove old
    
        # Merge defaults with user overrides just like run_cellpose.py
        dataset = images_to_dataset(mdir, **user_dataset_kwargs)
        if not dataset:
            continue
            
        df = dataset['df']
        exclude_meta = {'ch1', 'cell', 'path_ch0', 'path_cell'}
        meta_keys = [c for c in df.columns if c not in exclude_meta]
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Analyzing {mdir.name}"):
            try:
                # Assuming 'ch1' and 'cell' are the standard column names produced by images_to_dataset
                cells = crop_cells(
                    mask=row['directory'] / row['cell'], 
                    imgs=row['directory'] / row['ch1'], 
                    scale_factor=None, target_size=None, clip_mask=True, pad_square=False, rotate_horizontal=False)
                # print(np.max(cells[0]['cell_img']))
                
                results = intensity_bins_analysis(
                    cells, 
                    min_intensity=args.min_intensity, max_intensity=args.max_intensity, log_transform=args.log_transform,
                    normalize=args.normalize, n_bins=args.n_bins, bin_method=args.bin_method, manual_edges=args.manual_edges)
                # print(results)
                
                image_metadata = {k: str(row[k]) for k in meta_keys}
                export_to_sqlite(results, image_metadata, db_path, "cell")
            except Exception as e:
                logger.error(f"Error processing: {e}")

    logger.info("Pipeline complete.")

if __name__ == "__main__":
    main()
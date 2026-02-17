"""Helper utilities for image dataset management."""

from __future__ import annotations

import re
import string
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from natsort import natsorted

try:
    import imageio.v3 as iio
except ImportError:
    import imageio as iio



def find_measurement_dirs(
    root_dir: Union[str, Path],
    measurement_pattern: str = r".*Measurement(?:\s*\d+)?$",
    image_subdir: str = "Images"
) -> List[Path]:
    """Find all measurement directories containing an Images subfolder.

    Parameters
    ----------
    root_dir : str or Path
        Root directory to search.
    measurement_pattern : str
        Regex pattern for measurement directory names.
    image_subdir : str
        Name of the images subdirectory.

    Returns
    -------
    list of Path
        Sorted list of measurement directory paths.
    """
    root = Path(root_dir)
    
    if not root.is_dir():
        raise NotADirectoryError(f"{root} is not a valid directory")
    
    if not root.exists():
        raise FileNotFoundError(f"{root} does not exist")
    
    pattern = re.compile(measurement_pattern, re.IGNORECASE)
    
    images_dirs = root.rglob(image_subdir)
    
    measurement_dirs = []
    for img_dir in images_dirs:
        parent = img_dir.parent
        if pattern.match(parent.name):
            measurement_dirs.append(parent)
    
    return natsorted(measurement_dirs, key=lambda p: str(p))


def images_to_dataset(
    measurement_dir: Union[str, Path],
    image_pattern: str = (
        r"r(?P<row>.*)c(?P<column>.*)f(?P<field>.*)p(?P<stack>.*)-ch(?P<channel>.*)"
        r"sk(?P<timepoint>[0-9]{1,})fk1fl1"
        r"\.(?P<ext>tiff|png)"
    ),
    mask_pattern: str = (
        r"r(?P<row>.*)c(?P<column>.*)f(?P<field>.*)p(?P<stack>.*)-ch(?P<channel>.*)"
        r"sk(?P<timepoint>[0-9]{1,})fk1fl1"
        r"_cp_masks_(?P<mask_name>.*)"
        r"\.(?P<ext>tiff|png)"
    ),
    subset_pattern: Optional[str] = None,
    image_subdir: str = "Images",
    remove_na_row: bool = True
) -> Optional[Dict]:
    """Convert a folder of images into a tidy DataFrame.

    Parameters
    ----------
    measurement_dir : str or Path
        Root directory of the measurement.
    image_pattern : str
        Regex pattern for image filenames with named groups.
    mask_pattern : str
        Regex pattern for mask filenames (with _cp_masks_ suffix).
    subset_pattern : str, optional
        Additional regex to filter file paths.
    image_subdir : str
        Sub-directory containing images.
    remove_na_row : bool
        If True, drop rows with NaN values.

    Returns
    -------
    dict or None
        Dictionary with keys:
        - "metadata": pd.DataFrame
        - "metadata_colnames": list of str
        - "intensity_colnames": list of str
        - "mask_colnames": list of str or None
    """
    measurement_dir = Path(measurement_dir)
    if not measurement_dir.is_dir():
        return None

    glob_path = str(measurement_dir / image_subdir / "**/*")
    all_file_paths = [Path(p) for p in glob(glob_path, recursive=True)]
    
    image_paths = [p for p in all_file_paths if re.search(image_pattern, str(p))]
    mask_paths = [p for p in all_file_paths if re.search(mask_pattern, str(p))]
    
    if subset_pattern:
        image_paths = [p for p in image_paths if re.search(subset_pattern, str(p))]
        mask_paths = [p for p in mask_paths if re.search(subset_pattern, str(p))]

    if not image_paths:
        return None

    image_df = pd.DataFrame({
        "directory": [p.parent for p in image_paths],
        "filename": [p.name for p in image_paths]
    })
    
    parsed_image = image_df["filename"].str.extract(image_pattern)

    if "channel" not in parsed_image.columns:
        parsed_image["channel"] = "0"
    parsed_image["channel"] = "ch" + parsed_image["channel"].astype(str).replace("", "1")

    metadata_cols = [c for c in parsed_image.columns if c not in {"channel", "ext"}]

    intensity_df = pd.concat(
        [image_df.reset_index(drop=True), parsed_image.reset_index(drop=True)],
        axis=1
    )
    intensity_df = intensity_df.set_index(["directory"] + metadata_cols)
    intensity_channels = natsorted(intensity_df["channel"].unique())
    intensity_df = intensity_df.pivot(columns="channel", values="filename")

    mask_channels = None
    if mask_paths:
        mask_df = pd.DataFrame({
            "directory": [p.parent for p in mask_paths],
            "filename": [p.name for p in mask_paths]
        })
        
        parsed_mask = mask_df["filename"].str.extract(mask_pattern)
        
        mask_metadata_cols = [c for c in parsed_mask.columns if c not in {"mask_name", "channel", "ext"}]
        
        mask_combined = pd.concat(
            [mask_df.reset_index(drop=True), parsed_mask.reset_index(drop=True)],
            axis=1
        )
        mask_combined = mask_combined.set_index(["directory"] + mask_metadata_cols)
        mask_channels = natsorted(mask_combined["mask_name"].unique())
        mask_combined = mask_combined.pivot(columns="mask_name", values="filename")
        
        if intensity_df.index.names == mask_combined.index.names:
            intensity_df = intensity_df.join(mask_combined, how="left")
        else:
            print('indensity_df index:\n', intensity_df.index.names)
            print('mask_combined index:\n', mask_combined.index.names)

    tidy_df = intensity_df.reset_index()

    for col in tidy_df.select_dtypes(include="object"):
        try:
            tidy_df[col] = pd.to_numeric(tidy_df[col], errors="raise")
        except (ValueError, TypeError):
            pass

    if remove_na_row:
        tidy_df = tidy_df.dropna()

    metadata_colnames = ["directory"] + metadata_cols
    
    if "well" not in tidy_df.columns and {"row", "column"}.issubset(tidy_df.columns):
        row_sample = tidy_df["row"].dropna().astype(str).str.strip().str.upper()
        if row_sample.str.match(r"^\d+$").all():
            row_map = {i: letter for i, letter in enumerate(string.ascii_uppercase, 1)}
            tidy_df["row"] = pd.to_numeric(tidy_df["row"], errors="coerce").map(row_map).fillna("")
        else:
            tidy_df["row"] = tidy_df["row"].astype(str)
        tidy_df["column"] = tidy_df["column"].astype("Int64").astype(str)
        tidy_df.insert(
            tidy_df.columns.get_loc("directory") + 1,
            "well",
            tidy_df["row"] + tidy_df["column"]
        )
        tidy_df = tidy_df.drop(columns=["row", "column"])
        metadata_colnames = ["directory", "well"] + [
            c for c in metadata_cols if c not in {"row", "column"}
        ]

    return {
        "metadata": tidy_df,
        "metadata_colnames": metadata_colnames,
        "intensity_colnames": intensity_channels,
        "mask_colnames": mask_channels,
    }


def write_dataloader(
    metadata: pd.DataFrame,
    image_colnames: List[str],
    mask_colnames: Optional[List[str]],
    out_path: Optional[str] = None
) -> pd.DataFrame:
    """Convert metadata to CellProfiler-compatible CSV format.

    Parameters
    ----------
    metadata : pd.DataFrame
        Metadata DataFrame.
    image_colnames : list of str
        Column names for intensity images.
    mask_colnames : list of str or None
        Column names for mask images.
    out_path : str, optional
        Output CSV path.

    Returns
    -------
    pd.DataFrame
        Converted DataFrame.
    """
    df = metadata.copy()
    
    for ch in image_colnames:
        df[f"Image_PathName_{ch}"] = df["directory"]
        df = df.rename(columns={ch: f"Image_FileName_{ch}"})
    
    if mask_colnames:
        for m in mask_colnames:
            df[f"Image_ObjectsPathName_mask_{m}"] = df["directory"]
            df = df.rename(columns={m: f"Image_ObjectsFileName_mask_{m}"})
    
    for meta in metadata.columns:
        if meta not in image_colnames and (not mask_colnames or meta not in mask_colnames):
            df = df.rename(columns={meta: f"Metadata_{meta}"})
    
    if out_path:
        df.to_csv(out_path, index=False)
    
    return df

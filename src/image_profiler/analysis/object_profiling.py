"""Object-level profiling for extracting per-cell features."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from skimage.measure import regionprops_table
from image_profiler.analysis.extra_properties import rename_regionprops_table


def _skewness(arr: np.ndarray) -> float:
    """Calculate skewness of an array."""
    mean = np.mean(arr)
    std = np.std(arr)
    if std == 0:
        return 0.0
    return float(np.mean(((arr - mean) / std) ** 3))


def relate_masks(
    parent_mask: np.ndarray,
    child_mask: np.ndarray,
) -> Dict[int, int]:
    """Establish parent-child relationships between two mask layers.

    For each child object, find the parent object that contains
    the majority of the child's pixels.

    Parameters
    ----------
    parent_mask : np.ndarray
        Parent segmentation mask (e.g., cell body).
    child_mask : np.ndarray
        Child segmentation mask (e.g., nuclei).

    Returns
    -------
    dict of int to int
        Mapping child_id -> parent_id.  parent_id == 0 means no parent found.
    """
    child_ids = np.unique(child_mask[child_mask != 0])
    relationships: Dict[int, int] = {}

    for child_id in child_ids:
        child_pixels = child_mask == child_id
        parent_ids_at_child = parent_mask[child_pixels]
        parent_ids_at_child = parent_ids_at_child[parent_ids_at_child != 0]

        if len(parent_ids_at_child) == 0:
            relationships[child_id] = 0
            continue

        unique_parents, counts = np.unique(parent_ids_at_child, return_counts=True)
        relationships[child_id] = int(unique_parents[np.argmax(counts)])

    return relationships


def is_boundary_object(
    mask: np.ndarray,
    image_shape: tuple,
    threshold: float = 0.1,
) -> bool:
    """Check if an object touches the image boundary.

    An object is a boundary object when >= ``threshold`` fraction of its
    pixels touch any image edge (row 0, row max, col 0, col max).

    Parameters
    ----------
    mask : np.ndarray
        Binary mask of the object.
    image_shape : tuple
        Shape of the image (height, width).
    threshold : float
        Fraction of boundary pixels required.  Default 0.1 (10%).

    Returns
    -------
    bool
    """
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return False

    height, width = image_shape
    boundary_pixels = (
        (rows == 0) | (rows == height - 1) |
        (cols == 0) | (cols == width - 1)
    )
    return float(np.sum(boundary_pixels)) / len(rows) >= threshold


def _get_intensity_col(props_df: pd.DataFrame, stat: str, ch_idx: int) -> float:
    """Read a regionprops intensity column, handling single- vs multi-channel naming.

    skimage names multichannel intensity outputs as ``{stat}_{ch_idx}``
    (e.g. ``mean_intensity-0``) and single-channel outputs as ``{stat}``.
    """
    multichannel_key = f"{stat}_{ch_idx}"
    if multichannel_key in props_df.columns:
        return multichannel_key
    return stat


def _profile_objects(
    image: np.ndarray,
    mask: np.ndarray,
    channel_names: List[str],
    parent_mask: Optional[np.ndarray] = None,
    profile: Optional[List[str]] = ['shape', 'intensity'],
    extra_properties: Optional[List[Callable]] = None,
    col_names: Optional[Dict] = None,
) -> pd.DataFrame:
    """Profile objects in a single image.

    Parameters
    ----------
    image : np.ndarray
        Image stack with shape (C, Y, X).
    mask : np.ndarray
        Segmentation mask with shape (Y, X).  Each non-zero integer is one object.
    channel_names : list of str
        Channel names corresponding to the C axis of ``image``.
    parent_mask : np.ndarray, optional
        Parent mask for hierarchical relationship (adds ``parent_id`` column).
    profile : list of str, optional
        Feature families to compute.  Any combination of "shape" and "intensity".
        Default: both.
    extra_properties : list of callable, optional
        Additional functions for ``regionprops_table``, built via the factory
        functions in ``extra_properties.py``
        (``make_glcm_func``, ``make_granularity_func``, ``make_radial_func``).
    col_names : dict, optional
        Column-name metadata returned alongside factory functions.  Required
        to get human-readable column names for extra properties.  If None,
        raw skimage-generated names are kept.

    Returns
    -------
    pd.DataFrame
        One row per object.  Empty DataFrame when no objects are found.
    """
    # if profile is None:
    #     profile = ["shape", "intensity"]

    object_ids = np.unique(mask[mask != 0])
    if len(object_ids) == 0:
        return pd.DataFrame()

    # ------------------------------------------------------------------ #
    # Build regionprops property list                                      #
    # ------------------------------------------------------------------ #
    properties = ["label"]

    if profile and "shape" in profile:
        properties.extend([
            "area", "eccentricity", "solidity",
            "equivalent_diameter_area", "extent",
            "major_axis_length", "minor_axis_length", 
            # "perimeter","orientation","bbox", "centroid",
        ])

    ## do this part mannually
    # if "intensity" in profile:
    #     properties.extend([
    #         "mean_intensity", "max_intensity", "min_intensity", "std_intensity",
    #     ])

    # skimage regionprops expects intensity_image in (Y, X, C) for multichannel
    image_yxc = np.moveaxis(image, 0, -1)  # (C, Y, X) -> (Y, X, C)

    raw_props = regionprops_table(
        mask,
        intensity_image=image_yxc,
        properties=properties,
        extra_properties=extra_properties or [],
    )

    # Rename extra-property columns to human-readable names when col_names given
    if col_names:
        props_df = rename_regionprops_table(raw_props, col_names)
    else:
        props_df = pd.DataFrame(raw_props)

    n_objects = len(props_df)

    # ------------------------------------------------------------------ #
    # Pre-compute parent relationships once                                #
    # ------------------------------------------------------------------ #
    parent_relationships: Optional[Dict[int, int]] = None
    if parent_mask is not None:
        parent_relationships = relate_masks(parent_mask, mask)

    # Collect extra-property feature names for attachment
    extra_feat_names: List[str] = []
    if col_names:
        for entry in col_names.values():
            extra_feat_names.extend(
                n for n in entry["names"] if n in props_df.columns
            )

    # ------------------------------------------------------------------ #
    # Assemble per-object result rows                                      #
    # ------------------------------------------------------------------ #
    results = []
    for i in range(n_objects):
        row_raw = props_df.iloc[i]
        obj_id  = int(row_raw["label"])
        result: Dict[str, Any] = {"object_id": obj_id}
        
        # ----- boundary flag ------------------------------------------ #
        result["is_boundary"] = is_boundary_object(mask == obj_id, mask.shape)

        # ----- parent relationship ------------------------------------ #
        if parent_relationships is not None:
            result["parent_id"] = parent_relationships.get(obj_id, 0)
            
        # ----- shape features ----------------------------------------- #
        if profile and "shape" in profile:
            result["shape_area"]                = row_raw["area"]
            # result["shape_perimeter"]           = row_raw["perimeter"]
            result["shape_eccentricity"]        = row_raw["eccentricity"]
            result["shape_solidity"]            = row_raw["solidity"]
            result["shape_diameter"] = row_raw["equivalent_diameter_area"]
            result["shape_extent"]              = row_raw["extent"]
            result["shape_major_axis"]   = row_raw["major_axis_length"]
            result["shape_minor_axis"]   = row_raw["minor_axis_length"]
            # result["orientation"]         = row_raw["orientation"]
            # result["bbox_min_y"]          = row_raw["bbox-0"]
            # result["bbox_min_x"]          = row_raw["bbox-1"]
            # result["bbox_max_y"]          = row_raw["bbox-2"]
            # result["bbox_max_x"]          = row_raw["bbox-3"]
            # result["centroid_y"]          = row_raw["centroid-0"]
            # result["centroid_x"]          = row_raw["centroid-1"]

        # ----- intensity features ------------------------------------- #
        if profile and "intensity" in profile:
            obj_mask = mask == obj_id
            for ch_idx, ch_name in enumerate(channel_names):

                # for stat in ("intensity_mean", "intensity_max",
                #              "intensity_min", "intensity_std"):
                #     col_key = _get_intensity_col(props_df, stat, ch_idx)
                #     result[f"{stat}_{ch_name}"] = row_raw[col_key]

                obj_pixels = image[ch_idx][obj_mask]
                result[f"intensity_mean_{ch_name}"]   = float(np.mean(obj_pixels))
                result[f"intensity_std_{ch_name}"]    = float(np.std(obj_pixels))
                # result[f"intensity_min_{ch_name}"]    = float(np.min(obj_pixels))
                # result[f"intensity_max_{ch_name}"]    = float(np.max(obj_pixels))
                # result[f"intensity_median_{ch_name}"] = float(np.median(obj_pixels))
                result[f"intensity_sum_{ch_name}"]    = float(np.sum(obj_pixels))
                result[f"intensity_skew_{ch_name}"]   = _skewness(obj_pixels)
                for q in [0.1, 1, 25, 75, 99, 99.9]:
                    result[f"intensity_q{q}_{ch_name}"] = float(np.percentile(obj_pixels, q))

        # ----- extra property features -------------------------------- #
        for feat_name in extra_feat_names:
            result[feat_name] = row_raw[feat_name]

        results.append(result)

    return pd.DataFrame(results)


def profile_object_single_row(
    image_data: np.ndarray,
    mask_data: Dict[str, np.ndarray],
    metadata_row: Dict,
    channel_names: List[str],
    mask_name: str,
    parent_mask_name: Optional[str] = None,
    profile: Optional[List[str]] = None,
    extra_properties: Optional[List[Callable]] = None,
    col_names: Optional[Dict] = None,
) -> Optional[pd.DataFrame]:
    """Profile objects from a single dataset row.

    Parameters
    ----------
    image_data : np.ndarray
        Image stack with shape (C, Y, X).
    mask_data : dict of str to np.ndarray
        Mask arrays keyed by mask name (without the ``mask_`` prefix).
    metadata_row : dict
        Metadata for this row; image/mask path columns are stripped.
    channel_names : list of str
        Channel names corresponding to image_data channels.
    mask_name : str
        Key in ``mask_data`` to use as the segmentation mask.
    parent_mask_name : str, optional
        Key in ``mask_data`` to use as the parent mask.
    profile : list of str, optional
        Feature families: ``"shape"``, ``"intensity"``, or both.
    extra_properties : list of callable, optional
        Factory-built extra-property functions.
    col_names : dict, optional
        Column-name metadata for renaming extra-property outputs.

    Returns
    -------
    pd.DataFrame or None
        One row per object with metadata columns prepended,
        or None if inputs are invalid / no objects found.
    """
    if image_data is None or mask_name not in mask_data:
        return None

    parent_mask = mask_data.get(parent_mask_name) if parent_mask_name else None

    result_df = _profile_objects(
        image_data,
        mask_data[mask_name],
        channel_names,
        parent_mask=parent_mask,
        profile=profile,
        extra_properties=extra_properties,
        col_names=col_names,
    )

    if result_df.empty:
        return None

    # Strip image/mask path columns from metadata and prepend to result
    meta = {
        k: v for k, v in metadata_row.items() if not (k in channel_names)
    }
    
    for idx, key in enumerate(meta):
        result_df.insert(idx, key, meta[key])
    
    # for col, val in meta.items():
    #     result_df.insert(idx, col, val)
    #     result_df[col] = val

    if parent_mask_name is not None and "parent_id" in result_df.columns:
        result_df = result_df.rename(
            columns={"parent_id": f"parent_{parent_mask_name}_id"}
        )

    return result_df
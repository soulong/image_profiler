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
        Image stack with shape (C, Y, X), where C is the number of channels,
        Y is the height, and X is the width of the image.
    mask : np.ndarray
        Segmentation mask with shape (Y, X).  Each non-zero integer represents a unique object.
        Must be the same size as each channel in the image stack.
    channel_names : list of str
        Channel names corresponding to the C axis of ``image``.
        Must be in the same order as the channels in the image stack.
    parent_mask : np.ndarray, optional
        Parent segmentation mask with shape (Y, X) for hierarchical relationship.
        Adds a ``parent_id`` column to the output, indicating which parent object
        each child object belongs to.
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
        Columns include object ID, boundary flag, parent ID (if parent_mask provided),
        shape features, intensity features for each channel, and any extra properties.

    Examples
    --------
    >>> import numpy as np
    >>> image = np.random.rand(2, 100, 100)  # 2 channels, 100x100 image
    >>> mask = np.zeros((100, 100), dtype=int)
    >>> mask[25:75, 25:75] = 1  # Single object in the center
    >>> channel_names = ['DAPI', 'GFP']
    >>> result = _profile_objects(image, mask, channel_names)
    >>> print('Columns:', list(result.columns)[:10])  # Show first 10 columns
    ['object_id', 'is_boundary', 'shape_area', 'shape_eccentricity', 'shape_solidity', 'shape_diameter', 'shape_extent', 'shape_major_axis', 'shape_minor_axis', 'intensity_mean_DAPI']
    >>> print('Number of objects:', len(result))
    Number of objects: 1
    """
    # Validate inputs
    if not isinstance(image, np.ndarray):
        raise TypeError(f"image must be a numpy array, got {type(image).__name__}")
    
    if image.ndim != 3:
        raise ValueError(f"image must be 3D with shape (C, Y, X), got shape {image.shape}")
    
    if not isinstance(mask, np.ndarray):
        raise TypeError(f"mask must be a numpy array, got {type(mask).__name__}")
    
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D with shape (Y, X), got shape {mask.shape}")
    
    if image.shape[1:] != mask.shape:
        raise ValueError(f"Image shape ({image.shape[1:]}) must match mask shape ({mask.shape})")
    
    if not isinstance(channel_names, list) or not all(isinstance(name, str) for name in channel_names):
        raise TypeError("channel_names must be a list of strings")
    
    if len(channel_names) != image.shape[0]:
        raise ValueError(f"Number of channel names ({len(channel_names)}) must match number of channels in image ({image.shape[0]})")
    
    if parent_mask is not None:
        if not isinstance(parent_mask, np.ndarray):
            raise TypeError(f"parent_mask must be a numpy array, got {type(parent_mask).__name__}")
        if parent_mask.ndim != 2:
            raise ValueError(f"parent_mask must be 2D with shape (Y, X), got shape {parent_mask.shape}")
        if parent_mask.shape != mask.shape:
            raise ValueError(f"Parent mask shape ({parent_mask.shape}) must match mask shape ({mask.shape})")
    
    if profile is not None:
        if not isinstance(profile, list) or not all(isinstance(p, str) for p in profile):
            raise TypeError("profile must be a list of strings")
        for p in profile:
            if p not in ['shape', 'intensity']:
                raise ValueError(f"profile must contain only 'shape' and/or 'intensity', got '{p}'")
    
    if extra_properties is not None and not isinstance(extra_properties, list):
        raise TypeError("extra_properties must be a list of callables")
    
    if col_names is not None and not isinstance(col_names, dict):
        raise TypeError("col_names must be a dictionary")

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
        ])

    # skimage regionprops expects intensity_image in (Y, X, C) for multichannel
    image_yxc = np.moveaxis(image, 0, -1)  # (C, Y, X) -> (Y, X, C)

    try:
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
            try:
                parent_relationships = relate_masks(parent_mask, mask)
            except Exception as e:
                print(f"Warning: Failed to compute parent relationships: {e}")
                parent_relationships = None

        # Collect extra-property feature names for attachment
        extra_feat_names: List[str] = []
        if col_names:
            try:
                for entry in col_names.values():
                    extra_feat_names.extend(
                        n for n in entry["names"] if n in props_df.columns
                    )
            except Exception as e:
                print(f"Warning: Failed to collect extra property names: {e}")
                extra_feat_names = []

        # ------------------------------------------------------------------ #
        # Assemble per-object result rows                                      #
        # ------------------------------------------------------------------ #
        results = []
        for i in range(n_objects):
            try:
                row_raw = props_df.iloc[i]
                obj_id  = int(row_raw["label"])
                result: Dict[str, Any] = {"object_id": obj_id}
                
                # ----- boundary flag ------------------------------------------ #
                try:
                    result["is_boundary"] = is_boundary_object(mask == obj_id, mask.shape)
                except Exception as e:
                    print(f"Warning: Failed to compute boundary flag for object {obj_id}: {e}")
                    result["is_boundary"] = False

                # ----- parent relationship ------------------------------------ #
                if parent_relationships is not None:
                    result["parent_id"] = parent_relationships.get(obj_id, 0)
                    
                # ----- shape features ----------------------------------------- #
                if profile and "shape" in profile:
                    try:
                        result["shape_area"]                = row_raw["area"]
                        result["shape_eccentricity"]        = row_raw["eccentricity"]
                        result["shape_solidity"]            = row_raw["solidity"]
                        result["shape_diameter"] = row_raw["equivalent_diameter_area"]
                        result["shape_extent"]              = row_raw["extent"]
                        result["shape_major_axis"]   = row_raw["major_axis_length"]
                        result["shape_minor_axis"]   = row_raw["minor_axis_length"]
                    except Exception as e:
                        print(f"Warning: Failed to compute shape features for object {obj_id}: {e}")

                # ----- intensity features ------------------------------------- #
                if profile and "intensity" in profile:
                    try:
                        obj_mask = mask == obj_id
                        for ch_idx, ch_name in enumerate(channel_names):
                            obj_pixels = image[ch_idx][obj_mask]
                            if obj_pixels.size == 0:
                                result[f"intensity_mean_{ch_name}"]   = 0.0
                                result[f"intensity_std_{ch_name}"]    = 0.0
                                result[f"intensity_sum_{ch_name}"]    = 0.0
                                result[f"intensity_skew_{ch_name}"]   = 0.0
                                for q in [0.1, 1, 25, 75, 99, 99.9]:
                                    result[f"intensity_q{q}_{ch_name}"] = 0.0
                            else:
                                result[f"intensity_mean_{ch_name}"]   = float(np.mean(obj_pixels))
                                result[f"intensity_std_{ch_name}"]    = float(np.std(obj_pixels))
                                result[f"intensity_sum_{ch_name}"]    = float(np.sum(obj_pixels))
                                result[f"intensity_skew_{ch_name}"]   = _skewness(obj_pixels)
                                for q in [0.1, 1, 25, 75, 99, 99.9]:
                                    result[f"intensity_q{q}_{ch_name}"] = float(np.percentile(obj_pixels, q))
                    except Exception as e:
                        print(f"Warning: Failed to compute intensity features for object {obj_id}: {e}")

                # ----- extra property features -------------------------------- #
                try:
                    for feat_name in extra_feat_names:
                        result[feat_name] = row_raw[feat_name]
                except Exception as e:
                    print(f"Warning: Failed to compute extra properties for object {obj_id}: {e}")

                results.append(result)
            except Exception as e:
                print(f"Warning: Failed to process object {i}: {e}")
                continue

        return pd.DataFrame(results)
    except Exception as e:
        print(f"Error in _profile_objects: {e}")
        return pd.DataFrame()


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
        Image stack with shape (C, Y, X), where C is the number of channels,
        Y is the height, and X is the width of the image.
    mask_data : dict of str to np.ndarray
        Mask arrays keyed by mask name (without the ``mask_`` prefix).
        Each mask should have shape (Y, X) and contain integer labels for objects.
    metadata_row : dict
        Metadata for this row; image/mask path columns are stripped from the output.
    channel_names : list of str
        Channel names corresponding to image_data channels.
        Must be in the same order as the channels in the image stack.
    mask_name : str
        Key in ``mask_data`` to use as the segmentation mask.
        Must exist in mask_data.
    parent_mask_name : str, optional
        Key in ``mask_data`` to use as the parent mask for hierarchical relationships.
        If provided, must exist in mask_data.
    profile : list of str, optional
        Feature families to compute: ``"shape"``, ``"intensity"``, or both.
        Default: both.
    extra_properties : list of callable, optional
        Factory-built extra-property functions for additional feature computation.
    col_names : dict, optional
        Column-name metadata for renaming extra-property outputs to human-readable names.

    Returns
    -------
    pd.DataFrame or None
        One row per object with metadata columns prepended,
        or None if inputs are invalid / no objects found.
        The DataFrame includes object-level features and metadata information.

    Examples
    --------
    >>> import numpy as np
    >>> image_data = np.random.rand(2, 100, 100)  # 2 channels, 100x100 image
    >>> mask = np.zeros((100, 100), dtype=int)
    >>> mask[25:75, 25:75] = 1  # Single object in the center
    >>> mask_data = {'cell': mask}
    >>> metadata_row = {'row': '01', 'column': '01', 'field': '01', 'DAPI': 'dapi.tiff', 'GFP': 'gfp.tiff'}
    >>> channel_names = ['DAPI', 'GFP']
    >>> result = profile_object_single_row(image_data, mask_data, metadata_row, channel_names, 'cell')
    >>> print('Metadata columns:', [col for col in result.columns if col in metadata_row])
    Metadata columns: ['row', 'column', 'field']
    >>> print('Object columns:', [col for col in result.columns if 'shape' in col or 'intensity' in col][:5])
    Object columns: ['shape_area', 'shape_eccentricity', 'shape_solidity', 'shape_diameter', 'shape_extent']
    >>> print('Number of objects:', len(result))
    Number of objects: 1
    """
    # Validate inputs
    if image_data is None:
        return None
    
    if not isinstance(image_data, np.ndarray):
        raise TypeError(f"image_data must be a numpy array, got {type(image_data).__name__}")
    
    if not isinstance(mask_data, dict):
        raise TypeError(f"mask_data must be a dictionary, got {type(mask_data).__name__}")
    
    if mask_name not in mask_data:
        raise ValueError(f"mask_name '{mask_name}' not found in mask_data")
    
    if not isinstance(metadata_row, dict):
        raise TypeError(f"metadata_row must be a dictionary, got {type(metadata_row).__name__}")
    
    if not isinstance(channel_names, list) or not all(isinstance(name, str) for name in channel_names):
        raise TypeError("channel_names must be a list of strings")
    
    if parent_mask_name is not None and parent_mask_name not in mask_data:
        raise ValueError(f"parent_mask_name '{parent_mask_name}' not found in mask_data")

    try:
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
        
        if parent_mask_name is not None and "parent_id" in result_df.columns:
            result_df = result_df.rename(
                columns={"parent_id": f"parent_{parent_mask_name}_id"}
            )

        return result_df
    except Exception as e:
        print(f"Error in profile_object_single_row: {e}")
        return None
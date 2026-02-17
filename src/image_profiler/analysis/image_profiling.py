"""Image-level profiling for extracting whole-image features."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from skimage.measure import label, regionprops_table


def _skewness(arr: np.ndarray) -> float:
    """Calculate skewness of an array."""
    mean = np.mean(arr)
    std = np.std(arr)
    if std == 0:
        return 0.0
    return float(np.mean(((arr - mean) / std) ** 3))


def _profile_image(
    image: np.ndarray,
    channel_names: List[str],
    channels: Optional[List[str]] = None,
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Profile a single image stack at the whole-image level.

    Parameters
    ----------
    image : np.ndarray
        Image stack with shape (C, Y, X).
    channel_names : list of str
        Channel names corresponding to image channels (C axis).
    channels : list of str, optional
        Subset of channel names to profile. None = all channels.
    thresholds : dict of str to float, optional
        Per-channel intensity thresholds.  When provided for a channel,
        objects are detected via thresholding and area / count metrics
        are added ({ch}_area, {ch}_n_objects, {ch}_mean_object_area).

    Returns
    -------
    dict
        Flat dictionary of measured features keyed by "{channel_name}_{metric}".
    """
    if channels is None:
        channels = channel_names

    if thresholds is None:
        thresholds = {}

    result: Dict[str, Any] = {}

    for ch_name in channels:
        if ch_name not in channel_names:
            continue

        ch_idx = channel_names.index(ch_name)
        img = image[ch_idx]  # (Y, X) — image must be (C, Y, X)
        result[f"intensity_mean_{ch_name}"]   = float(np.mean(img))
        result[f"intensity_sum_{ch_name}"]    = float(np.sum(img))
        # result[f"intensity_std_{ch_name}"]    = float(np.std(img))
        # result[f"intensity_min_{ch_name}"]    = float(np.min(img))
        # result[f"intensity_max_{ch_name}"]    = float(np.max(img))
        # result[f"intensity_median_{ch_name}"] = float(np.median(img))
        # result[f"intensity_skew_{ch_name}"]   = _skewness(img)
        
        for q in [0.1, 1, 25, 75, 99, 99.9]:
            result[f"intensity_q{q}_{ch_name}"] = float(np.percentile(img, q))

        threshold = thresholds.get(ch_name)
        if threshold is not None:
            binary_mask  = img >= threshold
            labeled_mask = label(binary_mask)

            if labeled_mask.max() > 0:
                props = regionprops_table(labeled_mask, properties=["area", "label"])
                result[f"shape_area_{ch_name}"]             = int(np.sum(props["area"]))
                result[f"shape_n_object_{ch_name}"]         = len(props["label"])
                result[f"shape_mean_object_area_{ch_name}"] = float(np.mean(props["area"]))
            else:
                result[f"shape_area_{ch_name}"]             = 0
                result[f"shape_n_object_{ch_name}"]         = 0
                result[f"shape_mean_object_area_{ch_name}"] = 0.0

    return result


def profile_image_single_row(
    image_data: np.ndarray,
    channel_names: List[str],
    metadata_row: Dict,
    channels: Optional[List[str]] = None,
    thresholds: Optional[Dict[str, float]] = None,
) -> Optional[Dict[str, Any]]:
    """Profile a single dataset row at image level.

    Parameters
    ----------
    image_data : np.ndarray
        Image stack with shape (C, Y, X).
    mask_data : dict of str to np.ndarray
        Mask arrays (unused for image-level profiling; kept for API consistency).
    channel_names : list of str
        Channel names corresponding to image channels.
    metadata_row : dict
        Metadata for this row
    channels : list of str, optional
        Subset of channel names to profile.
    thresholds : dict of str to float, optional
        Per-channel intensity thresholds.

    Returns
    -------
    dict or None
        Profiling result merged with metadata, or None if image_data is None.
    """
    if image_data is None:
        return None

    image_result = _profile_image(
        image_data,
        channel_names,
        channels,
        thresholds,
    )

    # Strip image/mask path columns from metadata
    meta = {
        k: v for k, v in metadata_row.items() if not (k in channel_names)
    }

    return {**meta, **image_result}


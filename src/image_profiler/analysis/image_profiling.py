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
        Image stack with shape (C, Y, X), where C is the number of channels,
        Y is the height, and X is the width of the image.
    channel_names : list of str
        Channel names corresponding to image channels (C axis).
        Must be in the same order as the channels in the image stack.
    channels : list of str, optional
        Subset of channel names to profile. None = all channels.
        Must be a subset of channel_names.
    thresholds : dict of str to float, optional
        Per-channel intensity thresholds.  When provided for a channel,
        objects are detected via thresholding and area / count metrics
        are added ({ch}_area, {ch}_n_objects, {ch}_mean_object_area).
        Values should be within the intensity range of the corresponding channel.

    Returns
    -------
    dict
        Flat dictionary of measured features keyed by "{channel_name}_{metric}".
        Includes intensity statistics (mean, sum, percentiles) and optionally
        object statistics (area, count, mean object area) if thresholds are provided.

    Examples
    --------
    >>> import numpy as np
    >>> image = np.random.rand(2, 100, 100)  # 2 channels, 100x100 image
    >>> channel_names = ['DAPI', 'GFP']
    >>> result = _profile_image(image, channel_names)
    >>> print(list(result.keys())[:5])  # Show first 5 keys
    ['intensity_mean_DAPI', 'intensity_sum_DAPI', 'intensity_q0.1_DAPI', 'intensity_q1_DAPI', 'intensity_q25_DAPI']

    >>> # With thresholds
    >>> thresholds = {'DAPI': 0.5, 'GFP': 0.7}
    >>> result_with_thresholds = _profile_image(image, channel_names, thresholds=thresholds)
    >>> print('DAPI area:', result_with_thresholds['shape_area_DAPI'])
    DAPI area: 2500
    """
    # Validate inputs
    if not isinstance(image, np.ndarray):
        raise TypeError(f"image must be a numpy array, got {type(image).__name__}")
    
    if image.ndim != 3:
        raise ValueError(f"image must be 3D with shape (C, Y, X), got shape {image.shape}")
    
    if not isinstance(channel_names, list) or not all(isinstance(name, str) for name in channel_names):
        raise TypeError("channel_names must be a list of strings")
    
    if len(channel_names) != image.shape[0]:
        raise ValueError(f"Number of channel names ({len(channel_names)}) must match number of channels in image ({image.shape[0]})")
    
    if channels is None:
        channels = channel_names
    else:
        if not isinstance(channels, list) or not all(isinstance(name, str) for name in channels):
            raise TypeError("channels must be a list of strings")
        
        for ch_name in channels:
            if ch_name not in channel_names:
                raise ValueError(f"Channel name '{ch_name}' not found in channel_names")
    
    if thresholds is None:
        thresholds = {}
    else:
        if not isinstance(thresholds, dict):
            raise TypeError("thresholds must be a dictionary")
        
        for ch_name, threshold in thresholds.items():
            if ch_name not in channel_names:
                raise ValueError(f"Channel name '{ch_name}' in thresholds not found in channel_names")
            if not isinstance(threshold, (int, float)):
                raise TypeError(f"Threshold for channel '{ch_name}' must be a number, got {type(threshold).__name__}")

    result: Dict[str, Any] = {}

    for ch_name in channels:
        ch_idx = channel_names.index(ch_name)
        img = image[ch_idx]  # (Y, X) — image must be (C, Y, X)
        
        # Skip if image is empty or all zeros
        if img.size == 0 or np.all(img == 0):
            result[f"intensity_mean_{ch_name}"]   = 0.0
            result[f"intensity_sum_{ch_name}"]    = 0.0
            for q in [0.1, 1, 25, 75, 99, 99.9]:
                result[f"intensity_q{q}_{ch_name}"] = 0.0
        else:
            result[f"intensity_mean_{ch_name}"]   = float(np.mean(img))
            result[f"intensity_sum_{ch_name}"]    = float(np.sum(img))
            
            for q in [0.1, 1, 25, 75, 99, 99.9]:
                result[f"intensity_q{q}_{ch_name}"] = float(np.percentile(img, q))

        threshold = thresholds.get(ch_name)
        if threshold is not None:
            try:
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
            except Exception as e:
                # If thresholding fails, set default values
                result[f"shape_area_{ch_name}"]             = 0
                result[f"shape_n_object_{ch_name}"]         = 0
                result[f"shape_mean_object_area_{ch_name}"] = 0.0
                print(f"Warning: Thresholding failed for channel {ch_name}: {e}")

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
        Image stack with shape (C, Y, X), where C is the number of channels,
        Y is the height, and X is the width of the image.
    channel_names : list of str
        Channel names corresponding to image channels (C axis).
        Must be in the same order as the channels in the image stack.
    metadata_row : dict
        Metadata for this row, containing information about the image acquisition.
        Image and mask path columns will be stripped from the returned result.
    channels : list of str, optional
        Subset of channel names to profile. None = all channels.
        Must be a subset of channel_names.
    thresholds : dict of str to float, optional
        Per-channel intensity thresholds.  When provided for a channel,
        objects are detected via thresholding and area / count metrics
        are added ({ch}_area, {ch}_n_objects, {ch}_mean_object_area).
        Values should be within the intensity range of the corresponding channel.

    Returns
    -------
    dict or None
        Profiling result merged with metadata, or None if image_data is None.
        The returned dict contains both the metadata and the computed image features.

    Examples
    --------
    >>> import numpy as np
    >>> image_data = np.random.rand(2, 100, 100)  # 2 channels, 100x100 image
    >>> channel_names = ['DAPI', 'GFP']
    >>> metadata_row = {'row': '01', 'column': '01', 'field': '01', 'DAPI': 'dapi.tiff', 'GFP': 'gfp.tiff'}
    >>> result = profile_image_single_row(image_data, channel_names, metadata_row)
    >>> print('Metadata keys:', [k for k in result.keys() if k in metadata_row])
    Metadata keys: ['row', 'column', 'field']
    >>> print('Feature keys:', [k for k in result.keys() if 'intensity' in k][:3])
    Feature keys: ['intensity_mean_DAPI', 'intensity_sum_DAPI', 'intensity_q0.1_DAPI']
    """
    if image_data is None:
        return None

    # Validate inputs
    if not isinstance(image_data, np.ndarray):
        raise TypeError(f"image_data must be a numpy array, got {type(image_data).__name__}")
    
    if not isinstance(channel_names, list) or not all(isinstance(name, str) for name in channel_names):
        raise TypeError("channel_names must be a list of strings")
    
    if not isinstance(metadata_row, dict):
        raise TypeError(f"metadata_row must be a dictionary, got {type(metadata_row).__name__}")

    try:
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
    except Exception as e:
        print(f"Error profiling image: {e}")
        return None


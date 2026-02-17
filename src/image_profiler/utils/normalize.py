"""Image normalization utilities for image analysis pipeline."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np


def normalize_image(
    image: np.ndarray,
    method: str = "percentile",
    pmin: float = 1.0,
    pmax: float = 99.8,
    dtype: np.dtype = np.uint16
) -> np.ndarray:
    """Normalize image intensity values.
    
    Parameters
    ----------
    image : np.ndarray
        Input image (any shape).
    method : str
        Normalization method:
        - "percentile": Scale to [0, max_dtype] based on percentile range
        - "minmax": Scale to [0, max_dtype] based on min/max
        - "zscore": Z-score normalization (mean=0, std=1)
    pmin, pmax : float
        Percentile range for "percentile" method (default: 1.0, 99.8).
    dtype : np.dtype
        Output data type (default: uint16).
        
    Returns
    -------
    np.ndarray
        Normalized image.
    """
    if method not in ["percentile", "minmax", "zscore"]:
        raise ValueError(f'method must be "percentile", "minmax", or "zscore", got {method}')
    
    image = image.astype(np.float64)
    
    mask = image > 0
    nonzero_values = image[mask]
    
    if len(nonzero_values) == 0:
        return image.astype(dtype)
    
    if method == "percentile":
        p_low = np.percentile(nonzero_values, pmin)
        p_high = np.percentile(nonzero_values, pmax)
        if p_high - p_low > 0:
            normalized = np.clip((image - p_low) / (p_high - p_low), 0, 1)
        else:
            normalized = np.zeros_like(image)
    
    elif method == "minmax":
        min_val = np.min(nonzero_values)
        max_val = np.max(nonzero_values)
        if max_val - min_val > 0:
            normalized = (image - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(image)
    
    elif method == "zscore":
        mean = np.mean(nonzero_values)
        std = np.std(nonzero_values)
        if std > 0:
            normalized = (image - mean) / std
        else:
            normalized = np.zeros_like(image)
    
    if method != "zscore":
        max_val = np.iinfo(dtype).max if np.issubdtype(dtype, np.integer) else 1.0
        normalized = normalized * max_val
    
    return normalized.astype(dtype)


def normalize_imageset(
    images: np.ndarray,
    method: str = "percentile",
    pmin: float = 1.0,
    pmax: float = 99.8,
    channel_index: Optional[int] = None
) -> np.ndarray:
    """Normalize an image stack, optionally by channel.
    
    Parameters
    ----------
    images : np.ndarray
        Image stack with shape (C, Y, X).
    method : str
        Normalization method: "percentile", "minmax", or "zscore".
    pmin, pmax : float
        Percentile range for "percentile" method.
    channel_index : int, optional
        Index of specific channel to normalize. None = normalize all channels.
        
    Returns
    -------
    np.ndarray
        Normalized image stack.
    """
    if method not in ["percentile", "minmax", "zscore", None]:
        raise ValueError(f'method should be one of ["percentile", "minmax", "zscore", None]')
    
    if method is None:
        return images
    
    normalized = images.copy()
    
    if channel_index is not None:
        if channel_index >= images.shape[0] or channel_index < 0:
            raise ValueError(f"channel_index {channel_index} is out of range for image with shape {images.shape}")
        
        channel_img = images[channel_index]
        mask = channel_img > 0
        nonzero_values = channel_img[mask]
        
        if len(nonzero_values) == 0:
            return normalized
        
        if method == "percentile":
            p_low = np.percentile(nonzero_values, pmin)
            p_high = np.percentile(nonzero_values, pmax)
            if p_high - p_low > 0:
                normalized[channel_index] = np.clip((channel_img - p_low) / (p_high - p_low), 0, 1)
        elif method == "minmax":
            min_val = np.min(nonzero_values)
            max_val = np.max(nonzero_values)
            if max_val - min_val > 0:
                normalized[channel_index] = (channel_img - min_val) / (max_val - min_val)
        elif method == "zscore":
            mean = np.mean(nonzero_values)
            std = np.std(nonzero_values)
            if std > 0:
                normalized[channel_index] = (channel_img - mean) / std
    else:
        for i in range(images.shape[0]):
            img = images[i]
            mask = img > 0
            nonzero_values = img[mask]
            
            if len(nonzero_values) == 0:
                continue
            
            if method == "percentile":
                p_low = np.percentile(nonzero_values, pmin)
                p_high = np.percentile(nonzero_values, pmax)
                if p_high - p_low > 0:
                    normalized[i] = np.clip((img - p_low) / (p_high - p_low), 0, 1)
            elif method == "minmax":
                min_val = np.min(nonzero_values)
                max_val = np.max(nonzero_values)
                if max_val - min_val > 0:
                    normalized[i] = (img - min_val) / (max_val - min_val)
            elif method == "zscore":
                mean = np.mean(nonzero_values)
                std = np.std(nonzero_values)
                if std > 0:
                    normalized[i] = (img - mean) / std
    
    return normalized

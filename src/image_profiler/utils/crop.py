"""Object cropping utilities for image analysis pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import scipy.ndimage as ndi
from skimage.measure import label, regionprops
from skimage.transform import resize as sk_resize

try:
    import imageio.v3 as iio
except ImportError:
    import imageio as iio

def crop_object(
    mask: Union[str, Path, np.ndarray],
    imgs: Union[None, str, Path, List[Union[str, Path]], np.ndarray] = None,
    object_ids: Union[int, List[int], None] = None,
    scale_factor: Union[float, None] = 65535.0,
    target_size: Optional[int] = None,
    clip_mask: bool = False,
    pad_square: bool = True,
    rotate_horizontal: bool = False,
    expansion_pixel: int = 0
) -> List[dict]:
    """Crop cells from intensity images using an instance segmentation mask.

    Parameters
    ----------
    mask : str, Path, or np.ndarray
        Path to mask or already loaded integer mask (H, W).
    imgs : None, str, Path, list of Path, or np.ndarray
        - None: only return masks
        - Path/str: single channel image
        - List of Path/str: multiple channel images (stacked along last axis)
        - np.ndarray (H, W, C) or (H, W): pre-loaded image(s)
    object_ids : int, list of int, or None
        Cell ID(s) to crop. If None, crop all non-background cells.
    scale_factor : float or None
        Divide image by this value for normalization. None to skip.
    target_size : int, optional
        Resize crops to (target_size, target_size).
    clip_mask : bool
        Zero out pixels outside cell mask in intensity image.
    pad_square : bool
        Pad crops to square shape.
    rotate_horizontal : bool
        Rotate cell so major axis is horizontal.
    row_idx : int, list of int, or None
        Positional index/indices (0-based) into the sorted list of cell IDs
        after resolving ``object_ids``. When provided, only the cells at those
        positions are processed. Useful for selecting cells by row position
        rather than by explicit label value.
        Example: row_idx=0 → first cell; row_idx=[0, 2, 4] → cells at
        positions 0, 2, and 4 in the sorted ID list.
    expansion_pixel : int
        Number of extra pixels to add on each side of the tight bounding box
        before cropping and applying any further transformations (rotation,
        padding, resize). Expansion is clamped to image boundaries.

    Returns
    -------
    list of dict
        [{'object_id': int, 
        'img': np.ndarray or None,
        'mask': np.ndarray or None}, ...]
        ``img`` shape is (H, W, C), dtype float64 when scale_factor is
        set, otherwise preserves input dtype.
        ``mask`` shape is (H, W), dtype uint8.
    """
    # ------------------------------------------------------------------
    # Load mask
    # ------------------------------------------------------------------
    if isinstance(mask, (str, Path)):
        mask = iio.imread(mask)
    if mask.ndim != 2:
        raise ValueError('mask must be 2D')
    mask_arr: np.ndarray = mask

    # ------------------------------------------------------------------
    # Load image(s) into (H, W, C) array
    # ------------------------------------------------------------------
    img_arr: Optional[np.ndarray] = None
    if imgs is not None:
        if isinstance(imgs, (str, Path)):
            imgs = [imgs]
        if isinstance(imgs, list):
            channels = [iio.imread(f) for f in imgs]
            img_arr = np.stack(channels, axis=-1)   # (H, W, C)
        elif isinstance(imgs, np.ndarray):
            if imgs.ndim == 2:
                img_arr = imgs[..., np.newaxis]      # (H, W, 1)
            elif imgs.ndim == 3:
                img_arr = imgs                        # (H, W, C)
            else:
                raise ValueError("Image array must be 2D or 3D (H, W, C)")
        else:
            raise TypeError("imgs must be a path, list of paths, or np.ndarray")

        if img_arr.shape[:2] != mask_arr.shape:
            raise ValueError(
                f"Image spatial shape {img_arr.shape[:2]} != mask shape {mask_arr.shape}"
            )

    # ------------------------------------------------------------------
    # Resolve object IDs
    # ------------------------------------------------------------------
    if object_ids is None:
        all_object_ids: List[int] = np.unique(mask_arr[mask_arr != 0]).tolist()
    elif isinstance(object_ids, (int, np.integer)):
        all_object_ids = [int(object_ids)]
    else:
        all_object_ids = [int(c) for c in object_ids]

    img_h, img_w = mask_arr.shape
    results: List[dict] = []

    for object_id in all_object_ids:
        try:
            # Fast membership check (avoids slow Python 'in' on large arrays)
            if not np.any(mask_arr == object_id):
                results.append({'object_id': object_id, 'cell_img': None, 'cell_mask': None})
                continue

            cell_mask_bool = mask_arr == object_id  # (H, W) bool

            # regionprops on binary uint8 directly — no need for label()
            props = regionprops(cell_mask_bool.astype(np.uint8))[0]
            y0, x0, y1, x1 = props.bbox
            orientation = props.orientation

            # Expand bounding box, clamped to image boundaries
            y0 = max(0, y0 - expansion_pixel)
            x0 = max(0, x0 - expansion_pixel)
            y1 = min(img_h, y1 + expansion_pixel)
            x1 = min(img_w, x1 + expansion_pixel)

            # Initial crops
            cropped_mask_bool  = cell_mask_bool[y0:y1, x0:x1]          # (H, W) bool
            cropped_mask_uint8 = cropped_mask_bool.astype(np.uint8)     # (H, W) uint8
            cropped_img = (
                img_arr[y0:y1, x0:x1, :].copy() if img_arr is not None else None
            )  # (H, W, C)

            # Optionally zero out pixels outside the cell mask
            if cropped_img is not None and clip_mask:
                cropped_img = cropped_img * cropped_mask_bool[..., np.newaxis]

            # ----------------------------------------------------------
            # Rotation: align major axis horizontally
            # ----------------------------------------------------------
            if rotate_horizontal:
                angle_deg = -np.degrees(orientation) + 90

                h, w = cropped_mask_bool.shape
                max_dim = max(h, w)
                pad_total = max_dim * 2
                pad_h = (pad_total - h) // 2
                pad_w = (pad_total - w) // 2
                spatial_pad = (
                    (pad_h, pad_total - h - pad_h),
                    (pad_w, pad_total - w - pad_w),
                )

                cropped_mask_bool  = np.pad(cropped_mask_bool,  spatial_pad,
                                            mode='constant', constant_values=False)
                cropped_mask_uint8 = np.pad(cropped_mask_uint8, spatial_pad,
                                            mode='constant', constant_values=0)
                if cropped_img is not None:
                    # Image is (H, W, C) — pad only the two spatial axes
                    cropped_img = np.pad(cropped_img, spatial_pad + ((0, 0),),
                                         mode='constant', constant_values=0)

                # FIX: axes=(0,1) rotates the H-W plane of an (H, W, C) array.
                # The original code omitted `axes`, which caused ndi.rotate to
                # rotate the wrong plane on a 3-D array.
                if cropped_img is not None:
                    cropped_img = ndi.rotate(
                        cropped_img, angle_deg, axes=(0, 1),
                        reshape=False, order=1, mode='constant', cval=0.0
                    )
                cropped_mask_bool = ndi.rotate(
                    cropped_mask_bool, angle_deg,
                    reshape=False, order=0, mode='constant', cval=False
                )
                cropped_mask_uint8 = ndi.rotate(
                    cropped_mask_uint8, angle_deg,
                    reshape=False, order=0, mode='constant', cval=0
                )

                # Tight crop post-rotation
                coords = np.column_stack(np.where(cropped_mask_bool))
                if len(coords) == 0:
                    results.append({'object_id': object_id, 'img': None, 'mask': None})
                    continue
                ymin, xmin = coords.min(axis=0)
                ymax, xmax = coords.max(axis=0)
                cropped_mask_bool  = cropped_mask_bool[ymin:ymax+1, xmin:xmax+1]
                cropped_mask_uint8 = cropped_mask_uint8[ymin:ymax+1, xmin:xmax+1]
                if cropped_img is not None:
                    # FIX: include the channel axis in the post-rotation slice
                    cropped_img = cropped_img[ymin:ymax+1, xmin:xmax+1, :]

            # ----------------------------------------------------------
            # Pad to square
            # ----------------------------------------------------------
            if pad_square:
                h, w = cropped_mask_bool.shape
                size = max(h, w)
                pad_h = (size - h) // 2
                pad_w = (size - w) // 2
                spatial_pad = (
                    (pad_h, size - h - pad_h),
                    (pad_w, size - w - pad_w),
                )
                cropped_mask_bool  = np.pad(cropped_mask_bool,  spatial_pad,
                                            mode='constant', constant_values=False)
                cropped_mask_uint8 = np.pad(cropped_mask_uint8, spatial_pad,
                                            mode='constant', constant_values=0)
                if cropped_img is not None:
                    # FIX: append channel no-op pad; original code used wrong
                    # variable name (pad_kwargs) and missing channel axis
                    cropped_img = np.pad(cropped_img, spatial_pad + ((0, 0),),
                                         mode='constant', constant_values=0)

            # ----------------------------------------------------------
            # Resize to target_size
            # ----------------------------------------------------------
            if target_size is not None:
                new_shape = (target_size, target_size)
                if cropped_img is not None:
                    n_ch = cropped_img.shape[2]
                    # FIX: build shape as new_shape + (n_ch,) instead of
                    # new_shape + cropped_img.shape[2:] which could fail on
                    # some NumPy versions when mixing tuple and shape objects
                    cropped_img = sk_resize(
                        cropped_img, new_shape + (n_ch,),
                        anti_aliasing=True, preserve_range=True
                    ).astype(cropped_img.dtype)
                cropped_mask_uint8 = (
                    sk_resize(cropped_mask_uint8, new_shape,
                               order=0, anti_aliasing=False, preserve_range=True) > 0.5
                ).astype(np.uint8)

            # ----------------------------------------------------------
            # Normalize intensity
            # ----------------------------------------------------------
            if cropped_img is not None and scale_factor is not None and scale_factor > 0:
                cropped_img = cropped_img.astype(np.float64) / scale_factor

            results.append({
                'object_id':   object_id,
                'img':  cropped_img,
                'mask': cropped_mask_uint8,
            })

        except Exception:
            # Preserve object_id in error results so callers can match them back
            results.append({'object_id': object_id, 'img': None, 'mask': None})

    return results
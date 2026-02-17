"""
cellprofiler_regionprops.py
============================
Custom region-property functions compatible with ``skimage.measure.regionprops``
and ``regionprops_table`` via the ``extra_properties`` argument.

Three feature families matching CellProfiler:
  * GLCM texture        – MeasureTexture   (13 Haralick features × distances,
                           averaged over 4 angles: 0°, 45°, 90°, 135°)
  * Granularity         – MeasureGranularity (morphological opening spectrum)
  * Radial distribution – MeasureObjectRadialDistribution (rings from centroid)

Design
------
1. **Parameter passing via factory functions**

   ``regionprops`` / ``regionprops_table`` call each extra-property function
   with exactly two positional arguments: ``(image, intensity_image)``.
   There is no mechanism to pass additional arguments directly.

   The solution is **factory functions** (``make_glcm_func``, etc.) that return
   a closure.  All parameters are captured in the closure, so the returned
   function has the correct two-argument signature::

       glcm_fn, glcm_cols = make_glcm_func(n_channels=3, distances=[1,2,3])
       props = regionprops(label_image, intensity_image=img,
                           extra_properties=[glcm_fn])

2. **Custom channel names**

   Pass ``channel_names=["DAPI","GFP","mCherry"]`` to any factory or to
   ``build_extra_properties``.  When omitted, names default to
   ``ch0, ch1, …``::

       make_glcm_func(3, channel_names=["DAPI","GFP","mCherry"])
       make_glcm_func(3)          # → ch0, ch1, ch2

3. **Channel info is always last in feature names**::

       glcm_d1_asm_DAPI
       granularity_scale01_GFP
       radial_bin0_FracAtD_mCherry

4. **skimage multichannel behaviour (important!)**

   When ``intensity_image`` has shape ``(H, W, C)``, skimage calls each
   extra-property function **once per channel** (passing a 2D ``H×W`` slice)
   and stacks the results into a ``(n_features_per_channel, C)`` array.
   ``regionprops_table`` therefore produces column names like
   ``{func_name}-{feat_idx}-{ch_idx}``.
   The helper :func:`rename_regionprops_table` translates these back to the
   human-readable names produced by the factory.

Typical usage from another script
----------------------------------
::

    from cellprofiler_regionprops import build_extra_properties, rename_regionprops_table
    from skimage.measure import regionprops_table
    import pandas as pd

    extra_props, col_names = build_extra_properties(
        n_channels=3,
        channel_names=["DAPI", "GFP", "mCherry"],
        glcm_distances=[1, 2, 3],
        granularity_spectrum_length=16,
        radial_n_bins=4,
    )

    raw = regionprops_table(
        label_image,
        intensity_image=multichannel_img,   # shape (H, W, C)
        properties=["label", "area"],
        extra_properties=extra_props,
    )

    df = rename_regionprops_table(raw, col_names)
    df.head()

Dependencies: numpy, scipy, scikit-image >= 0.19
"""

from __future__ import annotations
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from skimage.feature import graycomatrix
from skimage.morphology import disk, erosion, opening, reconstruction

# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------
_GLCM_LEVELS: int = 256
_DEFAULT_DISTANCES: List[int] = [1, 2, 3]
# 4 angles: 0°, 45°, 90°, 135°
_GLCM_ANGLES: List[float] = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
_HARALICK_STATS: List[str] = [
    "asm", "contrast", "correlation", "variance", "idm",
    "sum_average", "sum_variance", "sum_entropy",
    "entropy", "diff_variance", "diff_entropy", "imc1", "imc2",
]

# col_names dict value structure:
#   {
#     "names":      List[str],   # all names, channels-outer order
#     "n_features": int,         # features per channel
#     "n_channels": int,         # number of channels
#   }
_ColNamesEntry = Dict[str, object]

def _resolve_channel_names(n_channels: int,
                            names: Optional[Sequence[str]]) -> List[str]:
    """Return validated channel name list, falling back to ch0, ch1, …"""
    if names is not None:
        if len(names) != n_channels:
            raise ValueError(
                f"channel_names has {len(names)} entries but n_channels={n_channels}."
            )
        return [str(n) for n in names]
    return [f"ch{i}" for i in range(n_channels)]


def _split_channels(intensity_image: np.ndarray) -> List[np.ndarray]:
    """Return list of 2-D channel arrays from a 2-D or 3-D image."""
    if intensity_image.ndim == 2:
        return [intensity_image]
    return [intensity_image[:, :, c] for c in range(intensity_image.shape[2])]


# ===========================================================================
# Section 1 – GLCM / Texture  (matches CellProfiler MeasureTexture)
# ===========================================================================

def _normalize_channel(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Rescale masked pixels to [0, LEVELS-1].
    Pixels outside the mask get sentinel value LEVELS so they are
    excluded from the co-occurrence matrix automatically.
    """
    out = np.full(img.shape, _GLCM_LEVELS, dtype=np.int32)
    if not mask.any():
        return out
    vals = img[mask].astype(float)
    lo, hi = vals.min(), vals.max()
    scaled = (vals - lo) / (hi - lo) * (_GLCM_LEVELS - 1) if hi > lo else np.zeros_like(vals)
    out[mask] = np.clip(scaled, 0, _GLCM_LEVELS - 1).astype(np.int32)
    return out


def _masked_glcm(img_int: np.ndarray,
                 distances: Sequence[int],
                 angles: Sequence[float]) -> np.ndarray:
    """
    Build normalised GLCM counting only pixel pairs where BOTH pixels are
    inside the mask.  Out-of-mask pixels carry sentinel value LEVELS; a
    (LEVELS+1)×(LEVELS+1) GLCM is built and the sentinel row/column is dropped.

    Returns shape (LEVELS, LEVELS, n_distances, n_angles).
    """
    raw = graycomatrix(
        img_int.astype(np.uint16),
        distances=list(distances),
        angles=list(angles),
        levels=_GLCM_LEVELS + 1,
        symmetric=True,
        normed=False,
    )
    glcm = raw[:_GLCM_LEVELS, :_GLCM_LEVELS, :, :].astype(float)
    totals = glcm.sum(axis=(0, 1), keepdims=True)
    totals[totals == 0] = 1.0
    return glcm / totals


def _haralick_from_glcm(P: np.ndarray) -> np.ndarray:
    """
    Compute 13 Haralick texture statistics from a single normalised
    co-occurrence matrix P of shape (N, N).

    Feature order matches _HARALICK_STATS and CellProfiler's MeasureTexture.
    """
    N = P.shape[0]
    I, J = np.mgrid[0:N, 0:N]
    px, py = P.sum(axis=1), P.sum(axis=0)

    asm      = float(np.sum(P ** 2))
    contrast = float(np.sum((I - J) ** 2 * P))

    mu_i    = float(np.sum(I * P))
    mu_j    = float(np.sum(J * P))
    sigma_i = float(np.sqrt(np.sum(P * (I - mu_i) ** 2)))
    sigma_j = float(np.sqrt(np.sum(P * (J - mu_j) ** 2)))
    correlation = (float(np.sum(P * (I - mu_i) * (J - mu_j)) / (sigma_i * sigma_j))
                   if sigma_i > 0 and sigma_j > 0 else 0.0)

    variance = float(np.sum(P * (I - mu_i) ** 2))
    idm      = float(np.sum(P / (1.0 + (I - J) ** 2)))

    p_flat    = P.ravel()
    p_xplusy  = np.bincount((I + J).ravel(),       weights=p_flat, minlength=2 * N - 1)
    p_xminusy = np.bincount(np.abs(I - J).ravel(), weights=p_flat, minlength=N)
    k_sum     = np.arange(len(p_xplusy))

    sum_average  = float(np.sum(k_sum * p_xplusy))
    nz_s         = p_xplusy > 0
    sum_entropy  = float(-np.sum(p_xplusy[nz_s] * np.log2(p_xplusy[nz_s])))
    # Sum variance uses sum_entropy as reference (per Haralick and CellProfiler)
    sum_variance = float(np.sum((k_sum - sum_entropy) ** 2 * p_xplusy))

    nz_P         = P > 0
    entropy      = float(-np.sum(P[nz_P] * np.log2(P[nz_P])))
    diff_var     = float(np.var(p_xminusy))
    nz_d         = p_xminusy > 0
    diff_entropy = float(-np.sum(p_xminusy[nz_d] * np.log2(p_xminusy[nz_d])))

    nz_px, nz_py = px > 0, py > 0
    hx   = float(-np.sum(px[nz_px] * np.log2(px[nz_px])))
    hy   = float(-np.sum(py[nz_py] * np.log2(py[nz_py])))
    outer    = np.outer(px, py)
    outer_nz = outer > 0
    hxy1 = float(-np.sum(P[outer_nz]     * np.log2(outer[outer_nz])))
    hxy2 = float(-np.sum(outer[outer_nz] * np.log2(outer[outer_nz])))

    denom = max(hx, hy)
    imc1  = (entropy - hxy1) / denom if denom > 0 else 0.0
    imc2  = float(np.sqrt(max(0.0, 1.0 - np.exp(-2.0 * (hxy2 - entropy)))))

    return np.array([asm, contrast, correlation, variance, idm,
                     sum_average, sum_variance, sum_entropy,
                     entropy, diff_var, diff_entropy, imc1, imc2], dtype=float)


def glcm_feature_names(
    channel_names: Sequence[str],
    distances: Optional[Sequence[int]] = None,
) -> List[str]:
    """
    Ordered feature names for GLCM output.

    Pattern: ``glcm_d{dist}_{stat}_{channel}``

    Channel is always last. The list is in **channels-outer** order,
    matching what the factory function's closure returns when called with
    a multi-channel image (all channels in one call).

    Parameters
    ----------
    channel_names : sequence of str
    distances : sequence of int, optional  – default ``[1, 2, 3]``

    Returns
    -------
    list of str, length = n_channels × n_distances × 13
    """
    if distances is None:
        distances = _DEFAULT_DISTANCES
    return [
        f"glcm_{stat}_d{d}_{ch}"
        for ch in channel_names
        for d in distances
        for stat in _HARALICK_STATS
    ]


def make_glcm_func(
    n_channels: int,
    channel_names: Optional[Sequence[str]] = None,
    distances: Optional[Sequence[int]] = None,
) -> Tuple[Callable, _ColNamesEntry]:
    """
    Create a GLCM texture function ready for ``extra_properties``.

    Parameters
    ----------
    n_channels : int
        Number of intensity image channels.
    channel_names : sequence of str, optional
        Custom channel labels.  Defaults to ``ch0, ch1, …``.
    distances : sequence of int, optional
        GLCM pixel distances.  Defaults to ``[1, 2, 3]``.

    Returns
    -------
    func : callable  – pass directly to ``extra_properties``
    col_entry : dict – pass to :func:`rename_regionprops_table` as
                       ``col_names["glcm"]``

    Notes
    -----
    skimage calls this function once per channel (with a 2-D intensity slice)
    and stacks results.  The closure therefore processes a single channel per
    call and returns a 1-D array of length ``n_distances × 13``.

    Example
    -------
    ::

        glcm_fn, glcm_entry = make_glcm_func(3, ["DAPI","GFP","mCherry"])
        props  = regionprops(label_image, intensity_image=img,
                             extra_properties=[glcm_fn])
        # Access by attribute name (same as func.__name__):
        feats  = props[0].glcm    # shape (n_dist*13, n_channels)

        # Or via regionprops_table + rename:
        raw = regionprops_table(label_image, intensity_image=img,
                                extra_properties=[glcm_fn])
        df  = rename_regionprops_table(raw, {"glcm": glcm_entry})
    """
    if distances is None:
        distances = _DEFAULT_DISTANCES
    ch_names = _resolve_channel_names(n_channels, channel_names)
    _distances = list(distances)
    n_feat_per_ch = len(_distances) * len(_HARALICK_STATS)

    def glcm(image: np.ndarray, intensity_image: np.ndarray) -> np.ndarray:
        # skimage passes a single-channel 2-D slice when intensity_image is multichannel.
        # The function also works when called manually with a full multichannel image.
        mask     = image.astype(bool)
        channels = _split_channels(intensity_image)

        if not mask.any():
            return np.zeros(len(channels) * n_feat_per_ch, dtype=float)

        out: List[float] = []
        for ch_img in channels:
            img_int  = _normalize_channel(ch_img, mask)
            glcm_mat = _masked_glcm(img_int, _distances, _GLCM_ANGLES)
            for d_idx in range(len(_distances)):
                P_avg = glcm_mat[:, :, d_idx, :].mean(axis=-1)  # average over 4 angles
                out.extend(_haralick_from_glcm(P_avg).tolist())
        return np.array(out, dtype=float)

    glcm.__name__ = "glcm"

    col_entry: _ColNamesEntry = {
        "names":      glcm_feature_names(ch_names, _distances),
        "n_features": n_feat_per_ch,
        "n_channels": n_channels,
    }
    return glcm, col_entry


# ===========================================================================
# Section 2 – Granularity  (matches CellProfiler MeasureGranularity)
# ===========================================================================

def _granularity_single_channel(
    img: np.ndarray,
    mask: np.ndarray,
    background_radius: int,
    spectrum_length: int,
    subsample_size: float,
) -> np.ndarray:
    """CellProfiler granularity algorithm for a single intensity channel."""
    img = img.astype(float)
    img[~mask] = 0.0

    if 0.0 < subsample_size < 1.0:
        from skimage.transform import rescale
        img  = rescale(img,  subsample_size, anti_aliasing=True,  channel_axis=None)
        mask = rescale(mask.astype(float), subsample_size,
                       anti_aliasing=False, channel_axis=None) > 0.5
        bg_r = max(1, round(background_radius * subsample_size))
    else:
        bg_r = background_radius

    if bg_r > 0:
        img = np.clip(img - opening(img, disk(bg_r)), 0.0, None)
    img[~mask] = 0.0

    image_total = float(img.sum())
    if image_total == 0.0:
        return np.zeros(spectrum_length, dtype=float)

    spectrum   = np.zeros(spectrum_length, dtype=float)
    prev_total = image_total
    current    = img.copy()

    for n in range(1, spectrum_length + 1):
        eroded  = erosion(current, disk(n))
        rebuilt = reconstruction(eroded, current, method="dilation")
        rebuilt[~mask] = 0.0
        curr_total      = float(rebuilt.sum())
        spectrum[n - 1] = 100.0 * (prev_total - curr_total) / image_total
        prev_total      = curr_total
        current         = rebuilt   # iterative: each step works on the rebuilt image

    return spectrum


def granularity_feature_names(
    channel_names: Sequence[str],
    spectrum_length: int = 16,
) -> List[str]:
    """
    Ordered feature names for granularity output.

    Pattern: ``granularity_scale_{n:02d}_{channel}``
    """
    return [
        f"granularity_scale_{n:02d}_{ch}"
        for ch in channel_names
        for n in range(1, spectrum_length + 1)
    ]


def make_granularity_func(
    n_channels: int,
    channel_names: Optional[Sequence[str]] = None,
    background_radius: int = 10,
    spectrum_length: int = 16,
    subsample_size: float = 0.25,
) -> Tuple[Callable, _ColNamesEntry]:
    """
    Create a granularity function ready for ``extra_properties``.

    Parameters
    ----------
    n_channels : int
    channel_names : sequence of str, optional
    background_radius : int
        Disk radius for morphological background subtraction
        (CellProfiler default: 10).
    spectrum_length : int
        Number of granularity scales to compute (CellProfiler default: 16).
    subsample_size : float
        Linear subsampling factor applied before computation for speed.
        CellProfiler default: 0.25.  Use ``1.0`` to disable.

    Returns
    -------
    func : callable
    col_entry : dict
    """
    ch_names = _resolve_channel_names(n_channels, channel_names)
    _bgr, _sl, _ss = background_radius, spectrum_length, subsample_size

    def granularity(image: np.ndarray, intensity_image: np.ndarray) -> np.ndarray:
        mask     = image.astype(bool)
        channels = _split_channels(intensity_image)
        if not mask.any():
            return np.zeros(len(channels) * _sl, dtype=float)
        spectra = [
            _granularity_single_channel(ch.copy(), mask.copy(), _bgr, _sl, _ss)
            for ch in channels
        ]
        return np.concatenate(spectra).astype(float)

    granularity.__name__ = "granularity"

    col_entry: _ColNamesEntry = {
        "names":      granularity_feature_names(ch_names, spectrum_length),
        "n_features": spectrum_length,
        "n_channels": n_channels,
    }
    return granularity, col_entry


# ===========================================================================
# Section 3 – Radial distribution  (matches MeasureObjectRadialDistribution)
# ===========================================================================

def _distance_map_from_centroid(mask: np.ndarray) -> Tuple[np.ndarray, float]:
    rows, cols = np.nonzero(mask)
    r_grid, c_grid = np.mgrid[0:mask.shape[0], 0:mask.shape[1]]
    dist_map = np.sqrt((r_grid - rows.mean()) ** 2 + (c_grid - cols.mean()) ** 2)
    return dist_map, max(float(dist_map[mask].max()), 1.0)


def _radial_single_channel(
    img: np.ndarray,
    mask: np.ndarray,
    norm_dist: np.ndarray,
    n_bins: int,
) -> np.ndarray:
    """
    Compute FracAtD, MeanFrac, RadialCV per radial bin for a single channel.
    """
    img   = img.astype(float)
    total = float(img[mask].sum())
    n_px  = int(mask.sum())
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out   = np.zeros(n_bins * 3, dtype=float)

    for b in range(n_bins):
        lo, hi    = edges[b], edges[b + 1]
        in_hi     = (norm_dist <= hi) if b == n_bins - 1 else (norm_dist < hi)
        ring      = mask & (norm_dist >= lo) & in_hi
        pixels    = img[ring]
        n_ring    = len(pixels)

        frac_at_d = float(pixels.sum()) / total if total > 0 else 0.0
        area_frac = n_ring / n_px if n_px > 0 else 0.0
        mean_frac = frac_at_d / area_frac if area_frac > 0 else 0.0
        mu        = pixels.mean() if n_ring > 0 else 0.0
        radial_cv = (pixels.std() / mu) if (n_ring > 1 and mu > 0) else 0.0

        out[b * 3: b * 3 + 3] = [frac_at_d, mean_frac, radial_cv]

    return out


def radial_feature_names(
    channel_names: Sequence[str],
    n_bins: int = 4,
) -> List[str]:
    """
    Ordered feature names for radial distribution output.

    Pattern: ``radial_{stat}_bin{b}_{channel}``
    where ``stat`` ∈ {frac_at_d, mean_frac, radial_cv}.
    """
    stats = ["frac_at_d", "mean_frac", "radial_cv"]
    return [
        f"radial_{stat}_bin{b}_{ch}"
        for ch in channel_names
        for b in range(n_bins)
        for stat in stats
    ]


def make_radial_func(
    n_channels: int,
    channel_names: Optional[Sequence[str]] = None,
    n_bins: int = 5,
) -> Tuple[Callable, _ColNamesEntry]:
    """
    Create a radial-distribution function ready for ``extra_properties``.

    Parameters
    ----------
    n_channels : int
    channel_names : sequence of str, optional
    n_bins : int
        Number of concentric radial bins (CellProfiler default: 4).

    Returns
    -------
    func : callable
    col_entry : dict
    """
    ch_names  = _resolve_channel_names(n_channels, channel_names)
    _n_bins   = n_bins
    n_feat_pc = n_bins * 3

    def radial_distribution(image: np.ndarray, intensity_image: np.ndarray) -> np.ndarray:
        mask     = image.astype(bool)
        channels = _split_channels(intensity_image)
        if not mask.any():
            return np.zeros(len(channels) * n_feat_pc, dtype=float)
        dist_map, max_dist = _distance_map_from_centroid(mask)
        norm_dist = dist_map / max_dist
        results = [_radial_single_channel(ch, mask, norm_dist, _n_bins) for ch in channels]
        return np.concatenate(results).astype(float)

    radial_distribution.__name__ = "radial_distribution"

    col_entry: _ColNamesEntry = {
        "names":      radial_feature_names(ch_names, n_bins),
        "n_features": n_feat_pc,
        "n_channels": n_channels,
    }
    return radial_distribution, col_entry


# ===========================================================================
# Main public API
# ===========================================================================


# Shorthand string keys accepted by profile_image / profile_object.
_EXTRA_PROPERTIES_KEYS = frozenset({"glcm", "radial", "granularity"})

def build_extra_properties(
    extra_properties: Optional[List[Union[str, Callable]]],
    n_channels: int,
    channel_names: List[str],
    extra_properties_kwargs: Optional[List[Optional[Dict]]] = None,
) -> tuple:
    """Build extra-property callables and their col_names metadata.

    Accepts a mix of shorthand strings (``'glcm'``, ``'radial'``,
    ``'granularity'``) and plain callables.  Strings are expanded into
    factory-built closures so they receive the correct ``n_channels`` and
    ``channel_names`` automatically.

    Parameters
    ----------
    extra_properties : list of str or callable, optional
        Each item is either a shorthand string or a ready-made callable.
    n_channels : int
        Number of image channels (needed by the factory functions).
    channel_names : list of str
        Channel names (needed by the factory functions for column naming).
    extra_properties_kwargs : list of dict or None, optional
        Per-item keyword arguments forwarded to string-based factory functions.
        Ignored for plain callables.  Use ``None`` entries as placeholders.

    Returns
    -------
    callables : list of callable or None
    col_names : dict or None
        Combined column-name metadata from all factory functions;
        ``None`` when no string shortcuts were used.
    """
    if not extra_properties:
        return None, None

    if extra_properties_kwargs is None:
        extra_properties_kwargs = [None] * len(extra_properties)

    callables: List[Callable] = []
    col_names: Dict = {}

    for prop, kwargs in zip(extra_properties, extra_properties_kwargs):
        kw = kwargs or {}

        if isinstance(prop, str):
            if prop not in _EXTRA_PROPERTIES_KEYS:
                raise ValueError(
                    f"Unknown extra property shorthand: '{prop}'. "
                    f"Available: {sorted(_EXTRA_PROPERTIES_KEYS)}"
                )
            if prop == "glcm":
                fn, entry = make_glcm_func(n_channels, channel_names, **kw)
                col_names["glcm"] = entry
            elif prop == "granularity":
                fn, entry = make_granularity_func(n_channels, channel_names, **kw)
                col_names["granularity"] = entry
            elif prop == "radial":
                fn, entry = make_radial_func(n_channels, channel_names, **kw)
                col_names["radial_distribution"] = entry
            callables.append(fn)

        elif callable(prop):
            callables.append(prop)

        else:
            raise TypeError(
                f"extra_properties items must be strings or callables, got {type(prop)}"
            )

    return (callables or None), (col_names or None)


def rename_regionprops_table(
    raw: Dict[str, np.ndarray],
    col_names: Dict[str, _ColNamesEntry],
) -> pd.DataFrame:
    """
    Convert the raw ``regionprops_table`` dict to a tidy DataFrame with
    descriptive column names.

    skimage names extra-property outputs differently depending on whether
    the function returns a scalar, 1-D array, or 2-D array:

    * scalar         → ``{func_name}``
    * 1-D (len N)   → ``{func_name}-0`` … ``{func_name}-{N-1}``
    * 2-D (M × C)  → ``{func_name}-0-0`` … ``{func_name}-{M-1}-{C-1}``

    When ``intensity_image`` is multichannel (H×W×C), skimage calls each
    extra-property function **once per channel** and stacks the results into
    a 2-D array (n_features_per_channel × n_channels), so the 2-D naming
    scheme applies.

    This function handles both cases automatically.

    Parameters
    ----------
    raw : dict
        Direct output of ``skimage.measure.regionprops_table``.
    col_names : dict
        The ``col_names`` dict from :func:`build_extra_properties`, or a
        manually assembled dict of the same structure.

    Returns
    -------
    pd.DataFrame
        One row per region, with human-readable column names.
    """
    import pandas as pd

    rename_map: Dict[str, str] = {}

    for func_name, entry in col_names.items():
        names: List[str]  = entry["names"]       # channels-outer flat list
        n_feat: int       = entry["n_features"]  # features per channel
        n_ch: int         = entry["n_channels"]  # number of channels

        for feat_idx in range(n_feat):
            for ch_idx in range(n_ch):
                # skimage 2-D stacked key: {func}-{feat_idx}-{ch_idx}
                key_2d = f"{func_name}-{feat_idx}-{ch_idx}"
                # Flat list is channels-outer: names[ch_idx * n_feat + feat_idx]
                friendly = names[ch_idx * n_feat + feat_idx]
                if key_2d in raw:
                    rename_map[key_2d] = friendly

            # Also handle 1-D case (single channel or manual call)
            key_1d = f"{func_name}-{feat_idx}"
            if key_1d in raw:
                rename_map[key_1d] = names[feat_idx]

    return pd.DataFrame(raw).rename(columns=rename_map)


# # ===========================================================================
# # Demo / self-test  (python cellprofiler_regionprops.py)
# # ===========================================================================

# if __name__ == "__main__":
#     import pandas as pd
#     from skimage.measure import regionprops, regionprops_table

#     print("=" * 65)
#     print("cellprofiler_regionprops – demo")
#     print("=" * 65)

#     np.random.seed(42)
#     H, W, C = 128, 128, 3
#     CHANNEL_NAMES = ["DAPI", "GFP", "mCherry"]

#     # Synthetic label image: two circular objects
#     label_image = np.zeros((H, W), dtype=np.int32)
#     yy, xx = np.ogrid[:H, :W]
#     label_image[(yy - 32) ** 2 + (xx - 32) ** 2 < 20 ** 2] = 1
#     label_image[(yy - 90) ** 2 + (xx - 90) ** 2 < 25 ** 2] = 2

#     # Synthetic 3-channel image (DAPI has a radial gradient, others are noise)
#     intensity = (np.random.rand(H, W, C) * 0.1).astype(np.float32)
#     intensity[:, :, 0] += np.exp(
#         -np.sqrt((yy - 32) ** 2 + (xx - 32) ** 2) / 15.0).astype(np.float32)

#     # ------------------------------------------------------------------
#     # Demo 1: build_extra_properties → regionprops_table → rename
#     # ------------------------------------------------------------------
#     print("\n[Demo 1]  build_extra_properties  +  regionprops_table  +  rename")
#     print("-" * 65)

#     extra_props, col_names = build_extra_properties(
#         n_channels=C,
#         channel_names=CHANNEL_NAMES,
#         glcm_distances=[1, 2],
#         granularity_spectrum_length=8,
#         granularity_subsample_size=1.0,   # disable subsampling for speed in demo
#         radial_n_bins=3,
#     )

#     raw = regionprops_table(
#         label_image,
#         intensity_image=intensity,
#         properties=["label", "area"],
#         extra_properties=extra_props,
#     )

#     df = rename_regionprops_table(raw, col_names)

#     print(f"  DataFrame shape : {df.shape}")
#     glcm_c = [c for c in df.columns if c.startswith("glcm")]
#     gran_c = [c for c in df.columns if c.startswith("gran")]
#     rad_c  = [c for c in df.columns if c.startswith("radial")]
#     print(f"  GLCM cols       : {glcm_c[:4]} …  (total {len(glcm_c)})")
#     print(f"  Granularity cols: {gran_c[:4]} …  (total {len(gran_c)})")
#     print(f"  Radial cols     : {rad_c[:4]}  …  (total {len(rad_c)})")
#     print(f"\n  Sample rows (first 2 GLCM cols):\n{df[['label','area'] + glcm_c[:2]].to_string(index=False)}")

#     # ------------------------------------------------------------------
#     # Demo 2: individual factories → regionprops (attribute access)
#     # ------------------------------------------------------------------
#     print("\n\n[Demo 2]  Individual make_*_func factories  +  regionprops()")
#     print("-" * 65)

#     glcm_fn,   glcm_entry   = make_glcm_func(C, CHANNEL_NAMES, distances=[1])
#     gran_fn,   gran_entry   = make_granularity_func(C, CHANNEL_NAMES,
#                                                     spectrum_length=4, subsample_size=1.0)
#     radial_fn, radial_entry = make_radial_func(C, CHANNEL_NAMES, n_bins=2)

#     props = regionprops(
#         label_image,
#         intensity_image=intensity,
#         extra_properties=[glcm_fn, gran_fn, radial_fn],
#     )

#     for r in props:
#         print(f"\n  Region {r.label}  (area={r.area} px):")
#         print(f"    r.glcm               shape={r.glcm.shape}")
#         print(f"    r.granularity        shape={r.granularity.shape}")
#         print(f"    r.radial_distribution shape={r.radial_distribution.shape}")

#     print("\n  GLCM col names (first 6)   :", glcm_entry["names"][:6])
#     print("  Granularity col names      :", gran_entry["names"])
#     print("  Radial col names           :", radial_entry["names"])

#     # ------------------------------------------------------------------
#     # Demo 3: default channel names (no channel_names argument)
#     # ------------------------------------------------------------------
#     print("\n\n[Demo 3]  Default channel names (no channel_names arg)")
#     print("-" * 65)
#     _, default_glcm = make_glcm_func(n_channels=C, distances=[1])
#     print("  First 6 GLCM cols:", default_glcm["names"][:6])

#     print("\nAll demos passed.")
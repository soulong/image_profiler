"""
CellProfiler-equivalent feature extraction functions compatible with
skimage.measure.regionprops_table `extra_properties` parameter.

────────────────────────────────────────────────────────────────────
KEY DESIGN: One callable per named output
────────────────────────────────────────────────────────────────────
regionprops_table uses fn.__name__ as the column name for scalars,
or fn.__name__ + "-0", "-1", … for arrays.  To get fully semantic
column names (no numeric suffix), every factory below returns a
*list* of single-scalar callables, one per output dimension.

    fns = make_radial_distribution(nbins=4, channel=0)
    # fns[0].__name__ == "RadialDistribution_bin0_ch0"
    # fns[1].__name__ == "RadialDistribution_bin1_ch0"
    # fns[2].__name__ == "RadialDistribution_bin2_ch0"
    # fns[3].__name__ == "RadialDistribution_bin3_ch0"

    props = regionprops_table(
        label_img, intensity,
        extra_properties=fns,
    )
    # columns: RadialDistribution_bin0_ch0 … RadialDistribution_bin3_ch0

────────────────────────────────────────────────────────────────────
Naming convention  (channel index ALWAYS last)
────────────────────────────────────────────────────────────────────
Feature               Column pattern
──────────────────────────────────────────────────────────────────
Radial distribution   RadialDistribution_bin{i}_ch{c}
                        i=0 → outermost ring, i=N-1 → centre

Granularity           Granularity_scale{s}_ch{c}
                        s is the actual scale value (e.g. 1, 2, …16)

GLCM                  GLCM_{prop}_d{distance}_ch{c}
                        prop ∈ {contrast, dissimilarity, homogeneity,
                                energy, correlation, ASM}
                        distance is the pixel offset (e.g. 1, 2, 4, 8)

Correlation           Correlation_pearson_ch{a}_ch{b}
  (standalone)          returned as a plain dict / DataFrame column
────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import functools
from typing import Sequence

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.feature import graycomatrix, graycoprops
from skimage.morphology import disk, erosion, dilation
from skimage.transform import resize

# ════════════════════════════════════════════════════════════════════════════
# Internal helper
# ════════════════════════════════════════════════════════════════════════════

def _named(fn, name: str):
    """Attach a __name__ to a callable so regionprops_table uses it as column name."""
    fn.__name__ = name
    fn.__qualname__ = name
    return fn


# ════════════════════════════════════════════════════════════════════════════
# 1. RADIAL DISTRIBUTION  (CellProfiler: MeasureObjectIntensityDistribution)
# ════════════════════════════════════════════════════════════════════════════

def _radial_distribution_all(
    regionmask: np.ndarray,
    intensity: np.ndarray,
    *,
    nbins: int,
    channel: int,
) -> np.ndarray:
    """
    Compute the full radial distribution vector (internal, returns array).

    Returns (nbins,) float array:
      index 0 = outermost ring (edge), index nbins-1 = centre.
      Values are fraction of total object intensity in each shell.

    Algorithm (CellProfiler MeasureObjectIntensityDistribution – FracAtD):
      1. distance_transform_edt on the binary mask gives each pixel its
         distance from the nearest background pixel.
      2. Normalise distances to [0, 1]: 0 = object edge, 1 = farthest interior.
      3. Divide [0, 1] into nbins equal-width shells; accumulate intensity.
      4. Divide each shell total by the object's total intensity.
      5. Reverse so bin 0 = outermost shell (CellProfiler convention).
    """
    img = intensity[..., channel] if intensity.ndim == 3 else intensity
    img = img.astype(float)
    mask = regionmask.astype(bool)

    if not mask.any():
        return np.zeros(nbins)

    dist = distance_transform_edt(mask)
    max_dist = dist[mask].max()
    if max_dist == 0:
        return np.zeros(nbins)

    norm_dist = dist[mask] / (max_dist + 1e-9)          # 0=edge … 1=centre
    bin_idx = np.clip(np.floor(norm_dist * nbins).astype(int), 0, nbins - 1)

    total = img[mask].sum()
    fracs = np.zeros(nbins)
    if total == 0:
        return fracs
    for b in range(nbins):
        fracs[b] = img[mask][bin_idx == b].sum() / total

    return fracs[::-1]   # reverse: index 0 = outermost ring


def make_radial_distribution(
    nbins: int = 4,
    channel: int = 0,
) -> list:
    """
    Return a list of `nbins` scalar callables for regionprops_table.

    Each callable returns one float (the intensity fraction for that shell).

    Column names
    ------------
    RadialDistribution_bin{i}_ch{c}
      i=0  → outermost ring (CellProfiler "bin 1")
      i=N-1 → centre        (CellProfiler "bin N")

    Usage
    -----
    fns = make_radial_distribution(nbins=4, channel=0)
    props = regionprops_table(labels, img, extra_properties=fns)
    # columns: RadialDistribution_bin0_ch0  (outermost)
    #          RadialDistribution_bin1_ch0
    #          RadialDistribution_bin2_ch0
    #          RadialDistribution_bin3_ch0  (centre)
    """
    fns = []
    for b in range(nbins):
        def _fn(mask, intensity, _b=b, _nbins=nbins, _ch=channel):
            return float(
                _radial_distribution_all(mask, intensity, nbins=_nbins, channel=_ch)[_b]
            )
        fns.append(_named(_fn, f"RadialDistribution_bin{b}_ch{channel}"))
    return fns


# ════════════════════════════════════════════════════════════════════════════
# 2. GRANULARITY  (CellProfiler: MeasureGranularity)
# ════════════════════════════════════════════════════════════════════════════

def _granularity_all(
    regionmask: np.ndarray,
    intensity: np.ndarray,
    *,
    scales: tuple[int, ...],
    channel: int,
    subsample_size: int,
    element_size: int,
) -> np.ndarray:
    """
    Compute the full granularity spectrum (internal, returns array).

    Algorithm (CellProfiler MeasureGranularity):
      1. Mask the image to the object region (zero outside).
      2. Optionally downsample so longest side ≤ subsample_size (speed).
      3. Initialise current = masked image; record prev_mean.
      4. For each scale S:
           radius = round(S * element_size / 10)   [element_size=10 → radius=S]
           opened = morphological opening with disk(radius)
           GS[i]  = (prev_mean - curr_mean) / prev_mean
           current = opened;  prev_mean = curr_mean
      The spectrum captures what fraction of texture is removed at each scale.
    """

    img = intensity[..., channel] if intensity.ndim == 3 else intensity
    img = img.astype(float)
    mask = regionmask.astype(bool)
    masked_img = img * mask

    h, w = masked_img.shape[:2]
    if max(h, w) > subsample_size:
        factor = subsample_size / max(h, w)
        nh, nw = max(1, int(round(h * factor))), max(1, int(round(w * factor)))
        masked_img = resize(masked_img, (nh, nw), anti_aliasing=True, order=3)
        mask = resize(mask.astype(float), (nh, nw), order=0) > 0.5

    current = masked_img.copy()
    prev_mean = current[mask].mean() if mask.any() else 0.0
    result = np.zeros(len(scales))
    if prev_mean == 0:
        return result

    for i, scale in enumerate(scales):
        radius = max(1, int(round(scale * element_size / 10)))
        se = disk(radius)
        opened = dilation(erosion(current, se), se)
        curr_mean = opened[mask].mean() if mask.any() else 0.0
        result[i] = (prev_mean - curr_mean) / prev_mean if prev_mean > 0 else 0.0
        current = opened
        prev_mean = curr_mean

    return result


def make_granularity(
    scales: Sequence[int] = tuple(range(1, 17)),
    channel: int = 0,
    subsample_size: int = 256,
    element_size: int = 10,
) -> list:
    """
    Return a list of len(scales) scalar callables for regionprops_table.

    Each callable returns one float (the granularity value at that scale).

    Column names
    ------------
    Granularity_scale{s}_ch{c}
      s is the actual scale integer (e.g. 1, 2, … 16), not an array index.

    Usage
    -----
    fns = make_granularity(scales=range(1, 17), channel=0)
    props = regionprops_table(labels, img, extra_properties=fns)
    # columns: Granularity_scale1_ch0, Granularity_scale2_ch0, …,
    #          Granularity_scale16_ch0

    Multiple channels
    -----------------
    fns = make_granularity(channel=0) + make_granularity(channel=1)
    # columns for both channels in one call
    """
    scales = tuple(scales)
    fns = []
    for i, s in enumerate(scales):
        def _fn(mask, intensity, _i=i, _s=scales, _ch=channel,
                _sub=subsample_size, _el=element_size):
            return float(
                _granularity_all(mask, intensity,
                                 scales=_s, channel=_ch,
                                 subsample_size=_sub,
                                 element_size=_el)[_i]
            )
        fns.append(_named(_fn, f"Granularity_scale{s}_ch{channel}"))
    return fns


# ════════════════════════════════════════════════════════════════════════════
# 3. GLCM (Gray-Level Co-occurrence Matrix)
#    CellProfiler: MeasureTexture
# ════════════════════════════════════════════════════════════════════════════

# Supported GLCM property names. ASM = energy² matches CellProfiler's
# AngularSecondMoment output. All others use skimage graycoprops names directly.
GLCM_PROPS = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM")


def _glcm_all(
    regionmask: np.ndarray,
    intensity: np.ndarray,
    *,
    distances: tuple[int, ...],
    angles: tuple[float, ...],
    levels: int,
    channel: int,
    props: tuple[str, ...],
) -> np.ndarray:
    """
    Compute all GLCM features (internal, returns flat array).

    Algorithm (CellProfiler MeasureTexture):
      1. Extract masked pixels; quantise to [0, levels-1] using min-max scaling.
      2. Build symmetric, normalised GLCM for each distance at all angles.
      3. For each (distance, prop): compute mean across the angle axis.
      4. ASM = energy² (CellProfiler AngularSecondMoment convention).

    Output layout (flat): [d0_p0, d0_p1, …, d0_pM, d1_p0, …, dN_pM]
    """
    img = intensity[..., channel] if intensity.ndim == 3 else intensity
    img = img.astype(float)
    mask = regionmask.astype(bool)

    roi = img[mask]
    n_out = len(distances) * len(props)
    if roi.size == 0:
        return np.zeros(n_out)

    img_min, img_max = roi.min(), roi.max()
    if img_max == img_min:
        return np.zeros(n_out)

    quantised = np.zeros_like(img, dtype=np.uint8)
    quantised[mask] = np.clip(
        ((img[mask] - img_min) / (img_max - img_min) * (levels - 1)).astype(int),
        0, levels - 1,
    )

    results = []
    for d in distances:
        glcm = graycomatrix(
            quantised,
            distances=[d],
            angles=list(angles),
            levels=levels,
            symmetric=True,
            normed=True,
        )
        for p in props:
            if p == "ASM":
                vals = graycoprops(glcm, "energy")[0] ** 2
            else:
                vals = graycoprops(glcm, p)[0]
            results.append(float(vals.mean()))   # mean over angles

    return np.array(results)


def make_glcm(
    distances: Sequence[int] = (1, 2, 4, 8),
    angles: Sequence[float] = (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
    levels: int = 8,
    channel: int = 0,
    props: Sequence[str] = GLCM_PROPS,
) -> list:
    """
    Return a list of len(distances)*len(props) scalar callables.

    Each callable returns one float (one GLCM feature at one distance).

    Column names
    ------------
    GLCM_{prop}_d{distance}_ch{c}
      prop     ∈ {contrast, dissimilarity, homogeneity, energy, correlation, ASM}
      distance = actual pixel offset integer (e.g. 1, 2, 4, 8)

    Usage
    -----
    fns = make_glcm(distances=[1, 4], channel=0)
    props = regionprops_table(labels, img, extra_properties=fns)
    # columns (12 total):
    #   GLCM_contrast_d1_ch0      GLCM_contrast_d4_ch0
    #   GLCM_dissimilarity_d1_ch0 GLCM_dissimilarity_d4_ch0
    #   GLCM_homogeneity_d1_ch0   GLCM_homogeneity_d4_ch0
    #   GLCM_energy_d1_ch0        GLCM_energy_d4_ch0
    #   GLCM_correlation_d1_ch0   GLCM_correlation_d4_ch0
    #   GLCM_ASM_d1_ch0           GLCM_ASM_d4_ch0

    Selecting a property subset
    ----------------------------
    fns = make_glcm(distances=[1], props=["contrast", "correlation"], channel=1)
    # → GLCM_contrast_d1_ch1, GLCM_correlation_d1_ch1
    """
    distances = tuple(distances)
    angles = tuple(angles)
    props = tuple(props)

    fns = []
    for di, d in enumerate(distances):
        for pi, p in enumerate(props):
            flat_idx = di * len(props) + pi

            def _fn(mask, intensity,
                    _idx=flat_idx, _dists=distances, _angles=angles,
                    _levels=levels, _ch=channel, _props=props):
                return float(
                    _glcm_all(mask, intensity,
                               distances=_dists, angles=_angles,
                               levels=_levels, channel=_ch,
                               props=_props)[_idx]
                )
            fns.append(_named(_fn, f"GLCM_{p}_d{d}_ch{channel}"))
    return fns


# ════════════════════════════════════════════════════════════════════════════
# 4. MULTI-CHANNEL PEARSON CORRELATION
#    CellProfiler: MeasureCorrelation
#
#    NOT passed as extra_properties — requires two channels simultaneously,
#    which conflicts with regionprops_table's single intensity-image contract.
#    Call this function directly alongside regionprops_table and merge results.
# ════════════════════════════════════════════════════════════════════════════

def measure_channel_correlation(
    label_image: np.ndarray,
    multichannel_image: np.ndarray,
    channel_pairs: Sequence[tuple[int, int]] | None = None,
) -> dict[str, np.ndarray]:
    """
    Pearson correlation between channel pairs, per labeled object.
    Replicates CellProfiler MeasureCorrelation (Pearson metric only).

    Parameters
    ----------
    label_image        : (H, W) int array, 0 = background.
    multichannel_image : (H, W, C) float array with C >= 2.
                         For separate channel images, stack first:
                         np.stack([img_ch0, img_ch1], axis=-1)
    channel_pairs      : list of (a, b) tuples to correlate.
                         Default: all unique unordered pairs (a < b).

    Returns
    -------
    dict with keys:
        "label"                           : (N,) object labels
        "Correlation_pearson_ch{a}_ch{b}" : (N,) Pearson r, NaN if
                                            either channel has zero variance

    Usage
    -----
    corr = measure_channel_correlation(label_img, img)
    df_corr = pd.DataFrame(corr)

    # Merge with regionprops_table output on "label":
    df_props = pd.DataFrame(regionprops_table(..., properties=["label", ...]))
    df = df_props.merge(df_corr, on="label")

    Notes
    -----
    - CellProfiler computes Pearson on raw pixel intensities inside the mask
      with no prior normalisation. This implementation matches that behaviour.
    - Why not extra_properties?
      regionprops_table passes a single intensity image to each callable.
      Pearson correlation requires two channels simultaneously, so it cannot
      fit the (regionmask, intensity) -> scalar signature without external
      state. The standalone function + merge pattern is cleaner and explicit.
    """
    if multichannel_image.ndim != 3:
        raise ValueError(
            "multichannel_image must be (H, W, C). "
            "Stack channels: np.stack([ch0, ch1], axis=-1)."
        )
    n_channels = multichannel_image.shape[2]

    if channel_pairs is None:
        channel_pairs = [
            (a, b) for a in range(n_channels) for b in range(a + 1, n_channels)
        ]

    labels = np.unique(label_image)
    labels = labels[labels != 0]
    n_objects = len(labels)

    result: dict[str, np.ndarray] = {"label": labels}

    for a, b in channel_pairs:
        ch_a = multichannel_image[..., a].astype(float)
        ch_b = multichannel_image[..., b].astype(float)
        pearson = np.full(n_objects, np.nan)

        for i, lbl in enumerate(labels):
            mask = label_image == lbl
            va, vb = ch_a[mask], ch_b[mask]
            if va.std() > 0 and vb.std() > 0:
                pearson[i] = float(np.corrcoef(va, vb)[0, 1])

        result[f"Correlation_pearson_ch{a}_ch{b}"] = pearson

    return result


# ════════════════════════════════════════════════════════════════════════════
# EFFICIENCY NOTE: avoiding redundant recomputation
# ════════════════════════════════════════════════════════════════════════════
"""
Each scalar callable in a make_* list recomputes the full underlying array
(e.g. all GLCM features for all distances) every time it is called for one
object. regionprops_table calls each extra_property separately, so for N
callables per object this triggers N full recomputations.

For typical cell biology datasets (cells < 200 px diameter, < 1000 objects)
the overhead is negligible. For larger datasets, wrap with a per-call cache:

    _cache: dict = {}

    def _cached_glcm(mask, intensity, *, key, **kw):
        obj_id = id(mask)   # regionmask is a new array per object
        # Better: hash mask bytes
        h = hash(mask.tobytes())
        if h not in _cache:
            _cache[h] = _glcm_all(mask, intensity, **kw)
        return _cache[h]

    # Clear between regionprops_table calls:
    _cache.clear()
    props = regionprops_table(...)
"""


# ════════════════════════════════════════════════════════════════════════════
# USAGE EXAMPLE
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import pandas as pd
    from skimage.measure import regionprops_table
    from skimage.draw import disk as draw_disk

    rng = np.random.default_rng(42)
    H, W, C = 128, 128, 3
    img = rng.random((H, W, C)).astype(np.float32)

    label_img = np.zeros((H, W), dtype=int)
    for obj_id, (cy, cx, r) in enumerate([(32, 32, 18), (90, 80, 22)], start=1):
        rr, cc = draw_disk((cy, cx), r, shape=(H, W))
        label_img[rr, cc] = obj_id

    # ── 1. Radial distribution ────────────────────────────────────────────
    # 4 bins, channel 0
    rad_fns = make_radial_distribution(nbins=4, channel=0)
    # → RadialDistribution_bin0_ch0 … RadialDistribution_bin3_ch0

    # ── 2. Granularity ────────────────────────────────────────────────────
    # scales 1–8, channels 0 and 1 together
    gran_fns = make_granularity(scales=range(1, 9), channel=0) \
             + make_granularity(scales=range(1, 9), channel=1)
    # → Granularity_scale1_ch0 … scale8_ch0,
    #   Granularity_scale1_ch1 … scale8_ch1

    # ── 3. GLCM ───────────────────────────────────────────────────────────
    # All 6 props at distances 1 & 4, channel 0
    glcm_fns = make_glcm(distances=[1, 4], channel=0)
    # → GLCM_contrast_d1_ch0, …, GLCM_ASM_d1_ch0,
    #   GLCM_contrast_d4_ch0, …, GLCM_ASM_d4_ch0

    # Subset of props, channel 1
    glcm_fns_ch1 = make_glcm(distances=[1], props=["contrast", "correlation"], channel=1)
    # → GLCM_contrast_d1_ch1, GLCM_correlation_d1_ch1

    # ── Assemble and run ──────────────────────────────────────────────────
    extra = rad_fns + gran_fns + glcm_fns + glcm_fns_ch1

    props = regionprops_table(
        label_img,
        img,
        properties=["label", "area"],
        extra_properties=extra,
    )
    df = pd.DataFrame(props)
    print("\n=== regionprops_table columns ===")
    for col in df.columns:
        print(" ", col)

    # ── 4. Pearson correlation (separate call, then merge) ────────────────
    corr = measure_channel_correlation(
        label_img,
        img,
        channel_pairs=[(0, 1), (0, 2), (1, 2)],
    )
    df_corr = pd.DataFrame(corr)
    # → columns: label, Correlation_pearson_ch0_ch1,
    #             Correlation_pearson_ch0_ch2, Correlation_pearson_ch1_ch2

    df_full = df.merge(df_corr, on="label")
    print("\n=== All columns after merge ===")
    for col in df_full.columns:
        print(" ", col)
    print()
    print(df_full.T.to_string())
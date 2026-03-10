"""
measure_objects.py
==================
High-level orchestrator that measures morphology, intensity, and texture
features for every labeled object in a multichannel image.

Depends on:  cellprofiler_regionprops_extras.py  (must be on sys.path)

────────────────────────────────────────────────────────────────────
Why one regionprops_table call per channel?
────────────────────────────────────────────────────────────────────
When regionprops_table receives a multichannel (H, W, C) intensity image
it calls every extra_property function C times — once per channel —
passing a 2-D (H, W) slice each time.  This causes every scalar fn to
produce C columns suffixed -0, -1, …, -(C-1) instead of a single clean
column.

The solution: run regionprops_table once with the full image for shape
descriptors only (no extra_properties), then once *per channel* with
the 2-D channel slice for all intensity / texture extras on that channel.
All partial DataFrames are merged on "label".

Correlation is computed separately (requires two channels simultaneously)
and merged in at the end.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from skimage.measure import regionprops_table
from scipy.ndimage import find_objects

from image_profiler.analysis.extra_properties import (
    make_radial_distribution,
    make_granularity,
    make_glcm,
    measure_channel_correlation,
)


# ── Shape properties forwarded verbatim to regionprops_table ──────────────────
_SHAPE_PROPERTIES: tuple[str, ...] = (
    "label",
    "area",
    # "bbox",
    # "centroid",
    "eccentricity",
    "equivalent_diameter_area",
    # "euler_number",
    "extent",
    "feret_diameter_max",
    "major_axis_length",
    "minor_axis_length",
    # "orientation",
    "perimeter",
    "solidity",
)

_SHAPE_COLUMN_RENAMES: dict[str, str] = {
    "centroid-0": "centroid_y",
    "centroid-1": "centroid_x",
    "bbox-0":     "bbox_min_y",
    "bbox-1":     "bbox_min_x",
    "bbox-2":     "bbox_max_y",
    "bbox-3":     "bbox_max_x",
}


# ══════════════════════════════════════════════════════════════════════════════
# relate_mask  (CellProfiler: RelateObjects)
# ══════════════════════════════════════════════════════════════════════════════

def relate_mask(
    child_mask: np.ndarray,
    parent_mask: np.ndarray,
) -> dict[int, int]:
    """
    Assign every child object to the parent object that contains the majority
    of its pixels.  Replicates CellProfiler's RelateObjects module.

    CellProfiler algorithm
    ----------------------
    For each child object label C:
      1. Extract all pixels belonging to C in child_mask.
      2. Look up the parent label values at exactly those pixel coordinates
         in parent_mask.
      3. The parent with the **most overlapping pixels** (plurality vote) is
         assigned as the parent.  Ties broken by lowest label index.
      4. If no parent pixel overlaps (child lies entirely in background of
         parent_mask), the child is assigned parent label 0 (unrelated).

    This is equivalent to the CellProfiler behaviour of using pixel-level
    overlap rather than centroid containment, making it robust to non-convex
    or irregularly shaped parents.

    Parameters
    ----------
    child_mask  : (Y, X) int ndarray — labeled child objects (0 = background).
    parent_mask : (Y, X) int ndarray — labeled parent objects (0 = background).
                  Must be the same spatial shape as child_mask.

    Returns
    -------
    dict[int, int]
        Mapping of {child_label: parent_label}.
        parent_label == 0 means the child has no parent (lies in background).

    Raises
    ------
    ValueError
        If child_mask and parent_mask have different shapes.

    Notes
    -----
    - Children that span multiple parents are assigned to the dominant one
      (the parent covering the majority of the child's pixels).
    - The result dict contains an entry for every non-zero label in
      child_mask, even if unrelated (parent = 0).
    - To produce a "Parent_{parent_mask_name}" column in measure_objects,
      this function is called internally and the result is merged on label.

    Example
    -------
    parent_map = relate_mask(nuclei_mask, cell_mask)
    # parent_map = {1: 3, 2: 3, 3: 5, 4: 0, ...}
    #   nucleus 1 belongs to cell 3, nucleus 4 has no parent cell
    """
    if child_mask.shape != parent_mask.shape:
        raise ValueError(
            f"child_mask shape {child_mask.shape} != "
            f"parent_mask shape {parent_mask.shape}"
        )

    child_labels = np.unique(child_mask)
    child_labels = child_labels[child_labels != 0]

    parent_map: dict[int, int] = {}

    # Use scipy.ndimage.find_objects to locate each child's bounding box
    # for efficient sub-image extraction instead of scanning the full array.
    slices = find_objects(child_mask)   # index i → slice for label i+1

    for child_lbl in child_labels:
        sl = slices[child_lbl - 1]     # bounding-box slice tuple
        if sl is None:
            parent_map[int(child_lbl)] = 0
            continue

        child_roi  = child_mask[sl]
        parent_roi = parent_mask[sl]

        # Pixels that belong to this child in its bounding box
        child_pixels = child_roi == child_lbl

        # Parent labels under those pixels
        parent_values = parent_roi[child_pixels]

        # Remove background (0) before voting
        parent_values = parent_values[parent_values != 0]

        if parent_values.size == 0:
            # Child lies entirely outside any parent region
            parent_map[int(child_lbl)] = 0
        else:
            # Plurality vote: label with the highest pixel count wins
            unique_parents, counts = np.unique(parent_values, return_counts=True)
            parent_map[int(child_lbl)] = int(unique_parents[np.argmax(counts)])

    return parent_map


# ══════════════════════════════════════════════════════════════════════════════
# Boundary detection
# ══════════════════════════════════════════════════════════════════════════════

def _is_boundary_object(
    child_mask: np.ndarray,
    boundary_fraction_threshold: float = 0.25,
) -> dict[int, bool]:
    """
    Flag objects whose pixels touching the image border exceed a threshold
    fraction of their total pixel count.

    CellProfiler defines a boundary object as one where > 25 % of its pixels
    lie on the image edge rows/columns (row 0, row H-1, col 0, col W-1).

    Parameters
    ----------
    child_mask                  : (Y, X) labeled int array.
    boundary_fraction_threshold : fraction above which an object is flagged
                                  (default 0.25, matching CellProfiler).

    Returns
    -------
    dict[int, bool]  — {label: is_boundary}
    """
    H, W = child_mask.shape

    # Build a boolean border-pixel mask: True for the outermost ring of pixels
    border_mask = np.zeros((H, W), dtype=bool)
    border_mask[0, :]  = True   # top row
    border_mask[-1, :] = True   # bottom row
    border_mask[:, 0]  = True   # left column
    border_mask[:, -1] = True   # right column

    labels = np.unique(child_mask)
    labels = labels[labels != 0]

    result: dict[int, bool] = {}
    slices = find_objects(child_mask)

    for lbl in labels:
        sl = slices[lbl - 1]
        if sl is None:
            result[int(lbl)] = False
            continue

        obj_pixels   = child_mask[sl] == lbl
        border_pixels = border_mask[sl]

        total          = int(obj_pixels.sum())
        on_border      = int((obj_pixels & border_pixels).sum())
        fraction       = on_border / total if total > 0 else 0.0

        result[int(lbl)] = fraction > boundary_fraction_threshold

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Internal helpers shared by measure_objects
# ══════════════════════════════════════════════════════════════════════════════

def _named(fn, name: str):
    fn.__name__ = name
    fn.__qualname__ = name
    return fn


def _resolve_channels(
    requested: Sequence[str] | None,
    channel_names: list[str],
    param_name: str,
) -> list[int]:
    """Validate and convert channel name strings to integer indices."""
    if not requested:
        return []
    unknown = [c for c in requested if c not in channel_names]
    if unknown:
        raise ValueError(
            f"'{param_name}' contains unknown channel name(s): {unknown}.\n"
            f"Available channels: {channel_names}"
        )
    return [channel_names.index(c) for c in requested]


def _build_intensity_fns_for_channel(ch_name: str) -> list:
    """
    Four scalar extra_properties callables for a single 2-D channel slice.

    Column names:
        Intensity_mean_{ch_name}, Intensity_median_{ch_name},
        Intensity_std_{ch_name},  Intensity_sum_{ch_name}
    """
    def _mean(mask, intensity):
        p = intensity[mask.astype(bool)]
        return float(p.mean()) if p.size > 0 else 0.0

    def _median(mask, intensity):
        p = intensity[mask.astype(bool)]
        return float(np.median(p)) if p.size > 0 else 0.0

    def _std(mask, intensity):
        p = intensity[mask.astype(bool)]
        return float(p.std()) if p.size > 0 else 0.0

    def _sum(mask, intensity):
        p = intensity[mask.astype(bool)]
        return float(p.sum()) if p.size > 0 else 0.0

    return [
        _named(_mean,   f"Intensity_mean_{ch_name}"),
        _named(_median, f"Intensity_median_{ch_name}"),
        _named(_std,    f"Intensity_std_{ch_name}"),
        _named(_sum,    f"Intensity_sum_{ch_name}"),
    ]


def _rename_channel_index_to_name(fns: list, ch_idx: int, ch_name: str) -> list:
    """Replace '_ch{idx}' suffix in fn.__name__ with the human-readable name."""
    for fn in fns:
        new_name = fn.__name__.replace(f"_ch{ch_idx}", f"_{ch_name}")
        fn.__name__ = new_name
        fn.__qualname__ = new_name
    return fns


def _run_per_channel_regionprops(
    mask: np.ndarray,
    img: np.ndarray,
    channel_names: list[str],
    channel_groups: dict[int, list],
) -> pd.DataFrame:
    """Run regionprops_table once per channel (2-D slice) and merge on label."""
    dfs: list[pd.DataFrame] = []
    for ch_idx, fns in channel_groups.items():
        if not fns:
            continue
        props = regionprops_table(
            mask,
            img[..., ch_idx],
            properties=["label"],
            extra_properties=fns,
        )
        dfs.append(pd.DataFrame(props))

    if not dfs:
        return pd.DataFrame()

    result = dfs[0]
    for df in dfs[1:]:
        result = result.merge(df, on="label", how="outer")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def measure_objects(
    mask: np.ndarray,
    img: np.ndarray,
    channel_names: list[str],
    metadata_row: dict[str, Any] | None = None,
    # ── parent relationship ────────────────────────────────────────────────────
    parent_mask: np.ndarray | None = None,
    parent_mask_name: str = "Parent",
    # ── intensity ─────────────────────────────────────────────────────────────
    intensity_channels: Sequence[str] | None = None,
    # ── radial distribution ───────────────────────────────────────────────────
    radial_distribution_channels: Sequence[str] | None = None,
    radial_distribution_kwargs: dict[str, Any] | None = None,
    # ── granularity ───────────────────────────────────────────────────────────
    granularity_channels: Sequence[str] | None = None,
    granularity_kwargs: dict[str, Any] | None = None,
    # ── GLCM ──────────────────────────────────────────────────────────────────
    glcm_channels: Sequence[str] | None = None,
    glcm_kwargs: dict[str, Any] | None = None,
    # ── Pearson correlation ────────────────────────────────────────────────────
    correlation_pairs: Sequence[tuple[str, str]] | None = None,
    # ── boundary detection ────────────────────────────────────────────────────
    boundary_fraction_threshold: float = 0.25,
) -> pd.DataFrame:
    """
    Measure shape, intensity, and texture features for every labeled object.

    Parameters
    ----------
    mask : (Y, X) int ndarray
        Labeled segmentation mask.  0 = background; each unique positive
        integer identifies a distinct object.

    img : (Y, X, C) float ndarray
        Multichannel intensity image.  The C axis must align with
        ``channel_names`` (same order).

    channel_names : list[str]
        Human-readable name for each channel.  Length must equal
        ``img.shape[2]``.  Names are used as column suffixes throughout.

    parent_mask : (Y, X) int ndarray, optional
        A second labeled mask representing parent objects (e.g. cells when
        mask contains nuclei).  When provided, ``relate_mask`` is called to
        assign each child object (in ``mask``) to its dominant parent (in
        ``parent_mask``) via pixel-overlap plurality voting — identical to
        CellProfiler's RelateObjects algorithm.
        Produces column:  ``Parent_{parent_mask_name}``  (int, 0 = no parent)

    parent_mask_name : str, default "Parent"
        Human-readable name of the parent object type, used as the column
        name suffix.
        Column:  ``Parent_{parent_mask_name}``
        Example: parent_mask_name="Cell"  →  ``Parent_Cell``

    intensity_channels : list[str], optional
        Channels for mean / median / std / sum intensity statistics.
        Defaults to **all** channels when None.
        Column pattern:  ``Intensity_{stat}_{channel_name}``

    radial_distribution_channels : list[str], optional
        Channels for radial distribution.  Skipped when None.
        Column pattern:  ``RadialDistribution_bin{i}_{channel_name}``
            i=0 → outermost ring, i=nbins-1 → centre

    radial_distribution_kwargs : dict, optional
        Forwarded to ``make_radial_distribution``.
        Supported keys:  ``nbins`` (int, default 4)

    granularity_channels : list[str], optional
        Channels for granularity spectrum.  Skipped when None.
        Column pattern:  ``Granularity_scale{s}_{channel_name}``

    granularity_kwargs : dict, optional
        Forwarded to ``make_granularity``.
        Supported keys:  ``scales``, ``subsample_size``, ``element_size``

    glcm_channels : list[str], optional
        Channels for GLCM texture features.  Skipped when None.
        Column pattern:  ``GLCM_{prop}_d{distance}_{channel_name}``

    glcm_kwargs : dict, optional
        Forwarded to ``make_glcm``.
        Supported keys:  ``distances``, ``angles``, ``levels``, ``props``

    correlation_pairs : list[(str, str)], optional
        Channel name pairs for Pearson correlation.  Skipped when None.
        Column pattern:  ``Correlation_pearson_{ch_A}_{ch_B}``

    boundary_fraction_threshold : float, default 0.25
        An object is flagged as a boundary object when the fraction of its
        pixels that lie on the image border (outermost row/column ring)
        exceeds this threshold.  Matches CellProfiler's default of 25 %.
        Column:  ``is_boundary``  (bool)

    Returns
    -------
    pd.DataFrame
        One row per labeled object.  Column groups (in order):

        Shape:
            label, area,
            bbox_min_y, bbox_min_x, bbox_max_y, bbox_max_x,
            centroid_y, centroid_x,
            eccentricity, equivalent_diameter_area, euler_number,
            extent, feret_diameter_max,
            major_axis_length, minor_axis_length,
            orientation, perimeter, solidity

        Boundary flag:
            is_boundary  (bool — True when > boundary_fraction_threshold of
                          the object's pixels touch the image edge)

        Parent relationship (only when parent_mask is provided):
            Parent_{parent_mask_name}  (int — parent label, 0 = no parent)

        Intensity:
            Intensity_mean_{ch}, Intensity_median_{ch},
            Intensity_std_{ch},  Intensity_sum_{ch}

        Radial distribution:
            RadialDistribution_bin{0..N-1}_{ch}

        Granularity:
            Granularity_scale{s}_{ch}

        GLCM texture:
            GLCM_{prop}_d{distance}_{ch}

        Pearson correlation:
            Correlation_pearson_{ch_A}_{ch_B}

    Raises
    ------
    ValueError
        - Any channel name not found in ``channel_names``.
        - ``img`` / ``mask`` shapes are incompatible.
        - ``parent_mask`` shape does not match ``mask``.
    """

    # ── Validation ────────────────────────────────────────────────────────────
    if img.ndim != 3:
        raise ValueError(f"img must be (Y, X, C), got shape {img.shape}")
    if mask.shape != img.shape[:2]:
        raise ValueError(
            f"mask shape {mask.shape} does not match img spatial shape {img.shape[:2]}"
        )
    if len(channel_names) != img.shape[2]:
        raise ValueError(
            f"len(channel_names)={len(channel_names)} but img has "
            f"{img.shape[2]} channels (axis 2)"
        )
    if parent_mask is not None and parent_mask.shape != mask.shape:
        raise ValueError(
            f"parent_mask shape {parent_mask.shape} does not match "
            f"mask shape {mask.shape}"
        )

    # ── Default: measure intensity on all channels ────────────────────────────
    if intensity_channels is None:
        intensity_channels = list(channel_names)

    # ── Resolve channel name strings → integer indices ────────────────────────
    intensity_idx   = _resolve_channels(intensity_channels,           channel_names, "intensity_channels")
    radial_idx      = _resolve_channels(radial_distribution_channels, channel_names, "radial_distribution_channels")
    granularity_idx = _resolve_channels(granularity_channels,         channel_names, "granularity_channels")
    glcm_idx        = _resolve_channels(glcm_channels,                channel_names, "glcm_channels")

    # ── Resolve correlation pairs (str, str) → (int, int) ────────────────────
    corr_idx_pairs: list[tuple[int, int]] = []
    if correlation_pairs:
        for ch_a, ch_b in correlation_pairs:
            for name, side in [(ch_a, "first"), (ch_b, "second")]:
                if name not in channel_names:
                    raise ValueError(
                        f"Correlation pair {side} channel '{name}' not found "
                        f"in channel_names: {channel_names}"
                    )
            corr_idx_pairs.append(
                (channel_names.index(ch_a), channel_names.index(ch_b))
            )

    # ── Step 1: Shape descriptors ─────────────────────────────────────────────
    shape_props = regionprops_table(mask, properties=_SHAPE_PROPERTIES)
    df = pd.DataFrame(shape_props).rename(
        columns={k: v for k, v in _SHAPE_COLUMN_RENAMES.items() if k in shape_props}
    )

    # ── Step 2: Boundary flag ─────────────────────────────────────────────────
    # Computed after shape so it sits early in the column order, right after
    # the core morphology block.  Uses the same label ordering as df.
    boundary_map = _is_boundary_object(mask, boundary_fraction_threshold)
    df["is_boundary"] = df["label"].map(boundary_map)

    # ── Step 3: Parent relationship via relate_mask ───────────────────────────
    if parent_mask is not None:
        parent_map = relate_mask(mask, parent_mask)
        col_name = f"Parent_{parent_mask_name}"
        df[col_name] = df["label"].map(parent_map).fillna(0).astype(int)

    # ── Step 4: Build per-channel extra_properties groups ────────────────────
    channel_groups: dict[int, list] = {}

    def _add(ch_idx: int, fns: list) -> None:
        channel_groups.setdefault(ch_idx, []).extend(fns)

    for idx in intensity_idx:
        _add(idx, _build_intensity_fns_for_channel(channel_names[idx]))

    rd_kw = dict(radial_distribution_kwargs or {})
    for idx in radial_idx:
        fns = make_radial_distribution(channel=0, **rd_kw)
        _rename_channel_index_to_name(fns, 0, channel_names[idx])
        _add(idx, fns)

    gr_kw = dict(granularity_kwargs or {})
    for idx in granularity_idx:
        fns = make_granularity(channel=0, **gr_kw)
        _rename_channel_index_to_name(fns, 0, channel_names[idx])
        _add(idx, fns)

    gl_kw = dict(glcm_kwargs or {})
    for idx in glcm_idx:
        fns = make_glcm(channel=0, **gl_kw)
        _rename_channel_index_to_name(fns, 0, channel_names[idx])
        _add(idx, fns)

    # ── Step 5: Run one regionprops_table per channel ─────────────────────────
    if channel_groups:
        df_extra = _run_per_channel_regionprops(mask, img, channel_names, channel_groups)
        df = df.merge(df_extra, on="label", how="left")

    # ── Step 6: Pearson correlation ───────────────────────────────────────────
    if corr_idx_pairs:
        corr_dict = measure_channel_correlation(
            label_image        = mask,
            multichannel_image = img,
            channel_pairs      = corr_idx_pairs,
        )
        renamed: dict[str, np.ndarray] = {"label": corr_dict["label"]}
        for a, b in corr_idx_pairs:
            old_key = f"Correlation_pearson_ch{a}_ch{b}"
            new_key = f"Correlation_pearson_{channel_names[a]}_{channel_names[b]}"
            renamed[new_key] = corr_dict[old_key]
        df = df.merge(pd.DataFrame(renamed), on="label", how="left")
    
    # ── Step 7: Insert metadata columns at the front (if provided) ───────────────────────────────────────────
    if metadata_row:
        df = pd.concat([pd.DataFrame([metadata_row]*len(df)), df], axis=1)
    
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Usage example
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from skimage.draw import disk as draw_disk

    rng = np.random.default_rng(0)
    H, W = 128, 128
    CHANNELS = ["DAPI", "GFP", "RFP"]
    img = rng.random((H, W, len(CHANNELS))).astype(np.float32)

    # ── Child mask: 5 nuclei (two partially outside the image boundary) ───────
    # Objects 4 and 5 are centred outside/at the very edge of the image so
    # their clipped visible portions have > 25 % of pixels on the border ring.
    child_mask = np.zeros((H, W), dtype=int)
    nuclei = [
        ( 30,  30, 14),   # obj 1 — interior
        ( 80,  75, 18),   # obj 2 — interior
        ( 40,  95, 10),   # obj 3 — interior
        ( -4,  64, 12),   # obj 4 — clipped at top  → is_boundary=True
        ( 64, 132, 10),   # obj 5 — clipped at right → is_boundary=True
    ]
    for obj_id, (cy, cx, r) in enumerate(nuclei, start=1):
        rr, cc = draw_disk((cy, cx), r, shape=(H, W))
        child_mask[rr, cc] = obj_id

    # ── Parent mask: 3 large cells containing the interior nuclei ────────────
    parent_mask = np.zeros((H, W), dtype=int)
    cells = [
        (30,  30, 28),   # cell 1 — contains nucleus 1
        (80,  75, 35),   # cell 2 — contains nucleus 2
        (40,  95, 22),   # cell 3 — contains nucleus 3
    ]
    for cell_id, (cy, cx, r) in enumerate(cells, start=1):
        rr, cc = draw_disk((cy, cx), r, shape=(H, W))
        parent_mask[rr, cc] = cell_id

    # ── Standalone relate_mask demo ───────────────────────────────────────────
    print("=== relate_mask result ===")
    parent_map = relate_mask(child_mask, parent_mask)
    for child_lbl, parent_lbl in sorted(parent_map.items()):
        status = f"→ Cell {parent_lbl}" if parent_lbl else "→ no parent (boundary / isolated)"
        print(f"  Nucleus {child_lbl:2d}  {status}")

    # ── Full measure_objects call ─────────────────────────────────────────────
    df = measure_objects(
        mask          = child_mask,
        img           = img,
        channel_names = CHANNELS,

        parent_mask      = parent_mask,
        parent_mask_name = "Cell",

        intensity_channels           = ["DAPI", "GFP", "RFP"],
        radial_distribution_channels = ["DAPI", "GFP"],
        radial_distribution_kwargs   = {"nbins": 4},
        granularity_channels         = ["DAPI"],
        granularity_kwargs           = {"scales": range(1, 9)},
        glcm_channels                = ["GFP", "RFP"],
        glcm_kwargs                  = {"distances": [1, 4], "levels": 8},
        correlation_pairs            = [("DAPI", "GFP"), ("GFP", "RFP")],

        boundary_fraction_threshold  = 0.25,
    )

    print(f"\n=== measure_objects result ===")
    print(f"Shape: {df.shape}  ({df.shape[0]} objects × {df.shape[1]} features)\n")
    print("All columns:")
    for col in df.columns:
        print(f"  {col}")

    print("\nKey columns (label / boundary / parent / intensity):")
    key_cols = ["label", "area", "is_boundary", "Parent_Cell",
                "Intensity_mean_DAPI", "Intensity_mean_GFP", "Intensity_mean_RFP"]
    print(df[key_cols].to_string(index=False))
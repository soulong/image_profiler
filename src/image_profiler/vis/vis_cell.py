"""Streamlit-based single cell image viewer.

This module provides a visualization tool for viewing cropped cell images
based on profiling results stored in a SQLite database.

Usage
-----
Run as a Streamlit app:

    streamlit run -m image_profiler.vis.vis_cell

Or programmatically:

    from image_profiler.vis import run_cell_viewer
    run_cell_viewer(db_path="path/to/result.db")
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import streamlit as st
from skimage import exposure

from image_profiler.utils.database import Database
from image_profiler.utils.crop import crop_cell


def normalize_image(img: np.ndarray, method: str, clip_limit: float = 0.02, gamma: float = 0.4) -> np.ndarray:
    """Normalize image using specified method.

    Parameters
    ----------
    img : np.ndarray
        Input image array.
    method : str
        Normalization method: "None", "Percentile", "Equalize Histogram", "CLAHE", or "Gamma".
    clip_limit : float
        Clip limit for CLAHE method.
    gamma : float
        Gamma value for Gamma correction.

    Returns
    -------
    np.ndarray
        Normalized image.
    """
    if method == "None":
        return img
    elif method == "Percentile":
        p1, p2 = np.percentile(img, (0.1, 99.9))
        return exposure.rescale_intensity(img, in_range=(p1, p2), out_range=(0, 0.9999))
    elif method == "Equalize Histogram":
        return exposure.equalize_hist(img)
    elif method == "CLAHE":
        return exposure.equalize_adapthist(img, clip_limit=clip_limit)
    elif method == "Gamma":
        return exposure.adjust_gamma(img, gamma=gamma, gain=1)
    return img


def normalize_image_by_group(
    img: np.ndarray,
    method: str,
    by: str = "None",
    clip_limit: float = 0.02,
    gamma: float = 0.4
) -> np.ndarray:
    """Normalize 4D image (B, H, W, C) along specified axes.

    Parameters
    ----------
    img : np.ndarray
        Shape (B, H, W, C)
    method : str
        Normalization method
    by : str
        "None"  -> no grouping, normalize whole volume at once
        "B"     -> normalize each batch item independently (HWC for each B)
        "C"     -> normalize each channel independently (across B,H,W)
        "BC"    -> normalize each (H,W) slice per batch and channel
    clip_limit : float
        Clip limit for CLAHE method.
    gamma : float
        Gamma value for Gamma correction.

    Returns
    -------
    np.ndarray
        Normalized image array.
    """
    if img.ndim != 4:
        raise ValueError("Input must be 4D array (B, H, W, C)")

    if method == "None" or by == "None":
        return normalize_image(img, method, clip_limit, gamma)

    img = img.astype(np.float32)

    if by == "B":
        normalized_batches = []
        for i in range(img.shape[0]):
            normalized_batches.append(normalize_image(img[i], method, clip_limit, gamma))
        return np.stack(normalized_batches, axis=0)

    elif by == "C":
        normalized_channels = []
        for c in range(img.shape[3]):
            channel = img[..., c]
            normalized_channels.append(normalize_image(channel, method, clip_limit, gamma)[..., np.newaxis])
        return np.concatenate(normalized_channels, axis=-1)

    elif by == "BC":
        B, H, W, C = img.shape
        img_reshaped = img.transpose(0, 3, 1, 2).reshape(B * C, H, W)
        normalized_reshaped = normalize_image(img_reshaped, method, clip_limit, gamma)
        return normalized_reshaped.reshape(B, C, H, W).transpose(0, 2, 3, 1)

    else:
        raise ValueError(f"Invalid 'by' parameter: {by}. Choose from 'B', 'C', 'BC', 'None'")


@st.cache_data(show_spinner="Loading data from database...")
def load_data_from_db(db_path: str) -> dict:
    """Load metadata and cell profile results from SQLite database.

    Parameters
    ----------
    db_path : str
        Path to SQLite database file.

    Returns
    -------
    dict
        Dictionary with keys: "metadata", "cell_profile", "tables"
    """
    db_path = Path(db_path)
    if not db_path.exists():
        st.error(f"Database file not found: {db_path}")
        return None

    with Database(db_path) as db:
        tables = db.get_tables()

        data = {"tables": tables}

        if "metadata" in tables:
            data["metadata"] = db.query("SELECT * FROM metadata")
        else:
            data["metadata"] = pd.DataFrame()

        cell_tables = [t for t in tables if t not in ["metadata", "image"]]
        data["cell_tables"] = cell_tables

        if cell_tables:
            first_cell_table = cell_tables[0]
            data["cell_profile"] = db.query(f"SELECT * FROM {first_cell_table}")
            data["selected_table"] = first_cell_table
        else:
            data["cell_profile"] = pd.DataFrame()
            data["selected_table"] = None

    return data


@st.cache_data(show_spinner="Loading cell profile table...")
def load_cell_table(db_path: str, table_name: str) -> pd.DataFrame:
    """Load a specific cell profile table from database.

    Parameters
    ----------
    db_path : str
        Path to SQLite database file.
    table_name : str
        Name of the table to load.

    Returns
    -------
    pd.DataFrame
        Cell profile data.
    """
    with Database(db_path) as db:
        return db.query(f"SELECT * FROM {table_name}")


def run_cell_viewer(db_path: Optional[Union[str, Path]] = None):
    """Run the Streamlit cell viewer application.

    Parameters
    ----------
    db_path : str or Path, optional
        Path to SQLite database file. If None, user must upload via UI.
    """
    st.set_page_config(page_title="Single Cell Image Viewer", page_icon=":microscope:", layout="wide")
    st.title("Single Cell Image Viewer")

    st.sidebar.header("Input Data")

    if db_path is None:
        db_file = st.sidebar.file_uploader("Upload result.db SQLite file", type=["db", "sqlite", "sqlite3"])
        if db_file is not None:
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
                tmp.write(db_file.getvalue())
                db_path = tmp.name
    else:
        db_path = str(db_path)
        st.sidebar.text(f"Database: {Path(db_path).name}")

    if db_path is None:
        st.info("Please provide a SQLite database file (result.db).")
        st.stop()

    data = load_data_from_db(db_path)

    if data is None:
        st.stop()

    metadata_df = data["metadata"]
    cell_profile_df = data["cell_profile"]
    tables = data["tables"]
    cell_tables = data.get("cell_tables", [])

    if metadata_df.empty:
        st.error("No 'metadata' table found in database.")
        st.stop()

    st.sidebar.header("Cell Profile Table")
    if cell_tables:
        selected_table = st.sidebar.selectbox(
            "Select cell profile table",
            cell_tables,
            index=cell_tables.index(data.get("selected_table", cell_tables[0]))
        )
        if selected_table != data.get("selected_table"):
            cell_profile_df = load_cell_table(db_path, selected_table)
    else:
        st.warning("No cell profile tables found in database.")
        st.stop()

    st.sidebar.header("Cell Cropping")
    clip_mask = st.sidebar.checkbox("Clip mask", value=False)
    pad_to_square = st.sidebar.checkbox("Pad to square", value=False)
    rotate_to_horizontal = st.sidebar.checkbox("Rotate horizontally", value=False)
    target_crop_size = st.sidebar.number_input("Resize (px)", min_value=64, max_value=512, value=128, step=64)

    st.sidebar.header("Cell Normalization")
    norm_method = st.sidebar.selectbox("Normalization", ["Percentile", "Equalize Histogram", "CLAHE", "Gamma", "None"])
    clip_limit = 0.02
    gamma = 0.4
    if norm_method == "CLAHE":
        clip_limit = st.sidebar.number_input("Clip limit", 0.0, 1.0, 0.02, step=0.005)
    if norm_method == "Gamma":
        gamma = st.sidebar.number_input("Gamma", 0.0, 5.0, 0.4, step=0.1)
    norm_axis = st.sidebar.radio("Norm axis", ["B", "C", "BC"], horizontal=True)

    st.sidebar.header("Display Option")
    show_filename = st.sidebar.checkbox("Show filename", value=False)
    n_fields_per_row = st.sidebar.number_input("N fields per row", min_value=5, max_value=50, value=10, step=5)

    metadata_cols = [c for c in cell_profile_df.columns if c.startswith("Metadata_")]
    numeric_cols = [c for c in cell_profile_df.columns if not c.startswith("Metadata_")]

    filtered_result = cell_profile_df.copy()

    with st.expander("Apply Filters", expanded=True):
        slider_n_per_row = 4
        cols = st.columns(slider_n_per_row, gap="small")

        for i, col in enumerate(metadata_cols):
            with cols[i % slider_n_per_row]:
                unique_vals = sorted(cell_profile_df[col].dropna().unique().tolist())
                if len(unique_vals) > 50:
                    selected = st.multiselect(f"{col}", unique_vals, default=unique_vals[:10])
                else:
                    options = ["All"] + unique_vals
                    selected = st.multiselect(f"{col}", options, default=["All"])
                    if "All" not in selected and selected:
                        pass
                    else:
                        selected = unique_vals
                if selected and selected != ["All"]:
                    filtered_result = filtered_result[filtered_result[col].isin(selected)]

        for i, col in enumerate(numeric_cols[:8]):
            with cols[i % slider_n_per_row]:
                mn, mx = float(cell_profile_df[col].min()), float(cell_profile_df[col].max())
                if mn == mx:
                    continue
                vmin, vmax = st.slider(f"{col}", mn, mx, (mn, mx), step=(mx - mn) / 20)
                filtered_result = filtered_result[(filtered_result[col] >= vmin) & (filtered_result[col] <= vmax)]

    if len(filtered_result) == 0:
        st.warning("No cells match the filters.")
        st.stop()

    st.success(f"Filtered to {len(filtered_result)} cells from result table.")

    common_keys = [c for c in metadata_cols if c in metadata_df.columns]

    if not common_keys:
        st.error("No common metadata columns found between cell profile and metadata tables.")
        st.stop()

    merged_df = pd.merge(filtered_result, metadata_df, on=common_keys, how="inner", suffixes=("", "_meta"))

    if len(merged_df) == 0:
        st.error("No matching records found between cell profile and metadata.")
        st.stop()

    st.info(f"Found {len(merged_df)} cell records with matching metadata.")

    display_df = merged_df.copy()

    max_cells_per_image = st.sidebar.number_input("Max cells per image", min_value=5, max_value=100, value=20, step=5)

    sort_options = numeric_cols if numeric_cols else ["index"]
    sort_col = st.sidebar.selectbox("Sort cells by", sort_options)
    sort_order = st.sidebar.radio("Sort order", ["Increase", "Random", "Decrease", "As is"], horizontal=True)

    if sort_col == "index":
        pass
    elif sort_order == "Increase":
        display_df = display_df.sort_values(sort_col, ascending=True)
    elif sort_order == "Decrease":
        display_df = display_df.sort_values(sort_col, ascending=False)
    elif sort_order == "Random":
        display_df = display_df.sample(frac=1, random_state=42)

    group_cols = st.sidebar.multiselect("Grouping by", metadata_cols, default=metadata_cols[:1] if metadata_cols else [])

    channel_cols = [c for c in metadata_df.columns if c.startswith("ch") and not c.startswith("Metadata_")]
    if not channel_cols:
        channel_cols = [c for c in metadata_df.columns if "channel" in c.lower() and not c.startswith("Metadata_")]

    show_channels = st.sidebar.multiselect("Show Channels", channel_cols, default=channel_cols[:1] if channel_cols else [])

    mask_cols = [c for c in metadata_df.columns if "mask" in c.lower() and not c.startswith("Metadata_")]
    mask_file_column = st.sidebar.selectbox("Mask Column", mask_cols if mask_cols else ["No mask columns found"])

    cell_id_candidates = [c for c in numeric_cols if "cell" in c.lower() and "id" in c.lower()]
    if not cell_id_candidates:
        cell_id_candidates = [c for c in numeric_cols if "label" in c.lower() or "object" in c.lower()]
    if not cell_id_candidates:
        cell_id_candidates = numeric_cols[:3] if numeric_cols else ["No numeric columns"]

    cell_id_column = st.sidebar.selectbox("Cell ID Column", cell_id_candidates)

    if not show_channels:
        st.warning("Please select at least one channel to display.")
        st.stop()

    if mask_file_column == "No mask columns found":
        st.error("No mask columns found in metadata.")
        st.stop()

    for group_key, group_df in display_df.groupby(group_cols, dropna=False) if group_cols else [(None, display_df)]:
        if group_key is not None:
            group_key = [group_key] if isinstance(group_key, str) else group_key
            group_name = " → ".join([str(k) for k in group_key])
        else:
            group_name = "All Cells"

        with st.expander(f"{group_name} ({len(group_df)} cells)", expanded=True):

            cropped_all = []
            count = 0
            cols = st.columns(n_fields_per_row, gap="small")

            for idx, (_, row) in enumerate(group_df.iterrows()):

                try:
                    directory = row.get("directory")
                    if directory is None:
                        continue

                    parent_dir = Path(directory)

                    mask_filename = row.get(mask_file_column)
                    if pd.isna(mask_filename):
                        continue
                    mask_path = parent_dir / mask_filename

                    image_channel_paths = []
                    for ch in show_channels:
                        ch_filename = row.get(ch)
                        if pd.notna(ch_filename):
                            image_channel_paths.append(parent_dir / ch_filename)

                    if not image_channel_paths:
                        continue

                    cell_id_val = row.get(cell_id_column)
                    if pd.isna(cell_id_val):
                        continue
                    cell_ids = [int(cell_id_val)]

                    if not mask_path.exists():
                        continue
                    if any(not img_path.exists() for img_path in image_channel_paths):
                        continue

                    cropped = crop_cell(
                        mask=mask_path,
                        imgs=image_channel_paths,
                        cell_ids=cell_ids,
                        scale_factor=65535.0,
                        target_size=target_crop_size if target_crop_size > 0 else None,
                        clip_mask=clip_mask,
                        pad_square=pad_to_square,
                        rotate_horizontal=rotate_to_horizontal
                    )

                    if not cropped or cropped[0].get("cell_img") is None:
                        continue

                    cropped_img = [c["cell_img"] for c in cropped if c.get("cell_img") is not None]
                    if not cropped_img:
                        continue

                    cropped_arr = np.stack(cropped_img, axis=0)

                    cropped_norm = normalize_image_by_group(cropped_arr, norm_method, by=norm_axis, clip_limit=clip_limit, gamma=gamma)

                    B, H, W, C = cropped_norm.shape
                    cropped_norm = cropped_norm.reshape(B * H, W * C)

                    caption = f"{image_channel_paths[0].name}" if show_filename else None
                    with cols[count % n_fields_per_row]:
                        st.image(cropped_norm, caption=caption, clamp=True, width='stretch')
                    count += 1

                    if count >= max_cells_per_image * n_fields_per_row:
                        break

                except Exception as e:
                    st.warning(f"Error processing cell: {str(e)}")
                    continue

    st.success("Done! Use filters to explore.")


if __name__ == "__main__":
    run_cell_viewer()

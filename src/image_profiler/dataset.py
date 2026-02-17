"""ImageDataset class for managing microscopy image datasets."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Union

import numpy as np
import pandas as pd
from natsort import natsorted
from tqdm import tqdm

try:
    import imageio.v3 as iio
except ImportError:
    import imageio as iio
                
from image_profiler.utils.helper import images_to_dataset, write_dataloader
from image_profiler.utils.database import write_results_to_db
from image_profiler.utils.segmentate import cellpose_segment_measurement
from image_profiler.utils.crop import crop_cell
from image_profiler.preprocessing.correction import fit_basic_models, transform_basic_models
from image_profiler.preprocessing.split_tile import tile_images_from_metadata
from image_profiler.preprocessing.z_projection import z_project_dataset
from image_profiler.analysis.image_profiling import profile_image_single_row
from image_profiler.analysis.object_profiling import profile_object_single_row
from image_profiler.analysis.extra_properties import build_extra_properties


class ImageDataset:
    """Central data structure for managing multi-well plate image datasets.
    
    Attributes
    ----------
    measurement_dir : Path
        Directory containing the measurement images.
    image_pattern : str
        Regex pattern for parsing image filenames.
    mask_pattern : str
        Regex pattern for parsing mask filenames (with _cp_masks_ suffix).
    metadata : pd.DataFrame
        Metadata table with one row per unique imaging site.
    intensity_colnames : list of str
        Column names for intensity images.
    mask_colnames : list of str
        Column names for mask images.
    """
    DEFAULT_IMAGE_PATTERN = (
    r"r(?P<row>.*)c(?P<column>.*)f(?P<field>.*)p(?P<stack>.*)-ch(?P<channel>.*)sk(?P<timepoint>[0-9]{1,})fk1fl1"
    r".tiff"
    )

    DEFAULT_MASK_PATTERN = (
        r"r(?P<row>.*)c(?P<column>.*)f(?P<field>.*)p(?P<stack>.*)-ch(?P<channel>.*)sk(?P<timepoint>[0-9]{1,})fk1fl1"
        r"_cp_masks_(?P<mask_name>.*)"
        r".png"
    )

    def __init__(
        self,
        measurement_dir: Union[str, Path],
        image_pattern: Optional[str] = None,
        mask_pattern: Optional[str] = None,
        dataset_kwargs: Optional[Dict] = None
    ) -> None:
        """Initialize ImageDataset.

        Parameters
        ----------
        measurement_dir : str or Path
            Path to measurement directory containing Images folder.
        image_pattern : str, optional
            Custom regex pattern for image filenames.
        mask_pattern : str, optional
            Custom regex pattern for mask filenames (with _cp_masks_ suffix).
        dataset_kwargs : dict, optional
            Additional arguments for build_metadata.
        """
        self.measurement_dir = Path(measurement_dir)
        self.image_pattern = image_pattern or self.DEFAULT_IMAGE_PATTERN
        self.mask_pattern = mask_pattern or self.DEFAULT_MASK_PATTERN
        self._dataset_kwargs = dataset_kwargs or {}
        self.meta_dict: Optional[Dict] = None
        self._metadata: Optional[pd.DataFrame] = None
        self._intensity_colnames: Optional[List[str]] = None
        self._mask_colnames: Optional[List[str]] = None
        
        self.build_metadata()
    
    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"\n--------- ImageDataset ----------\n"            f"# dir={self.measurement_dir!r}\n"
            f" - image_pattern:\n{self.image_pattern!r}\n"
            f" - mask_pattern:\n{self.mask_pattern!r}\n"
            f" - metadata_colnames: {(self.metadata.columns.to_list()) if self._metadata is not None else ''}\n"
            f" - intensity_colnames: {self._intensity_colnames}\n"
            f" - mask_colnames: {self._mask_colnames}\n"
            f" - rows: {len(self.metadata) if self._metadata is not None else 0}"
            f"\n-----------------------------------\n"
        )
    
    def __len__(self) -> int:
        """Return number of rows in metadata."""
        return len(self.metadata)
    
    def __iter__(self) -> Iterator[tuple]:
        """Iterate over all image sets in metadata.
        
        Yields
        ------
        tuple
            (image_data, mask_data) for each row.
        """
        for idx in range(len(self)):
            yield self.get_imageset(idx)
    
    @property
    def metadata(self) -> pd.DataFrame:
        """Get metadata DataFrame."""
        if self._metadata is None:
            self.build_metadata()
        return self._metadata
    
    @property
    def intensity_colnames(self) -> List[str]:
        """Get intensity column names."""
        if self._intensity_colnames is None:
            self.build_metadata()
        return self._intensity_colnames or []
    
    @property
    def mask_colnames(self) -> List[str]:
        """Get mask column names."""
        if self._mask_colnames is None:
            self.build_metadata()
        return self._mask_colnames or []
    
    @property
    def channels(self) -> List[str]:
        """Alias for intensity_colnames."""
        return self.intensity_colnames
    
    @property
    def masks(self) -> List[str]:
        """Alias for mask_colnames."""
        return self.mask_colnames
    
    @property
    def dataset_dir(self) -> Path:
        """Alias for measurement_dir for compatibility."""
        return self.measurement_dir
    
    def build_metadata(self, remove_na_row=True, **dataset_kwargs) -> None:
        """Build metadata by scanning image directory."""
        self.meta_dict = images_to_dataset(
            self.measurement_dir,
            self.image_pattern,
            self.mask_pattern,
            remove_na_row = remove_na_row,
            **dataset_kwargs
        )
        
        if self.meta_dict is not None:
            self._metadata = self.meta_dict.get('metadata')
            self._intensity_colnames = self.meta_dict.get('intensity_colnames', [])
            self._mask_colnames = self.meta_dict.get('mask_colnames', [])
        else:
            self._metadata = pd.DataFrame()
            self._intensity_colnames = []
            self._mask_colnames = []
        
        print(self.__repr__())
        
    def export_dataloader(self, 
                          remove_na_rows: bool = True,
                          output_path: Optional[Union[str, Path]] = None) -> Path:
        """Export metadata to CellProfiler-compatible CSV format.

        Parameters
        ----------
        remove_na_rows : bool, optional
            Whether to remove rows with missing intensity or mask values.
        output_path : str or Path, optional
            Output CSV path. Default: measurement_dir/dataloader.csv.

        Returns
        -------
        pd.DataFrame
            Converted DataFrame.
        """
        
        if remove_na_rows:
            print(f"Removing {len(self._metadata)} rows with missing intensity or mask values.")
            metadata_dataloader = self._metadata.dropna(subset=self.intensity_colnames + self.mask_colnames)
        else:
            metadata_dataloader = self._metadata
        
        if output_path is None:
            output_path = self.measurement_dir / "dataloader.csv"
        
        write_dataloader(
            metadata_dataloader,
            self.intensity_colnames,
            self.mask_colnames,
            str(output_path)
        )
        return metadata_dataloader
    
    def get_imageset(self, row_idx: int, channel_axis: int = 0) -> tuple:
        """Get image data and masks for a specific row.

        Parameters
        ----------
        row_idx : int
            Row index in metadata.

        Returns
        -------
        tuple
            (image_data, mask_data) where image_data is np.ndarray (C, Y, X)
            and mask_data is dict of np.ndarray.
        """
        row = self.metadata.iloc[row_idx]
        
        image_paths = []
        for ch_col in self.intensity_colnames:
            if pd.notna(row.get(ch_col)):
                image_paths.append(Path(row["directory"]) / row[ch_col])
        
        mask_paths = {}
        for mask_col in self.mask_colnames:
            if pd.notna(row.get(mask_col)):
                mask_name = mask_col.replace("mask_", "")
                mask_paths[mask_name] = Path(row["directory"]) / row[mask_col]
        
        images = []
        for path in image_paths:
            if path.exists():
                images.append(iio.imread(path))
        
        image_data = np.stack(images, axis=channel_axis) if images else None
        
        mask_data = {}
        for name, path in mask_paths.items():
            if path.exists():
                mask_data[name] = iio.imread(path)
        
        return image_data, mask_data
    
    def segmentate(
        self,
        object_name: str = 'cell',
        chan1: Optional[List[str]] = None,
        chan2: Optional[List[str]] = None,
        merge1: str = "mean",
        merge2: str = "mean",
        model_name: str = "cpsam",
        diameter: Optional[float] = None,
        normalize: Optional[Dict] = {"percentile": [0.1, 99.9]},
        resize_factor: float = 1.0,
        overwrite_mask: bool = False,
        flow_threshold: float = 0.4,
        cellprob_threshold: float = 0.0,
        gpu_batch_size: int = 16
    ) -> Dict:
        """Run Cellpose-SAM segmentation on the dataset.

        Parameters
        ----------
        object_name : str
            Suffix for mask filenames (e.g., "cell", "nuclei").
        chan1 : list of str, optional
            First channel group. Default: first intensity channel.
        chan2 : list of str, optional
            Second channel group. If None, C2 = 0.
        merge1 : str
            How to merge channels in chan1: "mean", "max", or "min".
        merge2 : str
            How to merge channels in chan2: "mean", "max", or "min".
        model_name : str
            Cellpose model name or path.
        diameter : float, optional
            Approximate object diameter in pixels. None = auto-estimate.
        normalize : dict, optional
            Normalization params, e.g., {"percentile": [0.1, 99.9]}.
        resize_factor : float
            Resize factor before segmentation (1.0 = original size).
        overwrite_mask : bool
            Overwrite existing mask files.
        flow_threshold : float
            Flow error threshold.
        cellprob_threshold : float
            Cell probability threshold.
        gpu_batch_size : int
            GPU batch size for processing.

        Returns
        -------
        dict
            Summary with keys: "success", "processed", "skipped", "failed", 
            "masks_saved", "errors".
        """
        if chan1 is None:
            chan1 = [self.intensity_colnames[0]] if self.intensity_colnames else ["ch1"]
        
        summary = cellpose_segment_measurement(
            dataset=self.meta_dict,
            chan1=chan1,
            chan2=chan2,
            merge1=merge1,
            merge2=merge2,
            model_name=model_name,
            diameter=diameter,
            normalize=normalize,
            resize_factor=resize_factor,
            mask_name=object_name,
            overwrite_mask=overwrite_mask,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            gpu_batch_size=gpu_batch_size,
        )
        
        self.build_metadata()
        self.segmentate_summary = summary
        
        return summary
    
    def export_metadata(self, 
                        write_db: Union[bool, str, None] = True, 
                        table_name: str = "metadata"):
        """Export metadata to database.
        
        Parameters
        ----------
        write_db : bool or str or None
            ``True``  → write to ``<measurement_dir>/result.db`` (table "metadata").
            ``str``   → write to ``<measurement_dir>/<write_db>``.
            ``False`` / ``None`` → return ``pd.DataFrame``.
        table_name : str
            table name
        """
        if write_db is True:
            db_path = self.dataset_dir / "result.db"
        elif isinstance(write_db, str):
            db_path = self.dataset_dir / f"{write_db}"
        else:
            db_path = None
        
        if db_path is None:
            return self.metadata
        else:
            write_results_to_db(
                    db_path,
                    table_name,
                    self.metadata,
                    if_exists="replace",
                )
            # self.metadata.to_sql(table_name, db_path, if_exists="replace", index=False)
            return None
    
    def profile_image(
        self,
        channels: Optional[List[str]] = None,
        thresholds: Optional[Dict[str, float]] = None,
        row_idx: Optional[int] = None,
        write_db: Union[bool, str, None] = True,
        table_name="image"
    ) -> Optional[pd.DataFrame]:
        """Profile images at whole-image level.

        Computes per-channel statistics (mean, std, min, max, median, skew,
        sum, percentiles) for every imaging site.  When ``thresholds`` are
        provided, additional object-count and area metrics are computed via
        simple thresholding.

        Parameters
        ----------
        channels : list of str, optional
            Channel names to profile.  ``None`` = all channels.
        thresholds : dict of str to float, optional
            Per-channel intensity thresholds for binary mask creation.
            Enables ``{ch}_area``, ``{ch}_n_objects``, ``{ch}_mean_object_area``
            columns.
        row_idx : int, optional
            Profile a single row by index.  ``None`` = profile all rows.
        write_db : bool or str or None
            ``True``  → write to ``<measurement_dir>/result.db`` (table "image").
            ``str``   → write to ``<measurement_dir>/<write_db>``.
            ``False`` / ``None`` → return aggregated ``pd.DataFrame``.
        table_name : str
            table name

        Returns
        -------
        pd.DataFrame or None
            Aggregated results when ``write_db`` is falsy, else ``None``.
        """
        if write_db is True:
            db_path = self.dataset_dir / "result.db"
        elif isinstance(write_db, str):
            db_path = self.dataset_dir / f"{write_db}"
        else:
            db_path = None

        if row_idx is None:
            indices = range(len(self.metadata))
        else:
            indices = [row_idx]

        results = []

        for idx in tqdm(indices, desc="Profiling images"):
            # image_data shape: (C, Y, X) — profile_single_image indexes on C axis
            image_data, _ = self.get_imageset(idx)

            if image_data is None:
                continue

            row = self.metadata.iloc[idx]
            metadata_row = row.to_dict()
            metadata_row = {k: v for k, v in metadata_row.items() 
                            if k not in self.mask_colnames + self.intensity_colnames}

            result = profile_image_single_row(
                image_data=image_data,
                channel_names=self.channels,
                metadata_row=metadata_row,
                channels=channels,
                thresholds=thresholds,
            )

            if result is None:
                continue

            if db_path:
                write_results_to_db(
                    db_path,
                    table_name,
                    pd.DataFrame([result]),
                    if_exists="append",
                )
            else:
                results.append(result)

        if db_path:
            return None

        return pd.DataFrame(results)
    
    def profile_object(
        self,
        mask_name: str,
        parent_mask_name: Optional[str] = None,
        row_idx: Optional[int] = None,
        channels: Optional[List[str]] = None,
        profile: Optional[List[str]] = None,
        extra_properties: Optional[List[Union[str, Callable]]] = None,
        extra_properties_kwargs: Optional[List[Optional[Dict]]] = None,
        write_db: Union[bool, str, None] = True,
        table_name: str = None,
    ) -> Optional[pd.DataFrame]:
        """Profile objects using the specified segmentation mask.

        Parameters
        ----------
        mask_name : str
            Name of the mask to use (e.g. ``"cell"``, ``"nuclei"``).
            Must match a ``mask_{name}`` column in the metadata.
        parent_mask_name : str, optional
            Parent mask name for hierarchical child→parent assignment.
            Adds a ``parent_{parent_mask_name}_id`` column to the output.
        row_idx : int, optional
            Profile a single row by index.  ``None`` = profile all rows.
        channels : list of str, optional
            Channel names to include in intensity measurements.
            ``None`` = all channels.
        profile : list of str, optional
            Feature families to compute.  Any combination of
            ``"shape"`` and ``"intensity"``.  Default: both.
        extra_properties : list of str or callable, optional
            Additional feature sets.  Accepts shorthand strings
            ``'glcm'``, ``'granularity'``, ``'radial'`` or plain callables
            built with the factory functions from ``extra_properties.py``.
        extra_properties_kwargs : list of dict or None, optional
            Per-item keyword arguments forwarded to string-shorthand factory
            functions (e.g. ``[{"distances": [1, 2]}, None, {"n_bins": 6}]``).
            Use ``None`` as a placeholder to keep defaults.
        write_db : bool or str or None
            ``True``  → write to ``<measurement_dir>/result.db``
                        (table named after ``mask_name``).
            ``str``   → write to ``<measurement_dir>/<write_db>``.
            ``False`` / ``None`` → return aggregated ``pd.DataFrame``.
        table_name : str
            table name, if None, use mask_name

        Returns
        -------
        pd.DataFrame or None
            Aggregated results (one row per object) when ``write_db`` is
            falsy, else ``None``.
        """
        resolved_properties, col_names = build_extra_properties(
            extra_properties,
            n_channels=len(self.channels),
            channel_names=self.channels,
            extra_properties_kwargs=extra_properties_kwargs,
        )

        # get_imageset already strips the prefix, so both references are correct.
        if mask_name not in self.metadata.columns:
            raise ValueError(f"Mask column '{mask_name}' not found in metadata")

        if write_db is True:
            db_path = self.dataset_dir / "result.db"
        elif isinstance(write_db, str):
            db_path = self.dataset_dir / f"{write_db}"
        else:
            db_path = None

        if row_idx is None:
            indices = range(len(self.metadata))
        else:
            indices = [row_idx]

        results = []

        for idx in tqdm(indices, desc=f"Profiling {mask_name}"):
            row = self.metadata.iloc[idx]

            if pd.isna(row.get(mask_name)):
                continue

            # image_data shape: (C, Y, X)
            image_data, mask_data = self.get_imageset(idx)
            metadata_row = row.to_dict()
            metadata_row = {k: v for k, v in metadata_row.items() 
                            if k not in self.mask_colnames + self.intensity_colnames}

            result_df = profile_object_single_row(
                image_data=image_data,
                mask_data=mask_data,
                channel_names=self.channels,
                metadata_row=metadata_row,
                mask_name=mask_name,
                parent_mask_name=parent_mask_name,
                channels=channels,
                profile=profile,
                extra_properties=resolved_properties,
                col_names=col_names,
            )

            if result_df is None or result_df.empty:
                continue

            if db_path:
                table_name = mask_name if table_name is None else table_name
                write_results_to_db(db_path, table_name, result_df, if_exists="append")
            else:
                results.append(result_df)

        if db_path:
            return None

        if not results:
            return pd.DataFrame()

        return pd.concat(results, ignore_index=True)
    
    def crop_object(
        self,
        mask_name: str = None,
        row_idx: Optional[Union[int, List[int]]] = None,
        channels: list[str] = None,
        cell_ids: list[int] = None,
        scale_factor: Union[float, None] = None,
        target_size: Optional[int] = None,
        clip_mask: bool = True,
        pad_square: bool = True,
        rotate_horizontal: bool = False,
        expansion_pixel: int = 0
    ) -> List[Dict[str, Any]]:
        """Crop cells from intensity images using segmentation mask.
        """
        results = []
        row_idx = [row_idx] if isinstance(row_idx, int) else row_idx
        indices = range(len(self.metadata)) if row_idx is None else row_idx
        for idx in indices:
            row = self.metadata.iloc[idx]
            
            if mask_name not in row or pd.isna(row[mask_name]):
                continue
            
            mask_path = row['directory'] / row[mask_name]
            
            img_paths = []
            channels = [channels] if isinstance(channels, str) else channels
            channels = self.intensity_colnames if channels is None else channels
            for ch in channels:
                if ch in row and pd.notna(row[ch]):
                    img_paths.append(row['directory'] / row[ch])
            
            crops = crop_cell(
                mask=mask_path,
                imgs=img_paths,
                cell_ids=cell_ids,
                scale_factor=scale_factor,
                target_size=target_size,
                clip_mask=clip_mask,
                pad_square=pad_square,
                rotate_horizontal=rotate_horizontal,
                expansion_pixel=expansion_pixel
            )
            
            for crop in crops:
                crop['metadata_idx'] = idx
                crop['mask_name'] = mask_name
            
            results.extend(crops)
        
        return results
    
    def preprocess_basic_correction(
        self,
        mode: str = 'fit',
        channels: Optional[List[str]] = None,
        n_image: int = 50,
        working_size: int = 128,
        enable_darkfield: bool = False,
        output_root: Optional[Path] = None
    ) -> Dict:
        """Apply BaSiC shading correction.

        Parameters
        ----------
        mode : str
            "fit", "transform", or "fit-transform".
        channels : list of str, optional
            Channels to correct. None = all intensity channels.
        n_image : int
            Number of images to use for fitting.
        working_size : int
            Working size for BaSiC model.
        enable_darkfield : bool
            Enable darkfield estimation.
        output_root : Path, optional
            Output root directory for corrected images.

        Returns
        -------
        dict
            Summary of correction operation with keys depending on mode.
        """
        if channels is None:
            channels = self.intensity_colnames
        
        if mode == 'fit':
            return fit_basic_models(
                self.meta_dict, channels, n_image, working_size, 
                enable_darkfield, output_root
            )
        elif mode == 'transform':
            return transform_basic_models(self.meta_dict, channels, output_root)
        else:
            fit_summary = fit_basic_models(
                self.meta_dict, channels, n_image, working_size,
                enable_darkfield, output_root
            )
            transform_summary = transform_basic_models(self.meta_dict, channels, output_root)
            return {"fit": fit_summary, "transform": transform_summary}
    
    def preprocess_tile_image(
        self,
        tile_w_px: int = 1024,
        tile_h_px: int = 1024,
        delete_originals: bool = False,
        image_pattern_replace: tuple[str, str] = (
            r".tiff",
            r"_tile(?P<t>[0-9]{1,}).tiff"
        ),
        mask_pattern_replace: tuple[str, str] = (
            r"_cp_masks_(?P<mask_name>.*).png",
            r"_tile(?P<tile>[0-9]{1,})_cp_masks_(?P<mask_name>.*).png"
        )
    ) -> Dict:
        """Split images into tiles.

        Parameters
        ----------
        tile_w_px : int
            Tile width in pixels.
        tile_h_px : int
            Tile height in pixels.
        delete_originals : bool
            Whether to delete original images after tiling.
        image_pattern_replace : tuple of str
            Pattern to replace in image_pattern (old, new).
        mask_pattern_replace : tuple of str
            Pattern to replace in mask_pattern (old, new).

        Returns
        -------
        dict
            Summary of tiling operation.
        """
        summary = tile_images_from_metadata(
            self.meta_dict,
            tile_w_px=tile_w_px,
            tile_h_px=tile_h_px,
            delete_originals=delete_originals
        )
        
        self.image_pattern = self.image_pattern.replace(image_pattern_replace[0], image_pattern_replace[1])
        self.mask_pattern = self.mask_pattern.replace(mask_pattern_replace[0], mask_pattern_replace[1])
        print("Update image_pattern:", self.image_pattern)
        print("Update mask_pattern:", self.mask_pattern)
        self.preprocess_tile_image_summary = summary
        
        self.build_metadata()
        
        return summary
    
    def preprocess_z_projection(
        self,
        method: str = "max",
        delete_originals: bool = False,
        image_pattern_replace: tuple[str, str] = (
            r".tiff",
            r"_zpmax.tiff"
        ),
        mask_pattern_replace: tuple[str, str] = (
            r"_cp_masks_(?P<mask_name>.*).png",
            r"_zpmax_cp_masks_(?P<mask_name>.*).png"
        )
    ) -> Dict:
        """Perform Z-stack projection.

        Groups images by all metadata columns except 'stack', then performs
        projection along the Z-axis for each group.

        Parameters
        ----------
        method : str
            Projection method: "max", "mean", or "min".
        delete_originals : bool
            Whether to delete original Z-stack images after projection.
        image_pattern_replace : tuple of str
            Pattern to replace in image_pattern (old, new).
        mask_pattern_replace : tuple of str
            Pattern to replace in mask_pattern (old, new).

        Returns
        -------
        dict
            Summary with keys: "projected", "groups", "deleted", "errors".
        """
        summary = z_project_dataset(
            self.metadata,
            self.intensity_colnames,
            self.mask_colnames,
            method=method,
            delete_originals=delete_originals
        )
        
        self.image_pattern = self.image_pattern.replace(image_pattern_replace[0], image_pattern_replace[1])
        self.mask_pattern = self.mask_pattern.replace(mask_pattern_replace[0], mask_pattern_replace[1])
        print("Update image_pattern:", self.image_pattern)
        print("Update mask_pattern:", self.mask_pattern)
        self.preprocess_z_project_summary = summary
        
        self.build_metadata()
        
        return summary


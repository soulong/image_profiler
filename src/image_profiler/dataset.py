"""ImageDataset class for managing microscopy image datasets."""

from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import ndimage

try:
    import imageio.v3 as iio
except ImportError:
    import imageio as iio

from image_profiler.utils.helper import images_to_dataset, write_dataloader
from image_profiler.utils.database import write_results_to_db
from image_profiler.utils.segmentate import cellpose_segment_measurement
from image_profiler.utils.crop import crop_object
from image_profiler.preprocessing.correction import fit_basic_models, transform_basic_models
from image_profiler.preprocessing.split_tile import tile_images_from_metadata
from image_profiler.preprocessing.z_projection import z_project_dataset
from image_profiler.analysis.image_profiling import measure_image
from image_profiler.analysis.object_profiling import measure_objects


class ImageDataset:
    """Central data structure for managing multi-well plate image datasets."""

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
        subset_pattern: Optional[str] = None,
        img_shape: Optional[tuple] = None,
        img_dtype: Optional[type] = None
    ) -> None:
        self.measurement_dir = Path(measurement_dir)
        self.image_pattern = image_pattern or self.DEFAULT_IMAGE_PATTERN
        self.mask_pattern = mask_pattern or self.DEFAULT_MASK_PATTERN
        self.subset_pattern = subset_pattern
        self._img_shape: Optional[tuple] = img_shape
        self._img_dtype: Optional[type] = img_dtype
        
        self.meta_dict: Optional[Dict] = None
        self._metadata: Optional[pd.DataFrame] = None
        self._intensity_colnames: Optional[List[str]] = None
        self._mask_colnames: Optional[List[str]] = None

        self.build_metadata()
        self._auto_detect_image_properties()
        print(self.__repr__())

    def __repr__(self) -> str:
        return (
            f"\n--------- ImageDataset ----------\n"
            f" - dir: {self.measurement_dir!r}\n"
            f" - intensity_colnames: {self._intensity_colnames}\n"
            f" - mask_colnames: {self._mask_colnames}\n"
            f" - img_shape: {self._img_shape}\n"
            f" - img_dtype: {self._img_dtype}\n"
            f" - rows: {len(self.metadata) if self._metadata is not None else 0}"
            f"\n-----------------------------------\n"
        )

    def __len__(self) -> int:
        return len(self.metadata)

    def __iter__(self) -> Iterator[tuple]:
        for idx in range(len(self)):
            yield self.get_imageset(idx)

    @property
    def metadata(self) -> pd.DataFrame:
        return self._metadata

    @property
    def intensity_colnames(self) -> List[str]:
        return self._intensity_colnames or []

    @property
    def mask_colnames(self) -> List[str]:
        return self._mask_colnames or []

    @property
    def dataset_dir(self) -> Path:
        return self.measurement_dir

    @property
    def img_shape(self) -> Optional[tuple]:
        return self._img_shape

    @img_shape.setter
    def img_shape(self, value: Optional[tuple]) -> None:
        self._img_shape = tuple(value) if value is not None else None

    @property
    def img_dtype(self) -> Optional[type]:
        return self._img_dtype

    @img_dtype.setter
    def img_dtype(self, value: Optional[type]) -> None:
        self._img_dtype = value

    def _auto_detect_image_properties(self) -> None:
        row = self._metadata.iloc[0]
        img_path = Path(row["directory"]) / row[self._intensity_colnames[0]]
        # print(img_path)
        first_image = iio.imread(img_path)
        detected_shape = first_image.shape
        detected_dtype = first_image.dtype
        h, w = detected_shape[0], detected_shape[1]
        dtype_mapping = {
            np.dtype('uint8'): np.uint8,
            np.dtype('uint16'): np.uint16,
            np.dtype('float32'): np.float32,
            np.dtype('float64'): np.float64,
        }
        self._img_shape = (h, w)
        self._img_dtype = dtype_mapping.get(detected_dtype, np.uint16)

    def build_metadata(self, image_subdir='Images', remove_na_row=True) -> None:
        self.meta_dict = images_to_dataset(
            self.measurement_dir,
            self.image_pattern,
            self.mask_pattern,
            self.subset_pattern,
            image_subdir,
            remove_na_row
        )
        self._metadata = self.meta_dict.get('metadata')
        self._intensity_colnames = self.meta_dict.get('intensity_colnames', [])
        self._mask_colnames = self.meta_dict.get('mask_colnames', [])


    def get_imageset(self, row_idx: int, channels: List[str] = None, masks: List[str] = None) -> tuple:
        row = self.metadata.iloc[row_idx]

        image_paths = []
        channels = self.intensity_colnames if channels is None else channels
        channels = [channels] if isinstance(channels, str) else channels
        for ch_col in channels:
            image_paths.append(Path(row["directory"]) / row[ch_col])

        mask_paths = {}
        masks = self.mask_colnames if masks is None else masks
        masks = [masks] if isinstance(masks, str) else masks
        for mask_col in masks:
            mask_name = mask_col.replace("mask_", "")
            mask_paths[mask_name] = Path(row["directory"]) / row[mask_col]

        images = []
        for path in image_paths:
            img = iio.imread(path)
            img = self._zoom_resize_image(img, is_mask=False)
            images.append(img)

        # Stack images as (Y, X, C) format - channels last
        image_data = np.stack(images, axis=-1)

        mask_data = {}
        for name, path in mask_paths.items():
            mask = iio.imread(path)
            mask = self._zoom_resize_image(mask, is_mask=True)
            mask_data[name] = mask

        return image_data, mask_data

    def _zoom_resize_image(self, image: np.ndarray, is_mask: bool = False) -> np.ndarray:
        target_h, target_w = self._img_shape
        current_h, current_w = image.shape

        zoom_h = target_h / current_h
        zoom_w = target_w / current_w

        order = 0 if is_mask else 1
        resized = ndimage.zoom(image, (zoom_h, zoom_w), order=order)

        return resized.astype(self._img_dtype)
    
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
        chan1 = chan1 or [self.intensity_colnames[0]]

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

        return summary

    def export_metadata(
        self,
        write_db: Union[bool, str, None] = True,
        table_name: str = "metadata"
    ):
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
        
        if isinstance(write_db, str):
            db_path = self.dataset_dir / write_db
        elif write_db:
            db_path = self.dataset_dir / "result.db"
        else:
            db_path = None
        
        if db_path:
            write_results_to_db(
                db_path,
                table_name,
                self.metadata,
                if_exists="replace",
            )

        return None

    def profile_image(
        self,
        channels: Optional[List[str]] = None,
        thresholds: Optional[Dict[str, float]] = None,
        row_idx: Optional[int] = None,
        write_db: Union[bool, str, None] = True,
        table_name: str = "image",
        max_workers: Optional[int] = None
    ) -> Optional[pd.DataFrame]:
        """Profile images at whole-image level."""
        if row_idx is None:
            indices = range(len(self.metadata))
        else:
            indices = [row_idx] if isinstance(row_idx, int) else row_idx

        results = []

        def process_row(idx):
            image_data, _ = self.get_imageset(idx)

            row = self.metadata.iloc[idx]
            metadata_row = row.to_dict()
            metadata_row = {
                k: v for k, v in metadata_row.items()
                if k not in self.mask_colnames + self.intensity_colnames
            }

            result = measure_image(
                image_data=image_data,
                channel_names=self.intensity_colnames,
                metadata_row=metadata_row,
                channels=channels,
                thresholds=thresholds,
            )
            result_df = pd.DataFrame([result])

            if isinstance(write_db, str):
                db_path = self.dataset_dir / write_db
            elif write_db:
                db_path = self.dataset_dir / "result.db"
            else:
                db_path = None

            if write_db:
                write_results_to_db(db_path, table_name, result_df, if_exists="append")
                return None
            else:
                return result_df

        if max_workers is None or len(indices) <= 1:
            for idx in tqdm(indices, desc="Profiling images"):
                result_df = process_row(idx)
                if result_df is not None:
                    results.append(result_df)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_idx = {executor.submit(process_row, idx): idx for idx in indices}
                for future in tqdm(as_completed(future_to_idx), total=len(indices), desc="Profiling images"):
                    result_df = future.result()
                    if result_df is not None:
                        results.append(result_df)

        return results if results else None

    def profile_object(
        self,
        mask_name: str,
        parent_mask_name: str = None,
        row_idx: Optional[Union[int, List[int]]] = None,
        write_db: Union[bool, str, None] = True,
        table_name: Optional[str] = None,
        max_workers: Optional[int] = None,
        intensity_channels: Optional[List[str]] = None,
        glcm_channels: Optional[List[str]] = None,
        glcm_distances: Optional[List[int]] = [2],
        granularity_channels: Optional[List[str]] = None,
        granularity_element_size: int = 10,
        granularity_scales: List[int] = [1, 3, 5, 7, 10],
        granularity_subsample_size: int = 128,
        radial_channels: Optional[List[str]] = None,
        radial_n_bins: int = 4,
        correlation_pairs: Optional[List[tuple]] = None,
    ) -> Optional[pd.DataFrame]:
        """Profile objects using the specified segmentation mask.
        
        Parameters
        ----------
        mask_name : str
            Name of the mask to use (e.g. "cell", "nuclei").
        parent_mask_name : str, optional
            Parent mask name for hierarchical child→parent assignment.
        row_idx : int or list of int, optional
            Profile a single row by index. None = profile all rows.
        write_db : bool or str or None
            True → write to result.db, str → write to custom path, None → return DataFrame.
        table_name : str, optional
            Table name. If None, use mask_name.
        max_workers : int, optional
            Maximum number of worker threads.
        intensity_channels : list of str, optional
            Channels to compute intensity features from.
        glcm_channels : list of str, optional
            Channels to compute GLCM features from.
        glcm_distances : list of int, optional
            GLCM distances. Default: [2].
        granularity_channels : list of str, optional
            Channels to compute granularity features from.
        granularity_element_size : int
            Element size for granularity. Default: 10.
        granularity_scales : list of int
            Spectrum scales for granularity. Default: [1, 3, 5, 7, 10].
        granularity_subsample_size : int
            Subsample size for granularity. Default: 128.
        radial_channels : list of str, optional
            Channels to compute radial distribution features from.
        radial_n_bins : int
            Number of radial bins. Default: 4.
        correlation_pairs : list of tuple, optional
            List of (channel1, channel2) tuples for correlation measurement.
        
        Returns
        -------
        pd.DataFrame or None
            Aggregated results when write_db is falsy, else None.
        """
        if row_idx is None:
            indices = range(len(self.metadata))
        else:
            indices = [row_idx] if isinstance(row_idx, int) else row_idx

        results = []

        def process_row(idx):
            row = self.metadata.iloc[idx]
            metadata_row = row.to_dict()
            metadata_row = {
                k: v for k, v in metadata_row.items()
                if k not in self.mask_colnames + self.intensity_colnames
            }
            
            image_data, mask_data = self.get_imageset(idx)
            
            # Get the main mask
            mask = mask_data[mask_name]
            
            # Get parent mask if parent_mask_name is provided
            parent_mask = None
            if parent_mask_name is not None:
                parent_mask = mask_data.get(parent_mask_name, None)

            # Build kwargs for measure_objects
            measure_kwargs = {}
            
            # Radial distribution kwargs
            if radial_channels is not None:
                measure_kwargs['radial_distribution_channels'] = radial_channels
                measure_kwargs['radial_distribution_kwargs'] = {'nbins': radial_n_bins}
            
            # Granularity kwargs
            if granularity_channels is not None:
                measure_kwargs['granularity_channels'] = granularity_channels
                measure_kwargs['granularity_kwargs'] = {
                    'scales': list(granularity_scales),
                    'subsample_size': granularity_subsample_size,
                    'element_size': granularity_element_size
                }
            
            # GLCM kwargs
            if glcm_channels is not None:
                measure_kwargs['glcm_channels'] = glcm_channels
                measure_kwargs['glcm_kwargs'] = {
                    'distances': glcm_distances if glcm_distances is not None else [1, 2, 3]
                }

            result_df = measure_objects(
                mask=mask,
                img=image_data,
                channel_names=self.intensity_colnames,
                metadata_row=metadata_row,
                parent_mask=parent_mask,
                parent_mask_name=parent_mask_name if parent_mask_name else "Parent",
                intensity_channels=intensity_channels,
                correlation_pairs=correlation_pairs,
                **measure_kwargs,
            )
            
            if isinstance(write_db, str):
                db_path = self.dataset_dir / write_db
            elif write_db:
                db_path = self.dataset_dir / "result.db"
            else:
                db_path = None

            if write_db and result_df is not None:
                current_table_name = mask_name if table_name is None else table_name
                write_results_to_db(db_path, current_table_name, result_df, if_exists="append")
                return None
            else:
                return result_df

        if max_workers is None or len(indices) <= 1:
            for idx in tqdm(indices, desc=f"Profiling {mask_name}"):
                result_df = process_row(idx)
                if result_df is not None:
                    results.append(result_df)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_idx = {executor.submit(process_row, idx): idx for idx in indices}
                for future in tqdm(as_completed(future_to_idx), total=len(indices), desc=f"Profiling {mask_name}"):
                    result_df = future.result()
                    if result_df is not None:
                        results.append(result_df)

        return results if results else None

    def crop_object(
        self,
        mask_name: str = None,
        row_idx: Optional[Union[int, List[int]]] = None,
        channels: list[str] = None,
        object_ids: list[int] = None,
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

            mask_path = row['directory'] / row[mask_name]

            img_paths = []
            channels = [channels] if isinstance(channels, str) else channels
            channels = self.intensity_colnames if channels is None else channels
            for ch in channels:
                img_paths.append(row['directory'] / row[ch])

            crops = crop_object(
                mask=mask_path,
                imgs=img_paths,
                object_ids=object_ids,
                scale_factor=scale_factor,
                target_size=target_size,
                clip_mask=clip_mask,
                pad_square=pad_square,
                rotate_horizontal=rotate_horizontal,
                expansion_pixel=expansion_pixel
            )

            for crop in crops:
                crop['metadata'] = row.to_dict() 
                crop['mask_name'] = mask_name

            results.extend(crops)

        return results

    def preprocess_basic_correction(
        self,
        mode: str = 'fit',
        channels: Optional[List[str]] = None,
        n_image: int = 50,
        working_size: int = 64,
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
        channels = channels or self.intensity_colnames

        if mode == 'fit' or mode == 'fit-transform':
            fit_basic_models(
                self.meta_dict, channels, n_image, working_size,
                enable_darkfield, output_root
            )
            
        if mode == 'transform' or mode == 'fit-transform':
            transform_basic_models(self.meta_dict, channels, output_root)
        
        return None

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

        self.image_pattern = self.image_pattern.replace(
            image_pattern_replace[0], image_pattern_replace[1]
        )
        self.mask_pattern = self.mask_pattern.replace(
            mask_pattern_replace[0], mask_pattern_replace[1]
        )
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

        self.image_pattern = self.image_pattern.replace(
            image_pattern_replace[0], image_pattern_replace[1]
        )
        self.mask_pattern = self.mask_pattern.replace(
            mask_pattern_replace[0], mask_pattern_replace[1]
        )
        print("Update image_pattern:", self.image_pattern)
        print("Update mask_pattern:", self.mask_pattern)
        self.preprocess_z_project_summary = summary

        self.build_metadata()

        return summary
# Image Profiler

A Python package for managing and analyzing microscopy image datasets, with support for segmentation, profiling, and preprocessing.

## Features

- **Dataset Management**: Organize multi-well plate image datasets with metadata parsing
- **Segmentation**: Run Cellpose-SAM segmentation on images
- **Profiling**: Compute image-level and object-level features
- **Preprocessing**: Apply BaSiC shading correction, tile images, and perform Z-stack projection
- **Cropping**: Extract individual cells from images using segmentation masks
- **Database Integration**: Save results to SQLite databases

## Installation

### From Source

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd image_profiler
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install the package in development mode:
   ```bash
   pip install -e .
   ```

## Quick Usage

```python
from image_profiler import ImageDataset

# initialization dataset
ds = ImageDataset('test_Measurement 1')

# pre-processing steps

## z projection over stacks
ds.preprocess_z_projection()

## split to small tiles
ds.preprocess_tile_image(540, 540)

## run flatfield correction with BaSiC
ds.preprocess_basic_correction('fit', n_image=20)
ds.preprocess_basic_correction('transform')

# or mannually defined and build metadata
# ds.image_pattern = 'r(?P<row>.*)c(?P<column>.*)f(?P<field>.*)p(?P<stack>.*)-ch(?P<channel>.*)sk(?P<timepoint>[0-9]{1,})fk1fl1_zpmax_tile(?P<tile>.*).tiff'
# ds.mask_pattern = 'r(?P<row>.*)c(?P<column>.*)f(?P<field>.*)p(?P<stack>.*)-ch(?P<channel>.*)sk(?P<timepoint>[0-9]{1,})fk1fl1_zpmax_tile(?P<tile>.*)_cp_masks_(?P<mask_name>.*).png'
# ds.build_metadata()

# segmentate on defined channels
ds.segmentate('cell', chan1='ch1', chan2=None)


# profiling

## profile over whole image
ds.profile_image(channels=['ch1', 'ch2'], 
                thresholds={'ch1': None, 'ch2': None},
                row_idx=None, write_db='result.db')

## profile over individual segmentated object
ds.profile_object(mask_name='cell', row_idx=None, channels=['ch1'], 
                profile=["shape", "intensity"], 
                write_db='result.db', table_name='cell')

## profile with extra properties over individual segmentated object
ds.profile_object(mask_name='cell', parent_mask_name=None, row_idx=None, channels=['ch1','ch2'], 
                profile=["intensity"], 
                extra_properties=['radial', 'granularity', 'glcm'],
                extra_properties_kwargs=[{'n_bins': 5}, {'background_radius': 20, 'spectrum_length': 10, 'subsample_size': 0.25}, {'distances': [2]}],
                write_db='result.db', table_name='cell2')


# crop single cell
crops = ds.crop_object(mask_name='cell', row_idx=[1], target_size=None, 
             clip_mask=True, pad_square=True, 
             rotate_horizontal=False, expansion_pixel=0)
crops[0]['cell_img'].shape
```

## Key Methods

- **preprocess_basic_correction()**: Apply BaSiC shading correction
- **preprocess_tile_image()**: Split images into tiles
- **preprocess_z_projection()**: Perform Z-stack projection
- **segmentate()**: Run Cellpose-SAM segmentation on the dataset
- **profile_image()**: Compute image-level features
- **profile_object()**: Compute object-level features using segmentation masks
- **export_metadata()**: Export metadata to result sqlite along with profile data
- **export_dataloader()**: Export metadata to CellProfiler-compatible CSV format
- **crop_cell()**: Extract individual cells from images


## Citations

This package builds upon the following projects:

- **BaSiCPy** - BaSiC illumination correction algorithm: [https://github.com/peng-lab/BaSiCPy](https://github.com/peng-lab/BaSiCPy)
- **CellProfiler** - Cell image analysis software: [https://github.com/CellProfiler/CellProfiler](https://github.com/CellProfiler/CellProfiler)
- **Cellpose** - Cell segmentation (Cellpose-SAM): [https://github.com/MouseLand/cellpose](https://github.com/MouseLand/cellpose)


## License

MIT License

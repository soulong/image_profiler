"""Example script demonstrating ImageDataset usage."""

from image_profiler import ImageDataset

measurement_dir = 'test_Measurement 1'
ds = ImageDataset(measurement_dir)

# ds.preprocess_z_projection()
# ds.preprocess_tile_image(540, 540)
# ds.preprocess_basic_correction('fit', n_image=20)
# ds.preprocess_basic_correction('transform')

# ds.image_pattern = 'r(?P<row>.*)c(?P<column>.*)f(?P<field>.*)p(?P<stack>.*)-ch(?P<channel>.*)sk(?P<timepoint>[0-9]{1,})fk1fl1_zpmax_tile(?P<tile>.*).tiff'
# ds.mask_pattern = 'r(?P<row>.*)c(?P<column>.*)f(?P<field>.*)p(?P<stack>.*)-ch(?P<channel>.*)sk(?P<timepoint>[0-9]{1,})fk1fl1_zpmax_tile(?P<tile>.*)_cp_masks_(?P<mask_name>.*).png'
# ds.build_metadata()

ds.segmentate('cell', chan1='ch1')

ds.export_metadata(write_db='result.db', table_name='metadata')

ds.profile_image(channels=['ch1', 'ch2'], 
                thresholds={'ch1': None, 'ch2': None},
                row_idx=None, write_db='result.db')

ds.profile_object(mask_name='cell', row_idx=None, channels=['ch1'], 
                profile=["shape", "intensity"], 
                write_db='result.db', table_name='cell')

ds.profile_object(mask_name='cell', row_idx=None, channels=['ch1','ch2'], 
                profile=["intensity"], 
                extra_properties=['radial', 'granularity', 'glcm'],
                extra_properties_kwargs=[{'n_bins': 5}, {'background_radius': 20, 'spectrum_length': 10, 'subsample_size': 0.25}, {'distances': [2]}],
                write_db='result.db', table_name='cell2')

# crop single cell
x = ds.crop_object(mask_name='cell', row_idx=[1], target_size=None, 
             clip_mask=True, pad_square=True, 
             rotate_horizontal=False, expansion_pixel=0)
x[0]['cell_img'].shape

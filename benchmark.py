#!/usr/bin/env python3
"""
Performance benchmark script for Image Profiler

This script measures the time taken to profile a dataset with and without multi-threading.
"""

import time
import numpy as np
from image_profiler import ImageDataset

# Path to the test dataset
test_dataset_path = "tests/test_Measurement 1"

def benchmark_profiling():
    """Benchmark the profiling performance with different thread counts."""
    print("=== Image Profiler Performance Benchmark ===")
    print(f"Testing with dataset: {test_dataset_path}")
    print()
    
    # Initialize the dataset
    ds = ImageDataset(test_dataset_path)
    print(f"Dataset loaded with {len(ds)} rows")
    print()
    
    # Test different thread counts
    thread_counts = [1, 2, 4, 8]
    
    for max_workers in thread_counts:
        print(f"Testing with {max_workers} thread(s)...")
        
        # Start timing
        start_time = time.time()
        
        # Run image profiling
        try:
            ds.profile_image(
                channels=ds.channels,
                thresholds=None,
                row_idx=None,
                write_db=False,
                max_workers=max_workers
            )
            elapsed_time = time.time() - start_time
            print(f"  Image profiling: {elapsed_time:.2f} seconds")
        except Exception as e:
            print(f"  Image profiling failed: {e}")
        
        # Run object profiling
        if 'cell' in ds.masks:
            start_time = time.time()
            try:
                ds.profile_object(
                    mask_name='cell',
                    row_idx=None,
                    channels=ds.channels,
                    profile=["shape", "intensity"],
                    write_db=False,
                    max_workers=max_workers
                )
                elapsed_time = time.time() - start_time
                print(f"  Object profiling: {elapsed_time:.2f} seconds")
            except Exception as e:
                print(f"  Object profiling failed: {e}")
        else:
            print("  Object profiling skipped: 'cell' mask not found")
        
        print()

if __name__ == "__main__":
    benchmark_profiling()

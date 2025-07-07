import os
import argparse
import numpy as np

from src.utils.read import read_bin
from src.utils.visualizers import (
    visualize_range_image,
    visualize_surface_encoding,
    visualize_unfit_encoding,
    visualize_stages,
    visualize_point_cloud
)
from src.range_image import to_range_image, to_point_cloud
from src.encoders import surface_encode, unfit_encode

# Based on typical Velodyne HDL-64E
H = 64
W = 1024
V_FOV = (-24.9, 2.0)
H_FOV = (0.0, 360.0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert KITTI LiDAR .bin file to a range image"
    )
    parser.add_argument(
        "--file-path", type=str, required=True, help="Path to the KITTI .bin LiDAR file"
    )
    args = parser.parse_args()

    file_path = args.file_path

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    print("Starting program")

    points = read_bin(file_path)
    print(f"Successfully read {file_path} data")

    range_image = to_range_image(points, H=H, W=W, v_fov=V_FOV, h_fov=H_FOV)
    print("Successfully completed range image conversion")
    
    surfaces, mask = surface_encode(range_image)
    print("Successfully completed surface encoding")

    unfit_image = np.where(mask, 0, range_image)
    quantized_coeffs, unfit_mask = unfit_encode(unfit_image, quantization_step=0.1)
    print("Successfully completed unfit data encoding")
    
    print("Showing stages for range image")
    visualize_stages(range_image, surfaces, unfit_image)

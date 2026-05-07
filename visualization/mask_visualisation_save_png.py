# import numpy as np
# import open3d as o3d
# import matplotlib.pyplot as plt
# from pathlib import Path
# import math
# import os

# def load_bin_file(bin_path):
#     """Load LiDAR point cloud from .bin file"""
#     points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
#     return points[:, :3]

# def load_mask_values(mask_path):
#     """Load mask values from text file"""
#     return np.loadtxt(mask_path)

# def export_mask_pngs(points, values, output_dir="mask_pngs"):
#     """Save all mask columns as PNGs with consistent view"""
#     output_dir = Path(output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)

#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points)

#     coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)

#     vis = o3d.visualization.Visualizer()
#     vis.create_window(visible=True)  # Headless rendering
#     vis.add_geometry(pcd)
#     vis.add_geometry(coord)

#     # Set consistent view
#     theta = math.radians(30)
#     front = [-math.cos(theta), 0, math.sin(theta)]
#     up = [0, 0, 1]
#     lookat = [0, 0, 0]
#     zoom = 0.02

#     vis.poll_events()
#     vis.update_renderer()

#     ctr = vis.get_view_control()
#     ctr.set_front(front)
#     ctr.set_up(up)
#     ctr.set_lookat(lookat)
#     ctr.set_zoom(zoom)

#     # Determine if single or multi-column
#     is_single_column = len(values.shape) == 1
#     num_columns = 1 if is_single_column else values.shape[1]

#     for i in range(num_columns):
#         if is_single_column:
#             mask_values = values
#         else:
#             mask_values = values[:, i]

#         normalized = (mask_values - mask_values.min()) / (mask_values.max() - mask_values.min() + 1e-8)
#         colors = plt.get_cmap("viridis")(normalized)[:, :3]
#         pcd.colors = o3d.utility.Vector3dVector(colors)

#         vis.update_geometry(pcd)
#         vis.poll_events()
#         vis.update_renderer()

#         img_path = output_dir / f"mask_{i:03d}.png"
#         vis.capture_screen_image(str(img_path))
#         print(f"Saved {img_path}")

#     vis.destroy_window()
#     print("\n✅ All mask images saved.")

# # === Update these paths ===
# bin_file_path = "/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/data/stu_dataset/validation/125/velodyne/000000.bin"
# mask_file_path = "/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/Mask4Former3D/query_masks_dump/125_000000_masks_arranged.txt"
# output_folder = "/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/mask_2_images_arranged"  # Or wherever you want

# # Load data
# points = load_bin_file(bin_file_path)
# mask_values = load_mask_values(mask_file_path)

# assert points.shape[0] == mask_values.shape[0], "Mismatch between point cloud and mask"

# # Export PNGs
# export_mask_pngs(points, mask_values, output_dir=output_folder)

#===============================================================================================

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from pathlib import Path
import math
import os
import argparse

def load_bin_file(bin_path):
    """Load LiDAR point cloud from .bin file"""
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return points[:, :3]

def load_mask_values(mask_path):
    """Load mask values from text file"""
    return np.loadtxt(mask_path)

def export_mask_pngs(points, values, output_dir="mask_pngs"):
    """Save all mask columns as PNGs with consistent view"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True)  # Headless rendering
    vis.add_geometry(pcd)
    vis.add_geometry(coord)

    # Set consistent view
    theta = math.radians(30)
    front = [-math.cos(theta), 0, math.sin(theta)]
    up = [0, 0, 1]
    lookat = [0, 0, 0]
    zoom = 0.02

    vis.poll_events()
    vis.update_renderer()

    ctr = vis.get_view_control()
    ctr.set_front(front)
    ctr.set_up(up)
    ctr.set_lookat(lookat)
    ctr.set_zoom(zoom)

    # Determine if single or multi-column
    is_single_column = len(values.shape) == 1
    num_columns = 1 if is_single_column else values.shape[1]

    for i in range(num_columns):
        if is_single_column:
            mask_values = values
        else:
            mask_values = values[:, i]

        normalized = (mask_values - mask_values.min()) / (mask_values.max() - mask_values.min() + 1e-8)
        colors = plt.get_cmap("viridis")(normalized)[:, :3]
        pcd.colors = o3d.utility.Vector3dVector(colors)

        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

        img_path = output_dir / f"mask_{i:03d}.png"
        vis.capture_screen_image(str(img_path))
        print(f"Saved {img_path}")

    vis.destroy_window()
    print("\n✅ All mask images saved.")

if __name__ == "__main__":
    # === Use command line arguments ===
    parser = argparse.ArgumentParser()
    parser.add_argument('--bin_file', type=str, required=True, help='Path to .bin file')
    parser.add_argument('--mask_file', type=str, required=True, help='Path to mask .txt file')
    parser.add_argument('--output_folder', type=str, required=True, help='Output folder for PNGs')
    args = parser.parse_args()

    # Load data
    points = load_bin_file(args.bin_file)
    mask_values = load_mask_values(args.mask_file)

    assert points.shape[0] == mask_values.shape[0], "Mismatch between point cloud and mask"

    # Export PNGs
    export_mask_pngs(points, mask_values, output_dir=args.output_folder)



'''
python /media/ubuntu22/HDD22T/_Sourabh/stu_dataset/My_Scripts/mask_visualisation_save_png.py \
    --bin_file "/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/Mask4Former3D/data/train/00/velodyne/000000.bin" \
    --mask_file "/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/Mask4Former3D/train_dataloader_dump/0_000000_query_masks.txt" \
    --output_folder "/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/mask_viz_large_scores_"

python /media/ubuntu22/HDD22T/_Sourabh/stu_dataset/My_Scripts/mask_visualisation_save_png.py \
    --bin_file "/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/data/stu_dataset/validation/125/velodyne/000000.bin" \
    --mask_file "query_masks_dump/125_000000_small_scores.txt" \
    --output_folder "/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/mask_viz_small_scores"

python /media/ubuntu22/HDD22T/_Sourabh/stu_dataset/My_Scripts/mask_visualisation_save_png.py \
    --bin_file "/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/data/stu_dataset/validation/125/velodyne/000000.bin" \
    --mask_file "query_masks_dump/125_000000_small_scores_inverted.txt" \
    --output_folder "/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/mask_viz_small_inverted"

python /media/ubuntu22/HDD22T/_Sourabh/stu_dataset/My_Scripts/mask_visualisation_save_png.py \
    --bin_file "/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/data/stu_dataset/validation/125/velodyne/000000.bin" \
    --mask_file "query_masks_dump/125_000000_combined_before_inv.txt" \
    --output_folder "/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/mask_viz_combined"

python /media/ubuntu22/HDD22T/_Sourabh/stu_dataset/My_Scripts/mask_visualisation_save_png.py \
    --bin_file "/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/data/stu_dataset/validation/125/velodyne/000000.bin" \
    --mask_file "query_masks_dump/125_000000_final_anomaly.txt" \
    --output_folder "/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/mask_viz_final"

#sorted queries
python /media/ubuntu22/HDD22T/_Sourabh/stu_dataset/My_Scripts/mask_visualisation_save_png.py \
    --bin_file "/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/data/stu_dataset/validation/125/velodyne/000000.bin" \
    --mask_file "/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/Mask4Former3D/query_masks_dump/125_000000_cumulative_after_41_size_56406.txt" \
    --output_folder "/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/mask_viz_final/125/000000_all_09"
'''

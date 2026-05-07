import numpy as np
import open3d as o3d
import os
import torch
import math


def load_point_cloud(scan_path):
    """Load LiDAR .bin file (x,y,z,intensity)"""
    scan = np.fromfile(scan_path, dtype=np.float32).reshape(-1, 4)
    return scan[:, :3]  # Return only x,y,z

# for nuscenes
# def load_point_cloud(scan_path):
#     """Load LiDAR .bin file (x,y,z,intensity, ring index)"""
#     scan = np.fromfile(scan_path, dtype=np.float32).reshape(-1, 5)
#     return scan[:, :3]  # Return only x,y,z

points = load_point_cloud("/media/ubuntu22/HDD22T/_Sourabh/PMF-master/data/semantic-kitti-fov/sequences/125/velodyne/000000.bin")
sem_preds = np.loadtxt("/media/ubuntu22/HDD22T/_Zhiran/3d_anomaly_segmentation/stu_dataset/Mask4Former3D/saved/2025-10-30_preds_ep_29/prediction_inv/139/sem_preds_000349.txt")
# label_file="/media/ubuntu22/HDD22T/_Sourabh/PMF-master/data/semantic-kitti-fov/sequences/125/labels/000000.label"
# sem_preds = np.loadtxt("/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/result/Result_on_clean_code_with_uq/Ensemble_1/2025-09-17_Ensemble1_allmethods_uq/prediction/139/sem_preds_000349.txt")
# rosbag
# points = load_point_cloud("/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/Mask4Former3D/data/test/202/velodyne/000000.bin")
# sem_preds = np.loadtxt("/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/Mask4Former3D/saved/2025-08-07_110226/prediction/202/sem_preds_000000.txt")


# stu
# points = load_point_cloud("/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/Mask4Former3D/data/test/141/velodyne/000000.bin")
# sem_preds = np.loadtxt("/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/Mask4Former3D/saved/2025-08-07_104651/prediction/141/sem_preds_000000.txt")

# 2. Define the color map (RGB 0-1)
COLOR_MAP = {
    0: [0, 0, 0],        # unlabeled (black)
    1: [0, 0, 1],        # car (blue)
    2: [1, 0, 0],        # bicycle (red)
    3: [0, 0, 1],        # motorcycle (blue)
    4: [1, 1, 0],        # truck (yellow)
    5: [1, 0, 1],        # other-vehicle (magenta)
    6: [0, 1, 1],        # person (cyan)
    7: [1, 0.5, 0],      # bicyclist (orange)
    8: [0.5, 0.5, 0.5],  # motorcyclist (gray)
    9: [1, 0, 1],        # road (purple)
    10: [0, 0.5, 0],     # parking (dark green)
    11: [0.5, 0, 0],     # sidewalk (dark red)
    12: [0, 0, 0.5],     # other-ground (dark blue)
    13: [1.0, 0.84, 0.0],       # building (light blue)
    14: [0, 0.5, 0.5],   # fence (teal)
    15: [0, 1, 0],       # vegetation (green)
    16: [0.55, 0.27, 0.07], # trunk (brown)
    17: [0, 1, 0.5],     # terrain (spring green)
    18: [1, 0.75, 0.8],  # pole (pink)
    19: [0.29, 0, 0.51]  # traffic-sign (indigo)
}

# 3. Create color array (unknown classes will remain black)
colors = np.zeros((len(sem_preds), 3))  # Initialize all to black
# colors = np.zeros((len(label_file), 3))

# Apply colors for known classes
for class_id, color in COLOR_MAP.items():
    mask = (sem_preds == class_id-1)
    # mask = (label_file == class_id-1)
    colors[mask] = color

# 4. Create and visualize point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

# === Add LIDAR coordinate axes ===
axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=2.0, origin=[0, 0, 0])  # size can be adjusted

# === Visualize with constant view ===
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Semantic Predictions with LIDAR Axes", width=800, height=600)
vis.add_geometry(pcd)
vis.add_geometry(axis)
vis.poll_events()
vis.update_renderer()

# Set constant view
ctr = vis.get_view_control()
theta = math.radians(30)  # adjust viewing angle
front = [-math.cos(theta), 0, math.sin(theta)]
up = [0, 0, 1]
lookat = [0, 0, 0]
zoom = 0.05

ctr.set_front(front)
ctr.set_up(up)
ctr.set_lookat(lookat)
ctr.set_zoom(zoom)

print("Constant view applied. Close window to exit.")
vis.run()
vis.destroy_window()

# # === Visualize ===
# o3d.visualization.draw_geometries(
#     [pcd, axis],
#     window_name="Semantic Predictions with LIDAR Axes",
#     width=800,
#     height=600,
#     point_show_normal=False
# )


from collections import Counter

# Convert sem_preds to int (in case it's float)
sem_preds = sem_preds.astype(int)

# Count each class occurrence
class_counts = Counter(sem_preds)

print("Semantic Class Distribution:")
for class_id_minus1, count in class_counts.items():
    class_id = class_id_minus1 + 1  # Because you used class_id-1 earlier
    color = COLOR_MAP.get(class_id, [0, 0, 0])
    print(f"Class ID {class_id:2d} | Count: {count:6d} | Color: {color}")


# import numpy as np
# import open3d as o3d

# def load_bin_file(bin_path):
#     """Load KITTI-style .bin point cloud files."""
#     points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)  # x,y,z,intensity
#     return points[:, :3]  # Return only XYZ (skip intensity)

# def visualize_with_axes(points, title="Point Cloud"):
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points)
#     coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
#     o3d.visualization.draw_geometries([pcd, coord_frame], window_name=title)

# # Load binary file correctly
# points = load_bin_file("/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/Mask4Former3D/data/test/202/velodyne/000100.bin")
# visualize_with_axes(points)
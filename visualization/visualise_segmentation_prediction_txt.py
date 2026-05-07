# This script visualises segmentation prediction
#   1) label file prediction
#   2) txt file prediction (below in 2nd script)
#==================================================================================000
# #Visualise label files prediction

# import numpy as np
# import open3d as o3d
# import os
# import torch
# import math
# from collections import Counter

# def load_point_cloud(scan_path):
#     """Load LiDAR .bin file (x,y,z,intensity)"""
#     scan = np.fromfile(scan_path, dtype=np.float32).reshape(-1, 4)
#     return scan[:, :3]  # Return only x,y,z

# def load_semantic_labels(label_path):
#     """Load SemanticKITTI .label files and extract semantic class"""
#     # Load 32-bit unsigned integers
#     labels = np.fromfile(label_path, dtype=np.uint32)
    
#     # Extract semantic class (lower 16 bits)
#     sem_labels = labels & 0xFFFF  # Take lower 16 bits for semantic class
#     return sem_labels

# def map_semantic_labels(sem_labels):
#     """Map SemanticKITTI official IDs to simplified 0-19 range"""
#     # Define the learning map
#     learning_map = {
#         0: 0,    # "unlabeled"
#         1: 0,    # "outlier" mapped to "unlabeled"
#         2: 0,    # "anomaly" mapped to "unlabeled"
#         10: 1,   # "car"
#         11: 2,   # "bicycle"
#         13: 5,   # "bus" mapped to "other-vehicle"
#         15: 3,   # "motorcycle"
#         16: 5,   # "on-rails" mapped to "other-vehicle"
#         18: 4,   # "truck"
#         20: 5,   # "other-vehicle"
#         30: 6,   # "person"
#         31: 7,   # "bicyclist"
#         32: 8,   # "motorcyclist"
#         40: 9,   # "road"
#         44: 10,  # "parking"
#         48: 11,  # "sidewalk"
#         49: 12,  # "other-ground"
#         50: 13,  # "building"
#         51: 14,  # "fence"
#         52: 0,   # "other-structure" mapped to "unlabeled"
#         60: 9,   # "lane-marking" to "road"
#         70: 15,  # "vegetation"
#         71: 16,  # "trunk"
#         72: 17,  # "terrain"
#         80: 18,  # "pole"
#         81: 19,  # "traffic-sign"
#         99: 0,   # "other-object" to "unlabeled"
#         252: 1,  # "moving-car" to "car"
#         253: 7,  # "moving-bicyclist" to "bicyclist"
#         254: 6,  # "moving-person" to "person"
#         255: 8,  # "moving-motorcyclist" to "motorcyclist"
#         256: 5,  # "moving-on-rails" mapped to "other-vehicle"
#         257: 5,  # "moving-bus" mapped to "other-vehicle"
#         258: 4,  # "moving-truck" to "truck"
#         259: 5,  # "moving-other"-vehicle to "other-vehicle"
#     }
    
#     # Create mapped labels array
#     mapped_labels = np.zeros_like(sem_labels)
    
#     for original_id, mapped_id in learning_map.items():
#         mask = (sem_labels == original_id)
#         mapped_labels[mask] = mapped_id
    
#     # Handle any unmapped classes (shouldn't happen with complete mapping)
#     unmapped_mask = (mapped_labels == 0) & (sem_labels != 0)
#     if np.any(unmapped_mask):
#         unmapped_classes = np.unique(sem_labels[unmapped_mask])
#         print(f"Warning: Found unmapped classes: {unmapped_classes}")
#         # Map them to unlabeled (0)
#         mapped_labels[unmapped_mask] = 0
    
#     return mapped_labels

# def visualize_semantic_point_cloud(points_path, labels_path):
#     """Main function to load and visualize semantic point cloud"""
    
#     # Load point cloud and labels
#     points = load_point_cloud(points_path)
#     sem_labels = load_semantic_labels(labels_path)
#     mapped_labels = map_semantic_labels(sem_labels)

#     # Define color map (for mapped classes 0-19)
#     COLOR_MAP = {
#         0: [0, 0, 0],        # unlabeled (black)
#         1: [0, 0, 1],        # car (blue)
#         2: [1, 0, 0],        # bicycle (red)
#         3: [0, 0, 1],        # motorcycle (blue)
#         4: [1, 1, 0],        # truck (yellow)
#         5: [1, 0, 1],        # other-vehicle (magenta)
#         6: [0, 1, 1],        # person (cyan)
#         7: [1, 0.5, 0],      # bicyclist (orange)
#         8: [0.5, 0.5, 0.5],  # motorcyclist (gray)
#         9: [1, 0, 1],        # road (purple)
#         10: [0, 0.5, 0],     # parking (dark green)
#         11: [0.5, 0, 0],     # sidewalk (dark red)
#         12: [0, 0, 0.5],     # other-ground (dark blue)
#         13: [1.0, 0.84, 0.0], # building (light blue/gold)
#         14: [0, 0.5, 0.5],   # fence (teal)
#         15: [0, 1, 0],       # vegetation (green)
#         16: [0.55, 0.27, 0.07], # trunk (brown)
#         17: [0, 1, 0.5],     # terrain (spring green)
#         18: [1, 0.75, 0.8],  # pole (pink)
#         19: [0.29, 0, 0.51]  # traffic-sign (indigo)
#     }

#     # Create color array using mapped labels
#     colors = np.zeros((len(mapped_labels), 3))  # Initialize all to black

#     # Apply colors for mapped classes
#     for mapped_id, color in COLOR_MAP.items():
#         mask = (mapped_labels == mapped_id)
#         colors[mask] = color

#     # Create and visualize point cloud
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points)
#     pcd.colors = o3d.utility.Vector3dVector(colors)

#     # Add LIDAR coordinate axes
#     axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
#         size=2.0, origin=[0, 0, 0])

#     # Visualize with constant view
#     vis = o3d.visualization.Visualizer()
#     vis.create_window(window_name="Semantic Labels with LIDAR Axes", width=800, height=600)
#     vis.add_geometry(pcd)
#     vis.add_geometry(axis)
#     vis.poll_events()
#     vis.update_renderer()

#     # Set constant view
#     ctr = vis.get_view_control()
#     theta = math.radians(30)
#     front = [-math.cos(theta), 0, math.sin(theta)]
#     up = [0, 0, 1]
#     lookat = [0, 0, 0]
#     zoom = 0.05

#     ctr.set_front(front)
#     ctr.set_up(up)
#     ctr.set_lookat(lookat)
#     ctr.set_zoom(zoom)

#     # Print class distribution for debugging
#     print("\nOriginal SemanticKITTI Class Distribution:")
#     original_counts = Counter(sem_labels)
#     for class_id, count in sorted(original_counts.items()):
#         print(f"Original ID {class_id:3d} | Count: {count:6d}")

#     print("\nMapped Class Distribution:")
#     mapped_counts = Counter(mapped_labels)
#     for mapped_id, count in sorted(mapped_counts.items()):
#         color = COLOR_MAP.get(mapped_id, [0, 0, 0])
#         color_name = {
#             0: "unlabeled", 1: "car", 2: "bicycle", 3: "motorcycle", 
#             4: "truck", 5: "other-vehicle", 6: "person", 7: "bicyclist",
#             8: "motorcyclist", 9: "road", 10: "parking", 11: "sidewalk",
#             12: "other-ground", 13: "building", 14: "fence", 15: "vegetation",
#             16: "trunk", 17: "terrain", 18: "pole", 19: "traffic-sign"
#         }.get(mapped_id, "unknown")
#         print(f"Mapped ID {mapped_id:2d} | {color_name:15} | Count: {count:6d} | Color: {color}")

#     print(f"\nTotal points: {len(points)}")
#     print("Close the visualization window to exit.")
#     vis.run()
#     vis.destroy_window()


# # =============================================================================
# # INPUT FILE PATHS - Modify these paths as needed
# # =============================================================================

# # Example file paths - replace with your actual files
# # points_path = "/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/Mask4Former3D/data/validation/08/velodyne/000000.bin"
# # labels_path = "/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/Mask4Former3D/data/validation/08/labels/000000.label"

# points_path = "/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/Mask4Former3D/data/test/125/velodyne/000000.bin"
# labels_path = "/media/ubuntu22/HDD22T/_Sourabh/PMF-master/experiments_semkitti_stu/PMF-SemanticKitti/log_SemanticKitti_PMFNet-resnet34_bs8-lr0.001_baseline_timestamp/Eval-SemanticKitti-PMFNet-best_IOU_model-noKNN-debug_timestamp/preds_all_methods/sequences/125/predictions/000000.label"

# # Alternative examples (commented out):
# # points_path = "/path/to/your/pointcloud.bin"
# # labels_path = "/path/to/your/labels.label"

# # Run the visualization
# if __name__ == "__main__":
#     visualize_semantic_point_cloud(points_path, labels_path)

#================================================================================================
# visualise semantic prediction .txt file

import numpy as np
import open3d as o3d
import os
import math
from collections import Counter

def load_point_cloud(scan_path):
    """Load LiDAR .bin file (x,y,z,intensity)"""
    scan = np.fromfile(scan_path, dtype=np.float32).reshape(-1, 4)
    return scan[:, :3]  # Return only x,y,z

def load_predictions_txt(pred_path):
    """Load predictions from .txt file (one class ID per line)"""
    predictions = np.loadtxt(pred_path, dtype=int)
    return predictions

def visualize_predicted_point_cloud(points_path, pred_path):
    """Main function to load and visualize predicted point cloud"""
    
    # Load point cloud and predictions
    points = load_point_cloud(points_path)
    pred_labels = load_predictions_txt(pred_path)
    
    # Verify that point cloud and predictions have same length
    if len(points) != len(pred_labels):
        print(f"Warning: Point cloud has {len(points)} points, but predictions have {len(pred_labels)} points")
        # Truncate to minimum length
        min_length = min(len(points), len(pred_labels))
        points = points[:min_length]
        pred_labels = pred_labels[:min_length]
        print(f"Using first {min_length} points for visualization")

    # Define color map (for classes 0-19)
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
        13: [1.0, 0.84, 0.0], # building (gold)
        14: [0, 0.5, 0.5],   # fence (teal)
        15: [0, 1, 0],       # vegetation (green)
        16: [0.55, 0.27, 0.07], # trunk (brown)
        17: [0, 1, 0.5],     # terrain (spring green)
        18: [1, 0.75, 0.8],  # pole (pink)
        19: [0.29, 0, 0.51]  # traffic-sign (indigo)
    }

    # Create color array using prediction labels
    colors = np.zeros((len(pred_labels), 3))  # Initialize all to black

    # Apply colors for predicted classes
    for class_id, color in COLOR_MAP.items():
        mask = (pred_labels == class_id - 1)
        colors[mask] = color

    # For any classes not in COLOR_MAP, assign random colors
    uncolored_mask = (np.linalg.norm(colors, axis=1) == 0) & (pred_labels != 0)
    if np.any(uncolored_mask):
        unique_uncolored = np.unique(pred_labels[uncolored_mask])
        print(f"Assigning random colors to unmapped classes: {unique_uncolored}")
        for class_id in unique_uncolored:
            if class_id not in COLOR_MAP:
                # Generate a random but consistent color
                np.random.seed(class_id)
                random_color = np.random.rand(3)
                mask = (pred_labels == class_id)
                colors[mask] = random_color
                print(f"  Class {class_id}: {random_color}")

    # Create and visualize point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Add LIDAR coordinate axes
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=2.0, origin=[0, 0, 0])

    # Visualize with constant view
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Predicted Semantic Classes", width=800, height=600)
    vis.add_geometry(pcd)
    vis.add_geometry(axis)
    vis.poll_events()
    vis.update_renderer()

    # Set constant view
    ctr = vis.get_view_control()
    theta = math.radians(30)
    front = [-math.cos(theta), 0, math.sin(theta)]
    up = [0, 0, 1]
    lookat = [0, 0, 0]
    zoom = 0.05

    ctr.set_front(front)
    ctr.set_up(up)
    ctr.set_lookat(lookat)
    ctr.set_zoom(zoom)

    # Print prediction distribution
    print("\nPredicted Class Distribution:")
    pred_counts = Counter(pred_labels)
    
    # Class name mapping for better readability
    CLASS_NAMES = {
        0: "unlabeled", 1: "car", 2: "bicycle", 3: "motorcycle", 
        4: "truck", 5: "other-vehicle", 6: "person", 7: "bicyclist",
        8: "motorcyclist", 9: "road", 10: "parking", 11: "sidewalk",
        12: "other-ground", 13: "building", 14: "fence", 15: "vegetation",
        16: "trunk", 17: "terrain", 18: "pole", 19: "traffic-sign"
    }
    
    for class_id, count in sorted(pred_counts.items()):
        color = COLOR_MAP.get(class_id, [0, 0, 0])
        class_name = CLASS_NAMES.get(class_id, f"unknown_{class_id}")
        print(f"Class ID {class_id-1:2d} | {class_name:15} | Count: {count:6d} | Color: {color}")

    print(f"\nTotal points: {len(points)}")
    print(f"Total predictions: {len(pred_labels)}")
    print("Close the visualization window to exit.")
    
    vis.run()
    vis.destroy_window()


# =============================================================================
# INPUT FILE PATHS - Modify these paths as needed
# =============================================================================

# Example file paths - replace with your actual files
points_path = "/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/Mask4Former3D/data/test/144/velodyne/000265.bin"
pred_path = "/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/Mask4Former3D/saved/2025-11-29_Final_modified_v3/prediction_sem_preds/144/000265.txt"

# Alternative examples (commented out):
# points_path = "/path/to/your/pointcloud.bin"
# pred_path = "/path/to/your/predictions.txt"

# Run the visualization
if __name__ == "__main__":
    visualize_predicted_point_cloud(points_path, pred_path)
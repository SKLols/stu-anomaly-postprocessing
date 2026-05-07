# ===================== This is for prediction file as .label format
# import numpy as np
# import open3d as o3d
# from matplotlib import cm
# from pathlib import Path
# import argparse
# from tqdm import tqdm
# from collections import defaultdict

# def load_point_cloud(pcd_file):
#     """Load .bin file (KITTI format)."""
#     points = np.fromfile(pcd_file, dtype=np.float32).reshape(-1, 4)
#     num_points = points.shape[0]
#     print(f"Loaded {num_points} points from {pcd_file}")
#     return points[:, :3], None  # XYZ only

# def load_prediction_labels(label_file):
#     """Load prediction .label file (uint32 format)."""
#     labels = np.fromfile(label_file, dtype=np.uint32)
#     print(f"Loaded {len(labels)} prediction labels from {label_file}")
#     return labels

# class PointOODVisualizer:
#     def __init__(self):
#         self.min_eval_distance = 2.5
#         self.max_eval_distance = 50

#         # Colour map - RGB
#         self.color_map = {
#             0 : [0, 0, 0],
#             1 : [0, 0, 255],
#             2 : [255, 0, 0],
#             10: [245, 150, 100],
#             11: [245, 230, 100],
#             13: [250, 80, 100],
#             15: [150, 60, 30],
#             16: [255, 0, 0],
#             18: [180, 30, 80],
#             20: [255, 0, 0],
#             30: [30, 30, 255],
#             31: [200, 40, 255],
#             32: [90, 30, 150],
#             40: [255, 0, 255],
#             44: [255, 150, 255],
#             48: [75, 0, 75],
#             49: [75, 0, 175],
#             50: [0, 200, 255],
#             51: [50, 120, 255],
#             52: [0, 150, 255],
#             60: [170, 255, 150],
#             70: [0, 175, 0],
#             71: [0, 60, 135],
#             72: [80, 240, 150],
#             80: [150, 240, 255],
#             81: [0, 0, 255],
#             99: [255, 255, 50],
#             252: [245, 150, 100],
#             256: [255, 0, 0],
#             253: [200, 40, 255],
#             254: [30, 30, 255],
#             255: [90, 30, 150],
#             257: [250, 80, 100],
#             258: [180, 30, 80],
#             259: [255, 0, 0]
#         }

#         # Convert BGR to RGB and normalize to [0,1]
#         self.normalized_color_map = {
#             k: [v[2]/255.0, v[1]/255.0, v[0]/255.0] 
#             for k, v in self.color_map.items()
#         }

#         self.learning_map = {
#             0: 0,    # "unlabeled"
#             1: 0,    # "outlier" -> "unlabeled"
#             2: 0,    # "anomaly" -> "unlabeled"
#             10: 1,   # "car"
#             11: 2,   # "bicycle"
#             13: 5,   # "bus" -> "other-vehicle"
#             15: 3,   # "motorcycle"
#             16: 5,   # "on-rails" -> "other-vehicle"
#             18: 4,   # "truck"
#             20: 5,   # "other-vehicle"
#             30: 6,   # "person"
#             31: 7,   # "bicyclist"
#             32: 8,   # "motorcyclist"
#             40: 9,   # "road"
#             44: 10,  # "parking"
#             48: 11,  # "sidewalk"
#             49: 12,  # "other-ground"
#             50: 13,  # "building"
#             51: 14,  # "fence"
#             52: 0,   # "other-structure" -> "unlabeled"
#             60: 9,   # "lane-marking" -> "road"
#             70: 15,  # "vegetation"
#             71: 16,  # "trunk"
#             72: 17,  # "terrain"
#             80: 18,  # "pole"
#             81: 19,  # "traffic-sign"
#             99: 0,   # "other-object" -> "unlabeled"
#             252: 1,  # "moving-car" -> "car"
#             253: 7,  # "moving-bicyclist" -> "bicyclist"
#             254: 6,  # "moving-person" -> "person"
#             255: 8,  # "moving-motorcyclist" -> "motorcyclist"
#             256: 5,  # "moving-on-rails" -> "other-vehicle"
#             257: 5,  # "moving-bus" -> "other-vehicle"
#             258: 4,  # "moving-truck" -> "truck"
#             259: 5   # "moving-other-vehicle" -> "other-vehicle"
#         }

#         # Color map for the 19 learned classes (0-19)
#         self.learned_color_map_v1 = {
#             0: [0, 0, 0],         # 0: unlabeled (black)
#             1: [100, 150, 245],   # 1: car (light blue)
#             2: [100, 230, 245],   # 2: bicycle (cyan)
#             3: [30, 60, 150],     # 3: motorcycle (dark blue)
#             4: [80, 30, 180],     # 4: truck (purple)
#             5: [0, 0, 255],       # 5: other-vehicle (blue)
#             6: [255, 30, 30],     # 6: person (red)
#             7: [255, 40, 200],    # 7: bicyclist (pink)
#             8: [150, 30, 90],     # 8: motorcyclist (dark pink)
#             9: [255, 0, 255],     # 9: road (magenta)
#             10: [255, 150, 255],  # 10: parking (light pink)
#             11: [75, 0, 75],      # 11: sidewalk (dark purple)
#             12: [175, 0, 75],     # 12: other-ground (red-purple)
#             13: [255, 200, 0],    # 13: building (gold)
#             14: [255, 120, 50],   # 14: fence (orange)
#             15: [0, 175, 0],      # 15: vegetation (green)
#             16: [135, 60, 0],     # 16: trunk (brown)
#             17: [150, 240, 80],   # 17: terrain (light green)
#             18: [255, 240, 150],  # 18: pole (light yellow)
#             19: [255, 0, 0]       # 19: traffic-sign (bright red)
#         }

#         self.learned_color_map = {
#             k: [v[0]/255.0, v[1]/255.0, v[2]/255.0] 
#             for k, v in self.learned_color_map_v1.items()
#         }

#     def _apply_distance_mask(self, points, labels):
#         """Filter labels based on distance (original logic)."""
#         distances = np.linalg.norm(points, axis=1)
#         labels = np.where(
#             (distances > self.max_eval_distance) | (distances < self.min_eval_distance),
#             -1,
#             labels,
#         )
#         return labels

#     def visualize_raw_semantic_labels(self, points, labels, window_name="Raw Semantic Labels"):
#         """Visualize original labels with exact colors from specification"""
#         labels = self._apply_distance_mask(points, labels)
        
#         colors = np.zeros((len(labels), 3))
#         for label, color in self.normalized_color_map.items():
#             mask = (labels == label)
#             colors[mask] = color
        
#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(points)
#         pcd.colors = o3d.utility.Vector3dVector(colors)

#         coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0)
#         o3d.visualization.draw_geometries([pcd, coord_frame], window_name=window_name)

#     def map_to_learned_classes(self, labels):
#         """Convert original labels to learned 19-class labels."""
#         learned_labels = np.zeros_like(labels)
#         for original_id, learned_id in self.learning_map.items():
#             learned_labels[labels == original_id] = learned_id
#         return learned_labels
    
#     def visualize_19_classes(self, points, labels, window_name="19-Class Segmentation"):
#         """Visualize only the 19 learned classes."""
#         # Map to learned classes first
#         learned_labels = self.map_to_learned_classes(labels)

#         # Apply distance mask
#         learned_labels = self._apply_distance_mask(points, learned_labels)
        
#         # Create colors array
#         colors = np.zeros((len(learned_labels), 3))
        
#         # Apply colors for each learned class
#         for class_id, color in self.learned_color_map.items():
#             mask = (learned_labels == class_id)
#             colors[mask] = color
        
#         # Create point cloud
#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(points)
#         pcd.colors = o3d.utility.Vector3dVector(colors)

#         coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0)
#         o3d.visualization.draw_geometries([pcd, coord_frame], window_name=window_name)

#     def visualize_predictions(self, points, pred_labels, window_name="Predictions"):
#         """Visualize prediction labels."""
#         # Handle length mismatch
#         if len(points) != len(pred_labels):
#             print(f"Warning: Points ({len(points)}) and predictions ({len(pred_labels)}) have different lengths!")
#             min_length = min(len(points), len(pred_labels))
#             points = points[:min_length]
#             pred_labels = pred_labels[:min_length]
#             print(f"Using first {min_length} points from both arrays")

#         # Apply distance mask
#         pred_labels = self._apply_distance_mask(points, pred_labels)
        
#         # Create colors array
#         colors = np.zeros((len(pred_labels), 3))
        
#         # Apply colors for each class in predictions
#         for class_id, color in self.learned_color_map.items():
#             mask = (pred_labels == class_id)
#             colors[mask] = color
        
#         # For classes not in learned_color_map, use gray
#         unknown_mask = ~np.isin(pred_labels, list(self.learned_color_map.keys()))
#         colors[unknown_mask] = [0.5, 0.5, 0.5]  # Gray for unknown classes
        
#         # Create point cloud
#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(points)
#         pcd.colors = o3d.utility.Vector3dVector(colors)

#         coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0)
#         o3d.visualization.draw_geometries([pcd, coord_frame], window_name=window_name)

# def visualize_single_frame(bin_file, pred_file=None):
#     """Visualize a specific frame with optional predictions."""
#     visualizer = PointOODVisualizer()
    
#     # Load point cloud
#     points, _ = load_point_cloud(bin_file)
    
#     if pred_file:
#         # Load and visualize predictions
#         pred_labels = load_prediction_labels(pred_file)
        
#         # Print prediction statistics
#         print(f"\nPrediction statistics for {pred_file.name}:")
#         unique_labels, counts = np.unique(pred_labels, return_counts=True)
#         for label, count in zip(unique_labels, counts):
#             percentage = count/len(pred_labels)*100
#             print(f"Class {label}: {count} points ({percentage:.2f}%)")
        
#         # Visualize predictions
#         visualizer.visualize_predictions(
#             points, pred_labels,
#             window_name=f"Predictions - {bin_file.name}"
#         )
        
#         # Also show with learned class mapping if needed
#         visualizer.visualize_19_classes(
#             points, pred_labels,
#             window_name=f"Predictions (19-Class) - {bin_file.name}"
#         )
#     else:
#         # If no predictions, just show the point cloud
#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(points)
#         pcd.paint_uniform_color([0.5, 0.5, 0.5])  # Gray color
        
#         coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0)
#         o3d.visualization.draw_geometries([pcd, coord_frame], 
#                                         window_name=f"Point Cloud - {bin_file.name}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Visualize point cloud with predictions")
#     parser.add_argument("--bin-file", type=Path, required=True, 
#                        help="Path to point cloud .bin file")
#     parser.add_argument("--pred-file", type=Path, 
#                        help="Path to prediction .label file (optional)")
    
#     args = parser.parse_args()

#     if not args.bin_file.exists():
#         print(f"Error: Point cloud file {args.bin_file} does not exist")
#         exit(1)
        
#     if args.pred_file and not args.pred_file.exists():
#         print(f"Error: Prediction file {args.pred_file} does not exist")
#         exit(1)

#     visualize_single_frame(args.bin_file, args.pred_file)

# # Usage examples:
# # python3 visualise_predictions.py --bin-file /path/to/pointcloud.bin
# # python3 /media/ubuntu22/HDD22T/_Sourabh/stu_dataset/My_Scripts/visualise_pmf_prediction_segmentation.py --bin-file /media/ubuntu22/HDD22T/_Sourabh/PMF-master/data/semantic-kitti-fov/sequences/125/velodyne/000000.bin --pred-file /media/ubuntu22/HDD22T/_Sourabh/PMF-master/experiments/PMF-SemanticKitti/log_SemanticKitti_PMFNet-resnet34_bs8-lr0.001_baseline_timestamp/Eval-SemanticKitti-PMFNet-best_IOU_model-noKNN-debug_timestamp/preds/sequences/125/predictions/000000.label
# #python3 /media/ubuntu22/HDD22T/_Sourabh/stu_dataset/My_Scripts/visualise_pmf_prediction_segmentation.py --bin-file /media/ubuntu22/HDD22T/_Sourabh/PMF-master/data/semantic-kitti-fov/sequences/08/velodyne/000000.bin --pred-file /media/ubuntu22/HDD22T/_Sourabh/PMF-master/experiments/PMF-SemanticKitti/log_SemanticKitti_PMFNet-resnet34_bs8-lr0.001_baseline_timestamp/Eval-SemanticKitti-PMFNet-best_IOU_model-noKNN-debug_timestamp/preds/sequences/08/predictions/000000.label

# # #================ the below version for.txt prediction files 

import numpy as np
import open3d as o3d
from matplotlib import cm
from pathlib import Path
import argparse
from tqdm import tqdm
from collections import defaultdict

def load_point_cloud(pcd_file):
    """Load .bin file (KITTI format)."""
    points = np.fromfile(pcd_file, dtype=np.float32).reshape(-1, 4)
    num_points = points.shape[0]
    print(f"Loaded {num_points} points from {pcd_file}")
    return points[:, :3], None  # XYZ only

def load_prediction_labels(label_file):
    """Load prediction .txt file (text format)."""
    labels = np.loadtxt(label_file, dtype=np.uint32)
    print(f"Loaded {len(labels)} prediction labels from {label_file}")
    return labels

class PointOODVisualizer:
    def __init__(self):
        self.min_eval_distance = 2.5
        self.max_eval_distance = 50

        # Colour map - RGB
        self.color_map = {
            0 : [0, 0, 0],
            1 : [0, 0, 255],
            2 : [255, 0, 0],
            10: [245, 150, 100],
            11: [245, 230, 100],
            13: [250, 80, 100],
            15: [150, 60, 30],
            16: [255, 0, 0],
            18: [180, 30, 80],
            20: [255, 0, 0],
            30: [30, 30, 255],
            31: [200, 40, 255],
            32: [90, 30, 150],
            40: [255, 0, 255],
            44: [255, 150, 255],
            48: [75, 0, 75],
            49: [75, 0, 175],
            50: [0, 200, 255],
            51: [50, 120, 255],
            52: [0, 150, 255],
            60: [170, 255, 150],
            70: [0, 175, 0],
            71: [0, 60, 135],
            72: [80, 240, 150],
            80: [150, 240, 255],
            81: [0, 0, 255],
            99: [255, 255, 50],
            252: [245, 150, 100],
            256: [255, 0, 0],
            253: [200, 40, 255],
            254: [30, 30, 255],
            255: [90, 30, 150],
            257: [250, 80, 100],
            258: [180, 30, 80],
            259: [255, 0, 0]
        }

        # Convert BGR to RGB and normalize to [0,1]
        self.normalized_color_map = {
            k: [v[2]/255.0, v[1]/255.0, v[0]/255.0] 
            for k, v in self.color_map.items()
        }

        self.learning_map = {
            0: 0,    # "unlabeled"
            1: 0,    # "outlier" -> "unlabeled"
            2: 0,    # "anomaly" -> "unlabeled"
            10: 1,   # "car"
            11: 2,   # "bicycle"
            13: 5,   # "bus" -> "other-vehicle"
            15: 3,   # "motorcycle"
            16: 5,   # "on-rails" -> "other-vehicle"
            18: 4,   # "truck"
            20: 5,   # "other-vehicle"
            30: 6,   # "person"
            31: 7,   # "bicyclist"
            32: 8,   # "motorcyclist"
            40: 9,   # "road"
            44: 10,  # "parking"
            48: 11,  # "sidewalk"
            49: 12,  # "other-ground"
            50: 13,  # "building"
            51: 14,  # "fence"
            52: 0,   # "other-structure" -> "unlabeled"
            60: 9,   # "lane-marking" -> "road"
            70: 15,  # "vegetation"
            71: 16,  # "trunk"
            72: 17,  # "terrain"
            80: 18,  # "pole"
            81: 19,  # "traffic-sign"
            99: 0,   # "other-object" -> "unlabeled"
            252: 1,  # "moving-car" -> "car"
            253: 7,  # "moving-bicyclist" -> "bicyclist"
            254: 6,  # "moving-person" -> "person"
            255: 8,  # "moving-motorcyclist" -> "motorcyclist"
            256: 5,  # "moving-on-rails" -> "other-vehicle"
            257: 5,  # "moving-bus" -> "other-vehicle"
            258: 4,  # "moving-truck" -> "truck"
            259: 5   # "moving-other-vehicle" -> "other-vehicle"
        }

        # Color map for the 19 learned classes (0-19)
        self.learned_color_map_v1 = {
            0: [0, 0, 0],         # 0: unlabeled (black)
            1: [100, 150, 245],   # 1: car (light blue)
            2: [100, 230, 245],   # 2: bicycle (cyan)
            3: [30, 60, 150],     # 3: motorcycle (dark blue)
            4: [80, 30, 180],     # 4: truck (purple)
            5: [0, 0, 255],       # 5: other-vehicle (blue)
            6: [255, 30, 30],     # 6: person (red)
            7: [255, 40, 200],    # 7: bicyclist (pink)
            8: [150, 30, 90],     # 8: motorcyclist (dark pink)
            9: [255, 0, 255],     # 9: road (magenta)
            10: [255, 150, 255],  # 10: parking (light pink)
            11: [75, 0, 75],      # 11: sidewalk (dark purple)
            12: [175, 0, 75],     # 12: other-ground (red-purple)
            13: [255, 200, 0],    # 13: building (gold)
            14: [255, 120, 50],   # 14: fence (orange)
            15: [0, 175, 0],      # 15: vegetation (green)
            16: [135, 60, 0],     # 16: trunk (brown)
            17: [150, 240, 80],   # 17: terrain (light green)
            18: [255, 240, 150],  # 18: pole (light yellow)
            19: [255, 0, 0]       # 19: traffic-sign (bright red)
        }

        self.learned_color_map = {
            k: [v[0]/255.0, v[1]/255.0, v[2]/255.0] 
            for k, v in self.learned_color_map_v1.items()
        }

    def _apply_distance_mask(self, points, labels):
        """Filter labels based on distance (original logic)."""
        distances = np.linalg.norm(points, axis=1)
        labels = np.where(
            (distances > self.max_eval_distance) | (distances < self.min_eval_distance),
            -1,
            labels,
        )
        return labels

    def visualize_raw_semantic_labels(self, points, labels, window_name="Raw Semantic Labels"):
        """Visualize original labels with exact colors from specification"""
        labels = self._apply_distance_mask(points, labels)
        
        colors = np.zeros((len(labels), 3))
        for label, color in self.normalized_color_map.items():
            mask = (labels == label)
            colors[mask] = color
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0)
        o3d.visualization.draw_geometries([pcd, coord_frame], window_name=window_name)

    def map_to_learned_classes(self, labels):
        """Convert original labels to learned 19-class labels."""
        learned_labels = np.zeros_like(labels)
        for original_id, learned_id in self.learning_map.items():
            learned_labels[labels == original_id] = learned_id
        return learned_labels
    
    def visualize_19_classes(self, points, labels, window_name="19-Class Segmentation"):
        """Visualize only the 19 learned classes."""
        # Map to learned classes first
        learned_labels = self.map_to_learned_classes(labels)

        # Apply distance mask
        learned_labels = self._apply_distance_mask(points, learned_labels)
        
        # Create colors array
        colors = np.zeros((len(learned_labels), 3))
        
        # Apply colors for each learned class
        for class_id, color in self.learned_color_map.items():
            mask = (learned_labels == class_id)
            colors[mask] = color
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0)
        o3d.visualization.draw_geometries([pcd, coord_frame], window_name=window_name)

    def visualize_predictions(self, points, pred_labels, window_name="Predictions"):
        """Visualize prediction labels."""
        # Handle length mismatch
        if len(points) != len(pred_labels):
            print(f"Warning: Points ({len(points)}) and predictions ({len(pred_labels)}) have different lengths!")
            min_length = min(len(points), len(pred_labels))
            points = points[:min_length]
            pred_labels = pred_labels[:min_length]
            print(f"Using first {min_length} points from both arrays")

        # Apply distance mask
        pred_labels = self._apply_distance_mask(points, pred_labels)
        
        # Create colors array
        colors = np.zeros((len(pred_labels), 3))
        
        # Apply colors for each class in predictions
        for class_id, color in self.learned_color_map.items():
            mask = (pred_labels == class_id)
            colors[mask] = color
        
        # For classes not in learned_color_map, use gray
        unknown_mask = ~np.isin(pred_labels, list(self.learned_color_map.keys()))
        colors[unknown_mask] = [0.5, 0.5, 0.5]  # Gray for unknown classes
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0)
        o3d.visualization.draw_geometries([pcd, coord_frame], window_name=window_name)

def visualize_single_frame(bin_file, pred_file=None):
    """Visualize a specific frame with optional predictions."""
    visualizer = PointOODVisualizer()
    
    # Load point cloud
    points, _ = load_point_cloud(bin_file)
    
    if pred_file:
        # Load and visualize predictions
        pred_labels = load_prediction_labels(pred_file)
        
        # Print prediction statistics
        print(f"\nPrediction statistics for {pred_file.name}:")
        unique_labels, counts = np.unique(pred_labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            percentage = count/len(pred_labels)*100
            print(f"Class {label}: {count} points ({percentage:.2f}%)")
        
        # Visualize predictions
        visualizer.visualize_predictions(
            points, pred_labels,
            window_name=f"Predictions - {bin_file.name}"
        )
        
        # Also show with learned class mapping if needed
        visualizer.visualize_19_classes(
            points, pred_labels,
            window_name=f"Predictions (19-Class) - {bin_file.name}"
        )
    else:
        # If no predictions, just show the point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.paint_uniform_color([0.5, 0.5, 0.5])  # Gray color
        
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0)
        o3d.visualization.draw_geometries([pcd, coord_frame], 
                                        window_name=f"Point Cloud - {bin_file.name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize point cloud with predictions")
    parser.add_argument("--bin-file", type=Path, required=True, 
                       help="Path to point cloud .bin file")
    parser.add_argument("--pred-file", type=Path, 
                       help="Path to prediction .txt file (optional)")
    
    args = parser.parse_args()

    if not args.bin_file.exists():
        print(f"Error: Point cloud file {args.bin_file} does not exist")
        exit(1)
        
    if args.pred_file and not args.pred_file.exists():
        print(f"Error: Prediction file {args.pred_file} does not exist")
        exit(1)

    visualize_single_frame(args.bin_file, args.pred_file)

# Usage examples:
# python3 visualise_predictions.py --bin-file /path/to/pointcloud.bin
# python3 /media/ubuntu22/HDD22T/_Sourabh/stu_dataset/My_Scripts/visualise_pmf_prediction_segmentation.py --bin-file /media/ubuntu22/HDD22T/_Sourabh/stu_dataset/Mask4Former3D/data/test/125/velodyne/000000.bin --pred-file /media/ubuntu22/HDD22T/_Sourabh/stu_dataset/Mask4Former3D/saved/2025-11-11_190847/prediction_msp/125/classes_000000.txt
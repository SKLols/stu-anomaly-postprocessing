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

def load_labels(label_file, return_raw=False, return_learned=False, labels_file = "labels_file.txt"):
    """Load .label file and convert to OOD format (-1=ignore, 0=inlier, 1=anomaly)."""
    visualizer = PointOODVisualizer()
    labels_raw = np.fromfile(label_file, dtype=np.uint32).astype(np.int32)
    
    labels = labels_raw & 0xFFFF  # Semantic labels
    instance_label = (labels_raw >> 16)

    print("labels:",instance_label)
    with open(labels_file, 'w') as f:
        f.write("labels:n")
        np.savetxt(f, instance_label, fmt='%d', header='Label Values', comments='')

    if return_raw:
        return labels, instance_label
    else:
        if visualizer is None:
            raise ValueError("visualizer parameter required when return_learned=True")
        return visualizer.map_to_learned_classes(labels), instance_label

class PointOODVisualizer:
    def __init__(self):
        self.min_eval_distance = 2.5
        self.max_eval_distance = 50

        #Colour map - RGB
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
        learned_labels = labels

        # Apply distance mask
        learned_labels = self._apply_distance_mask(points, learned_labels)
        
        # Create colors array
        colors = np.zeros((len(learned_labels), 3))
        
        # Apply colors for each learned class
        for class_id, color in self.learned_color_map.items():
            #print("Color being applied to class 9:", self.learned_color_map[9])
            mask = (learned_labels == class_id)
            colors[mask] = color
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0)
        o3d.visualization.draw_geometries([pcd, coord_frame], window_name=window_name)

def visualize_single_frame(frame_path, pred_path=None):
    """Visualize a specific frame."""
    visualizer = PointOODVisualizer()
    
    points, _ = load_point_cloud(frame_path)
    #labels, _ = load_labels(frame_path.parent.parent / "labels" / f"{frame_path.stem}.label",return_raw=False)
    raw_labels, _ = load_labels(frame_path.parent.parent / "labels" / f"{frame_path.stem}.label", return_raw=True)
    learned_labels, _ = load_labels(frame_path.parent.parent / "labels" / f"{frame_path.stem}.label", return_learned=True)
    visualizer.visualize_19_classes(points, learned_labels, 
                                   window_name=f"19-Class Segmentation - {frame_path.name}")

    
    # Print label statistics
    print(f"\nLabel for learned_labels {frame_path.name}:")
    unique_labels, counts = np.unique(learned_labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"Label {label}: {count} points ({count/len(learned_labels)*100:.2f}%)")
    
    
    # Print label statistics
    print(f"\nLabel for raw_labels {frame_path.name}:")
    unique_labels, counts = np.unique(raw_labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"Label {label}: {count} points ({count/len(raw_labels)*100:.2f}%)")
    
    # Visualize with original colors
    visualizer.visualize_raw_semantic_labels(
        points, raw_labels,
        window_name=f"Raw Labels - {frame_path.name}"
    )

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize OOD labels with anomaly detection")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--scan", action="store_true", help="Scan and show top anomaly frames")
    group.add_argument("--frame", type=Path, help="Visualize specific frame (e.g., 'sequences/00/velodyne/000000.bin')")
    
    parser.add_argument("--data-dir", type=Path, help="Path to dataset (required for scan mode)")
    parser.add_argument("--pred-dir", type=Path, help="Path to prediction files (optional)")
    args = parser.parse_args()

    #Max_unique_labels = find_max_unique_label_frames(args.data_dir)

    if args.scan:
        if not args.data_dir:
            parser.error("--data-dir required for scan mode")
        #analyze_void_distribution(args.data_dir)
        #find_max_unique_label_frames(args.data_dir)
        #visualize_top_anomalies(args.data_dir, args.pred_dir)
    else:
        pred_file = None
        if args.pred_dir:
            seq_name = args.frame.parent.parent.name
            pred_file = args.pred_dir / seq_name / f"{args.frame.stem}.txt"
        visualize_single_frame(args.frame, pred_file)



#visualise a specific frame
#python3 visualise_segmentation_gt.py --frame /media/ubuntu22/HDD22T/_Sourabh/PMF-master/data/semantic-kitti-fov/sequences/08/velodyne/000000.bin

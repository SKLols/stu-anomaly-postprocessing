import numpy as np
import open3d as o3d
from matplotlib import cm
from pathlib import Path
import argparse
from tqdm import tqdm
from collections import defaultdict
import math

def load_point_cloud(pcd_file):
    """Load .bin file (KITTI format)."""
    points = np.fromfile(pcd_file, dtype=np.float32).reshape(-1, 4)
    num_points = points.shape[0]
    print(f"Loaded {num_points} points from {pcd_file}")
    return points[:, :3], None  # XYZ only


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
        self.learned_color_map = {
            0: [0, 0, 0],         # 0: unlabeled (black)
            1: [255, 0, 0],       # 1: car (red)
            2: [0, 255, 0],       # 2: bicycle (green)
            3: [0, 0, 255],       # 3: motorcycle (blue)
            4: [255, 255, 0],     # 4: truck (yellow)
            5: [255, 0, 255],     # 5: other-vehicle (magenta)
            6: [0, 255, 255],     # 6: person (cyan)
            7: [255, 128, 0],     # 7: bicyclist (orange)
            8: [128, 128, 128],     # 8: motorcyclist (gray)
            9: [255, 0, 255],   # 9: road (purple)
            10: [0, 128, 0],      # 10: parking (dark green)
            11: [128, 0, 0],      # 11: sidewalk (dark red)
            12: [0, 0, 128],      # 12: other-ground (dark blue)
            13: [0, 255, 0],    # 13: building (orange)
            14: [0, 128, 128],    # 14: fence (teal)
            15: [255, 0, 0],    # 15: vegetation (lime green)
            16: [139, 69, 19],    # 16: trunk (brown)
            17: [0, 255, 127],    # 17: terrain (spring green)
            18: [255, 192, 203],  # 18: pole (pink)
            19: [75, 0, 130]      # 19: traffic-sign (indigo)
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

    def count_anomalies(self, points, labels):
        """Count valid anomaly points (label=1 within distance range)."""
        labels = self._apply_distance_mask(points, labels)
        return np.sum(labels == 1)

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

    def visualize_predictions(self,points, scores):
        """Visualize anomaly scores (jet colormap)."""

        # Create dummy labels (all zeros) just to use the distance mask function
        dummy_labels = np.zeros(len(points))
        masked_labels = self._apply_distance_mask(points, dummy_labels)

        # Apply the same mask to points and scores
        mask = (masked_labels != -1)  # Points within distance range
        points = points[mask]
        scores = scores[mask]    

        #normalized_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        #normalized_scores = (scores.max() - scores) / (scores.max() - scores.min() + 1e-8)
        normalized_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        jet = cm.get_cmap("jet")
        score_colors = jet(normalized_scores)[:, :3]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(score_colors)

        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)

        # === Visualize with constant view ===
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Semantic Predictions with LIDAR Axes", width=800, height=600)
        vis.add_geometry(pcd)
        vis.add_geometry(coord_frame)
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

def load_label_file(label_file):
    """Load predictions from .label file (uint32 format)"""
    labels = np.fromfile(label_file, dtype=np.uint32)
    print(f"Loaded {len(labels)} labels from {label_file}")
    return labels

def visualize_single_frame(points, scores):
    """Visualize a specific frame."""
    visualizer = PointOODVisualizer()
    visualizer.visualize_predictions(points, scores)
    
def summarize_points(scores, threshold=None):
        total_points = len(scores)
        min_score, max_score = scores.min(), scores.max()
        mean_score = scores.mean()

        print(f"Total points: {total_points}")
        print(f"Score range: {min_score:.4f} – {max_score:.4f}, mean = {mean_score:.4f}")

        if threshold is not None:
            anomalies = np.sum(scores > threshold)
            inliers = total_points - anomalies
            print(f"Anomaly threshold = {threshold}")
            print(f"Anomaly points : {anomalies} ({100*anomalies/total_points:.2f}%)")
            print(f"Inlier points  : {inliers} ({100*inliers/total_points:.2f}%)")


if __name__ == "__main__":
    Seq = 0
    Frame =12
    points, _ = load_point_cloud(f"/media/ubuntu22/HDD22T/_Zhiran/3d_anomaly_segmentation/stu_dataset/Mask4Former3D/data/test/139/velodyne/000349.bin")
    
    # Changed from .txt to .label files
    # scores = np.loadtxt(f"/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/Mask4Former3D/saved/2025-09-17_004130/prediction/{Seq}/{Frame:06d}.txt", dtype=np.float32)
    # scores = np.loadtxt(f"/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/Mask4Former3D/saved/2025-09-17_004130/prediction_energy/{Seq}/{Frame:06d}.txt", dtype=np.float32)
    # scores = np.loadtxt(f"/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/Mask4Former3D/saved/2025-09-17_004130/prediction_energy_calibrated/{Seq}/{Frame:06d}.txt", dtype=np.float32)
    ##ToDO# scores = np.load(f"/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/Mask4Former3D/saved/2025-09-17_004130/prediction_ensemble/{Seq}/{Frame:06d}.npy").astype(np.float32)
    # scores = np.loadtxt(f"/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/Mask4Former3D/saved/2025-09-17_004130/prediction_maxlogit_raw/{Seq}/{Frame:06d}.txt", dtype=np.float32)
    # scores = np.loadtxt(f"/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/Mask4Former3D/saved/2025-09-17_004130/prediction_rba/{Seq}/{Frame:06d}.txt", dtype=np.float32)
    # scores = np.loadtxt(f"/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/Mask4Former3D/saved/2025-09-17_004130/prediction_sml/{Seq}/{Frame:06d}.txt", dtype=np.float32)
    # scores = np.loadtxt(f"/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/Mask4Former3D/saved/2025-09-17_004130/prediction_sml_calibrated/{Seq}/{Frame:06d}.txt", dtype=np.float32)
    
    # Using .label file instead of .txt
    scores = load_label_file(f"/media/ubuntu22/HDD22T/_Zhiran/3d_anomaly_segmentation/stu_dataset/Mask4Former3D/data/test/139/labels/000349.label")
    
    # If your .label files contain integer labels but you need float scores, convert them:
    # scores = scores.astype(np.float32)  # Convert uint32 to float32 if needed

    summarize_points(scores, threshold=0.8)
    visualize_single_frame(points, scores)

'''
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
            # Change from .txt to .label
            pred_file = args.pred_dir / seq_name / f"{args.frame.stem}.label"
        visualize_single_frame(args.frame, pred_file)




#visualise frames with max anomaly points
#python3 visualise_point_level_ood.py --scan --data-dir  data/stu_dataset/validation

#visualise frames with max anomaly oints and prediction
#python3 visualise_point_level_ood.py --scan --data-dir data/stu_dataset/validation --pred-dir result/2025-06-09_deepensemble00/prediction

#visualise a specific frame
#python3 visualise_point_level_ood.py --frame data/stu_dataset/validation/141/velodyne/000461.bin

#visualise specific file with prediction 
#python3 visualise_point_level_ood.py --frame data/stu_dataset/validation/141/velodyne/000461.bin --pred-dir result/2025-06-09_deepensemble00/prediction


'''
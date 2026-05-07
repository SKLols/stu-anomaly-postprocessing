'''

import numpy as np
import open3d as o3d
from pathlib import Path

class PointOODVisualizer:
    def __init__(self):
        print("Initialized!")
    
    def load_data(self, pcd_path, label_path):
        #Raw lidar data points
        points = np.fromfile(pcd_path,dtype=np.float32)
        points = points.reshape(-1,4) # The point cloud data is stored in a Nx4 format (x, y, z, intensity)
        points = points[:, :3] # Extracting the (x, y, z) coordinates
        intensities = scan[:, 3:]  

        #labels
        labels = np.fromfilr(label_path,dtype=np.uint32).astype(np.int32)
        semantic_label = labels & 0xFFFF
        instance_label = labels >> 16

        return points, intensities, semantic_label, instance_label
    
    def visualize_data(self, points, labels):
        print(f"Points shape: {points.shape}, Labels: {np.unique(labels)}")

if __name__ == "__main__":
    visualizer = PointOODVisualizer()
    points, labels = visualizer.load_data ("/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/data/stu_dataset/validation/125/velodyne/000001.bin","/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/data/stu_dataset/validation/125/labels/000001.label")
    visualizer.visualize_data(points, labels)

'''
#The below script works for sequence with green, black and purple
#================================================================================================================================================================
'''
import numpy as np
import open3d as o3d
from matplotlib import cm
from pathlib import Path
import argparse
from tqdm import tqdm

def load_point_cloud(pcd_file):
    """Load .bin file (KITTI format)."""
    points = np.fromfile(pcd_file, dtype=np.float32).reshape(-1, 4)
    return points[:, :3], None  # XYZ only

def load_labels(label_file):
    """Load .label file and convert to OOD format (-1=ignore, 0=inlier, 1=anomaly)."""
    labels = np.fromfile(label_file, dtype=np.uint32)
    labels = labels & 0xFFFF  # Semantic labels
    # Convert to OOD format (original logic)
    ood_labels = np.where(labels != 0, 0, -1)  # Non-zero -> 0 (inlier), 0 -> -1 (ignore)
    ood_labels = np.where(labels == 2, 1, ood_labels)  # 2 -> 1 (anomaly)
    return ood_labels, None

class PointOODVisualizer:
    def __init__(self):
        self.min_eval_distance = 2.5
        self.max_eval_distance = 50

    def _apply_distance_mask(self, points, labels):
        """Filter labels based on distance (original logic)."""
        distances = np.linalg.norm(points, axis=1)
        labels = np.where(
            (distances > self.max_eval_distance) | (distances < self.min_eval_distance),
            -1,
            labels,
        )
        return labels

    def visualize_ood_labels(self, points, labels):
        """Visualize labels: -1=black (ignore), 0=purple (inlier), 1=green (anomaly)."""
        # Apply distance mask (original logic)
        labels = self._apply_distance_mask(points, labels)
        
        # Assign colors
        colors = np.zeros((len(labels), 3))
        colors[labels == 0] = [0.5, 0, 0.5]  # Purple (inlier)
        colors[labels == 1] = [0, 1, 0]      # Green (anomaly)
        colors[labels == -1] = [0, 0, 0]      # Black (ignore)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0)
        o3d.visualization.draw_geometries([pcd, coord_frame], window_name="OOD Labels (GT)")

    def visualize_predictions(self, points, scores):
        """Visualize anomaly scores (jet colormap)."""
        normalized_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        jet = cm.get_cmap("jet")
        score_colors = jet(normalized_scores)[:, :3]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(score_colors)

        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0)
        o3d.visualization.draw_geometries([pcd, coord_frame], window_name="Anomaly Scores")

def main(args):
    visualizer = PointOODVisualizer()

    for seq_path in tqdm(sorted(list(args.data_dir.glob("1[0-9][0-9]")))):
        if seq_path.is_dir():
            lidar_files = sorted((seq_path / "velodyne").glob("*.bin"))

            for pcd_file in tqdm(lidar_files, leave=False, position=1):
                points, _ = load_point_cloud(pcd_file)
                labels, _ = load_labels(seq_path / "labels" / f"{pcd_file.stem}.label")

                # Visualize GT labels (-1/0/1 format)
                visualizer.visualize_ood_labels(points, labels)

                # (Optional) Visualize predictions
                if args.pred_dir:
                    pred_file = args.pred_dir / seq_path.name / f"{pcd_file.stem}.txt"
                    if pred_file.exists():
                        scores = np.loadtxt(pred_file, dtype=np.float32)
                        visualizer.visualize_predictions(points, scores)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize OOD labels (-1=ignore, 0=inlier, 1=anomaly)")
    parser.add_argument("--data-dir", type=Path, required=True, help="Path to dataset (KITTI format)")
    parser.add_argument("--pred-dir", type=Path, help="(Optional) Path to predicted anomaly scores")
    args = parser.parse_args()
    main(args)

'''
#Works for max anomaly findind, specific file ploting for GT
#==============================================================================================================================================

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
    return points[:, :3], None  # XYZ only

def load_labels(label_file):
    """Load .label file and convert to OOD format (-1=ignore, 0=inlier, 1=anomaly)."""
    labels = np.fromfile(label_file, dtype=np.uint32)
    labels = labels & 0xFFFF  # Semantic labels
    ood_labels = np.where(labels != 0, 0, -1)  # Non-zero -> 0 (inlier), 0 -> -1 (ignore)
    ood_labels = np.where(labels == 2, 1, ood_labels)  # 2 -> 1 (anomaly)
    return ood_labels, None

class PointOODVisualizer:
    def __init__(self):
        self.min_eval_distance = 2.5
        self.max_eval_distance = 50

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

    def visualize_ood_labels(self, points, labels, window_name="OOD Labels (GT)"):
        """Visualize labels: -1=black (ignore), 0=purple (inlier), 1=green (anomaly)."""
        labels = self._apply_distance_mask(points, labels)
        colors = np.zeros((len(labels), 3))
        colors[labels == 0] = [0.5, 0, 0.5]  # Purple (inlier)
        colors[labels == 1] = [0, 1, 0]       # Green (anomaly)
        colors[labels == -1] = [0, 0, 0]      # Black (ignore)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0)
        o3d.visualization.draw_geometries([pcd, coord_frame], window_name=window_name)

    def visualize_predictions(self, points, scores):
        """Visualize anomaly scores (jet colormap)."""

        # Create dummy labels (all zeros) just to use the distance mask function
        dummy_labels = np.zeros(len(points))
        masked_labels = self._apply_distance_mask(points, dummy_labels)

        # Apply the same mask to points and scores
        mask = (masked_labels != -1)  # Points within distance range
        points = points[mask]
        scores = scores[mask]    

        normalized_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        jet = cm.get_cmap("jet")
        score_colors = jet(normalized_scores)[:, :3]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(score_colors)

        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0)
        o3d.visualization.draw_geometries([pcd, coord_frame], window_name="Anomaly Scores (Predictions)")

def analyze_void_distribution(data_dir, output_txt="void_stats.txt", class_void=-1):
    """Scan all sequences to quantify void regions per frame and save to TXT."""
    visualizer = PointOODVisualizer()
    void_stats = []
    
    print("Scanning sequences for void distribution...")
    for seq_path in tqdm(sorted(data_dir.glob("1[0-9][0-9]"))):  # Assuming sequence folders are 100-199
        if seq_path.is_dir():
            #label_files = sorted((seq_path / "labels").glob("*.label"))
            lidar_files = sorted((seq_path / "velodyne").glob("*.bin"))
            #points, _ = load_point_cloud(frame_path)
            #labels, _ = load_labels(frame_path.parent.parent / "labels" / f"{frame_path.stem}.label")
            
            for pcd_file in lidar_files:
                points, _ = load_point_cloud(pcd_file)
                labels, _ = load_labels(seq_path / "labels" / f"{pcd_file.stem}.label")
                # Load labels (adjust based on your label format)
                final_labels = visualizer._apply_distance_mask(points, labels)  # Shape: [N_points]
                
                # Count void points
                void_points = np.sum(final_labels == class_void)
                total_points = len(labels)
                void_ratio = (void_points / total_points) * 100
                
                # Format: "sequence frame_id void_points void_ratio"
                void_stats.append(f"{seq_path.name} {pcd_file.stem} {void_points} {void_ratio:.4f}\n")
    
    # Save to file
    with open(output_txt, "w") as f:
        f.write("sequence frame void_pixels void_ratio\n")
        f.writelines(void_stats)
    print(f"Void statistics saved to {output_txt}")

def find_top_anomaly_frames(data_dir, top_n=2):
    """Scan all sequences to find frames with most anomalies."""
    visualizer = PointOODVisualizer()
    sequence_anomalies = defaultdict(list)

    print("Scanning sequences for frames with most anomalies...")
    for seq_path in tqdm(sorted(list(data_dir.glob("1[0-9][0-9]")))):
        if seq_path.is_dir():
            lidar_files = sorted((seq_path / "velodyne").glob("*.bin"))
            
            for pcd_file in lidar_files:
                points, _ = load_point_cloud(pcd_file)
                labels, _ = load_labels(seq_path / "labels" / f"{pcd_file.stem}.label")
                anomaly_count = visualizer.count_anomalies(points, labels)
                
                if anomaly_count > 0:
                    sequence_anomalies[seq_path.name].append((pcd_file, anomaly_count))
            
            # Sort by anomaly count and keep top N
            sequence_anomalies[seq_path.name].sort(key=lambda x: x[1], reverse=True)
            sequence_anomalies[seq_path.name] = sequence_anomalies[seq_path.name][:top_n]

    return sequence_anomalies

def visualize_top_anomalies(data_dir, pred_dir=None):
    """Visualize top anomaly frames from each sequence."""
    visualizer = PointOODVisualizer()
    top_frames = find_top_anomaly_frames(data_dir)

    for seq_name, frames in top_frames.items():
        print(f"\nSequence {seq_name} - Top anomaly frames:")
        for pcd_file, anomaly_count in frames:
            print(f"  {pcd_file.name}: {anomaly_count} anomalies")
            
            points, _ = load_point_cloud(pcd_file)
            labels, _ = load_labels(pcd_file.parent.parent / "labels" / f"{pcd_file.stem}.label")
            
            # Visualize with count in window title
            visualizer.visualize_ood_labels(
                points, labels, 
                window_name=f"{seq_name}/{pcd_file.name} ({anomaly_count} anomalies)"
            )
            
            if pred_dir:
                pred_file = pred_dir / seq_name / f"{pcd_file.stem}.txt"
                if pred_file.exists():
                    scores = np.loadtxt(pred_file, dtype=np.float32)
                    visualizer.visualize_predictions(points, scores)

def visualize_single_frame(frame_path, pred_path=None):
    """Visualize a specific frame."""
    visualizer = PointOODVisualizer()
    
    points, _ = load_point_cloud(frame_path)
    labels, _ = load_labels(frame_path.parent.parent / "labels" / f"{frame_path.stem}.label")

    final_labels = visualizer._apply_distance_mask (points, labels)

    total_points = len(final_labels)
    outlier_count = np.sum(final_labels == 1)
    inlier_count = np.sum(final_labels == 0)
    void_count = np.sum(final_labels == -1)

    outlier_percentage = (outlier_count / total_points) * 100
    inlier_percentage = (inlier_count / total_points) * 100
    void_percentage = (void_count / total_points) * 100

    print("Total_points : ", total_points)
    print("Outlier_count: ", outlier_count, "% :", outlier_percentage)
    print("Inlier_count: ", inlier_count, "% :", inlier_percentage)
    print("Void_count: ", void_count, "% :", void_percentage)
    
    anomaly_count = visualizer.count_anomalies(points, labels)
    print(f"Frame {frame_path.name} has {anomaly_count} anomalies")
    
    #print(f"Prediction path provided: {pred_path}")

    visualizer.visualize_ood_labels(
        points, labels,
        window_name=f"{frame_path.name} ({anomaly_count} anomalies)"
    )
    
    if pred_path and pred_path.exists():
        print(f"prediction path exists: {pred_path}")
        scores = np.loadtxt(pred_path, dtype=np.float32)

        #Create path for saving beside prediction file
        output_txt = pred_path.parent / f"{pred_path.stem}_accessed.txt"

        np.savetxt(output_txt,scores, fmt='%.6f')
        print(f"scores saved to {output_txt}")
        
        visualizer.visualize_predictions(points, scores)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize OOD labels with anomaly detection")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--scan", action="store_true", help="Scan and show top anomaly frames")
    group.add_argument("--frame", type=Path, help="Visualize specific frame (e.g., 'sequences/00/velodyne/000000.bin')")
    
    parser.add_argument("--data-dir", type=Path, help="Path to dataset (required for scan mode)")
    parser.add_argument("--pred-dir", type=Path, help="Path to prediction files (optional)")
    args = parser.parse_args()

    if args.scan:
        if not args.data_dir:
            parser.error("--data-dir required for scan mode")
        #analyze_void_distribution(args.data_dir)
        visualize_top_anomalies(args.data_dir, args.pred_dir)
    else:
        pred_file = None
        if args.pred_dir:
            seq_name = args.frame.parent.parent.name
            pred_file = args.pred_dir / seq_name / f"{args.frame.stem}.txt"
        visualize_single_frame(args.frame, pred_file)




#visualise frames with max anomaly points
#python3 visualise_point_level_ood.py --scan --data-dir  data/stu_dataset/validation

#visualise frames with max anomaly oints and prediction
#python3 visualise_point_level_ood.py --scan --data-dir data/stu_dataset/validation --pred-dir result/2025-06-09_deepensemble00/prediction

#visualise a specific frame
#python3 visualise_point_level_ood.py --frame data/stu_dataset/validation/141/velodyne/000461.bin

#visualise specific file with prediction 
#python3 visualise_point_level_ood.py --frame data/stu_dataset/validation/141/velodyne/000461.bin --pred-dir result/2025-06-09_deepensemble00/prediction




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
'''
def load_labels___(label_file, labels_file = "labels_file.txt"):
    """Load .label file and convert to OOD format (-1=ignore, 0=inlier, 1=anomaly)."""
    #visualizer = PointOODVisualizer()
    labels_raw = np.fromfile(label_file, dtype=np.uint32).astype(np.int32)
    
    labels = labels_raw & 0xFFFF  # Semantic labels
    instance_label = (labels_raw >> 16)

    print("labels:",instance_label)
    with open(labels_file, 'w') as f:
        f.write("labels:n")
        np.savetxt(f, instance_label, fmt='%d', header='Label Values', comments='')

x=load_labels___("/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/data/stu_dataset/validation/138/velodyne/000365.bin")
'''
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
    elif return_learned:
        if visualizer is None:
            raise ValueError("visualizer parameter required when return_learned=True")
        return visualizer.map_to_learned_classes(labels), instance_label
    else:
        ood_labels = np.where(labels != 0, 0, -1)  # Non-zero -> 0 (inlier), 0 -> -1 (ignore)
        ood_labels = np.where(labels == 2, 1, ood_labels)  # 2 -> 1 (anomaly)
        return ood_labels, instance_label

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
        '''
        unique_learned, counts = np.unique(learned_labels, return_counts=True)
        print("\n=== After Learning Map ===")
        for lbl, cnt in zip(unique_learned, counts):
            print(f"Class {lbl}: {cnt} points ({(cnt/len(points))*100:.1f}%)")
        '''

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

def find_max_unique_label_frames(data_dir, top_n=2, output_file="unique_label_stats.txt"):
    """Scan all sequences to find frames with most unique semantic labels."""
    visualizer = PointOODVisualizer()
    sequence_unique_labels = defaultdict(list)

    with open(output_file, 'w') as f:
        f.write("Sequence\tFrame\tNum_Unique_Labels\tUnique_Labels\n")

    print("Result will be saved to:{output_file}")
    for seq_path in tqdm(sorted(list(data_dir.glob("1[0-9][0-9]")))):
        if seq_path.is_dir():
            current_seq = seq_path.name
            lidar_files = sorted((seq_path / "velodyne").glob("*.bin"))
            
            # Open file in append mode once per sequence
            with open(output_file, 'a') as f:
                for pcd_file in lidar_files:
                    frame_num = pcd_file.stem
                    points, _ = load_point_cloud(pcd_file)
                    raw_labels, _ = load_labels(seq_path / "labels" / f"{frame_num}.label", return_raw=True)
                    
                    # Apply distance mask
                    labels = visualizer._apply_distance_mask(points, raw_labels)
                    
                    # Get unique labels within valid distance
                    valid_labels = labels[labels != -1]
                    unique_labels = np.unique(valid_labels)
                    num_unique = len(unique_labels)
                    
                    # Store results
                    sequence_unique_labels[current_seq].append((pcd_file, num_unique, unique_labels))
                    
                    # Write to file
                    unique_str = ','.join(map(str, sorted(unique_labels.tolist())))
                    f.write(f"{current_seq}\t{frame_num}\t{num_unique}\t{unique_str}\n")
            
            # Sort by number of unique labels and keep top N
            sequence_unique_labels[current_seq].sort(key=lambda x: x[1], reverse=True)
            sequence_unique_labels[current_seq] = sequence_unique_labels[current_seq][:top_n]

    return sequence_unique_labels

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
    labels, _ = load_labels(frame_path.parent.parent / "labels" / f"{frame_path.stem}.label",return_raw=False)
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




#visualise frames with max anomaly points
#python3 visualise_point_level_ood.py --scan --data-dir  data/stu_dataset/validation

#visualise frames with max anomaly oints and prediction
#python3 visualise_point_level_ood.py --scan --data-dir data/stu_dataset/validation --pred-dir result/2025-06-09_deepensemble00/prediction

#visualise a specific frame
#python3 /media/ubuntu22/HDD22T/_Sourabh/stu_dataset/My_Scripts/visualise_point_level_ood_GT_copy.py --frame data/stu_dataset/validation/141/velodyne/000306.bin

#visualise specific file with prediction 
#python3 visualise_point_level_ood.py --frame data/stu_dataset/validation/141/velodyne/000461.bin --pred-dir result/2025-06-09_deepensemble00/prediction



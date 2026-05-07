
import numpy as np
import open3d as o3d

def load_point_cloud(pcd_file):
    """Load .bin file (KITTI format)."""
    points = np.fromfile(pcd_file, dtype=np.float32).reshape(-1, 4)
    num_points = points.shape[0]
    print(f"Loaded {num_points} points from {pcd_file}")
    return points[:, :3], None  # XYZ only

def load_labels(label_file):
    """Load .label file and convert to OOD format (-1=ignore, 0=inlier, 1=anomaly)."""
    labels_raw = np.fromfile(label_file, dtype=np.uint32)
    labels = labels_raw & 0xFFFF  # Semantic labels
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
    
    def _visualize_with_constant_view(self, pcd, coord_frame, window_name="Visualization"):
        """Open3D visualizer with fixed view."""
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=window_name, width=800, height=600)
        vis.add_geometry(pcd)
        vis.add_geometry(coord_frame)
        vis.poll_events()
        vis.update_renderer()

        # Set constant view
        ctr = vis.get_view_control()
        theta = np.radians(30)  # viewing angle
        front = [-np.cos(theta), 0, np.sin(theta)]
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

    def visualize_semantic_labels(self, points, labels, window_name="Semantic Labels"):
        """Visualize labels with 19 class colors."""
        # Define the color map (RGB 0-1)
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
            12: [0, 0, 0.5],    # other-ground (dark blue)
            13: [1.0, 0.84, 0.0], # building (light blue)
            14: [0, 0.5, 0.5],  # fence (teal)
            15: [0, 1, 0],      # vegetation (green)
            16: [0.55, 0.27, 0.07], # trunk (brown)
            17: [0, 1, 0.5],    # terrain (spring green)
            18: [1, 0.75, 0.8], # pole (pink)
            19: [0.29, 0, 0.51]  # traffic-sign (indigo)
        }
        
        # Create color array (unknown classes will remain black)
        colors = np.zeros((len(labels), 3))  # Initialize all to black
        
        # Apply colors for known classes
        for class_id, color in COLOR_MAP.items():
            mask = (labels == class_id)
            colors[mask] = color

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0)
        o3d.visualization.draw_geometries([pcd, coord_frame], window_name=window_name)

    def count_anomalies(self, points, labels):
        """Count valid anomaly points (label=1 within distance range)."""
        labels = self._apply_distance_mask(points, labels)
        return np.sum(labels == 1)
    
    def visualize_ood_labels(self, points, labels, window_name="OOD Labels (GT)"):
        """Visualize labels: -1=black (ignore), 0=purple (inlier), 1=green (anomaly) with fixed view."""
        labels = self._apply_distance_mask(points, labels)
        colors = np.zeros((len(labels), 3))
        colors[labels == 0] = [0.5, 0, 0.5]  # Purple (inlier)
        colors[labels == 1] = [0, 1, 0]       # Green (anomaly)
        colors[labels == -1] = [0, 0, 0]      # Black (ignore)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)

        # Use fixed view visualizer
        self._visualize_with_constant_view(pcd, coord_frame, window_name=window_name)
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


def visualize_single_frame(points,label_file):
    """Visualize a specific frame."""
    visualizer = PointOODVisualizer()
    
    points, _ = load_point_cloud(points)
    labels, _ = load_labels(label_file)
    final_labels = visualizer._apply_distance_mask (points, labels)

    # # Print the raw labels before any processing
    # labels_raw = np.fromfile(label_file, dtype=np.uint32)
    # print("\nRaw labels from file (first 20):")
    # print(labels_raw[:20])

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
    print(f"Frame has {anomaly_count} anomalies")
    
    visualizer.visualize_semantic_labels(
        points, final_labels,
        window_name="Semantic Labels Visualization"
    )

    visualizer.visualize_ood_labels(
         points, labels,
         window_name=f"({anomaly_count} anomalies)"
     )

    

if __name__ == "__main__":

    points="/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/Mask4Former3D/data/test/125/velodyne/000000.bin"
    label_file="/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/Mask4Former3D/data/test/125/labels/000000.label"
    visualize_single_frame(points,label_file)


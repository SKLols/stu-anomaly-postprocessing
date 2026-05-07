import numpy as np
import open3d as o3d
import argparse
from matplotlib import cm, pyplot as plt
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


def load_point_cloud(pcd_file):
    points = np.fromfile(pcd_file, dtype=np.float32).reshape(-1, 4)
    print(f"Loaded {points.shape[0]} points from {pcd_file}")
    return points[:, :3]


class PointOODVisualizer:
    def __init__(self, min_distance=2.5, max_distance=50.0):
        self.min_eval_distance = min_distance
        self.max_eval_distance = max_distance

    def _apply_distance_mask(self, points):
        distances = np.linalg.norm(points, axis=1)
        return (distances >= self.min_eval_distance) & (distances <= self.max_eval_distance)

    def visualize_scores(self, points, scores, window_name="Anomaly Score Visualization"):
        mask = self._apply_distance_mask(points)
        points = points[mask]
        scores = scores[mask]

        normalized_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        jet = cm.get_cmap("jet")
        colors = jet(normalized_scores)[:, :3]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0)
        o3d.visualization.draw_geometries([pcd, coord_frame], window_name=window_name)

    def analyze_scores(self, scores):
        print("Anomaly Score Analysis:")
        print(f"  Max     : {np.max(scores):.6f}")
        print(f"  Min     : {np.min(scores):.6f}")
        print(f"  Mean    : {np.mean(scores):.6f}")
        print(f"  Median  : {np.median(scores):.6f}")
        print(f"  Std Dev : {np.std(scores):.6f}")

        plt.figure()
        plt.hist(scores, bins=20, edgecolor='black')
        plt.title("Anomaly Score Distribution")
        plt.xlabel("Score")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def main(bin_path, score_path, analyze=False):
    bin_path = Path(bin_path)
    score_path = Path(score_path)

    if not bin_path.exists():
        raise FileNotFoundError(f"Point cloud file not found: {bin_path}")
    if not score_path.exists():
        raise FileNotFoundError(f"Score file not found: {score_path}")

    points = load_point_cloud(bin_path)
    scores = np.loadtxt(score_path, dtype=np.float32)

    visualizer = PointOODVisualizer()
    if analyze:
        visualizer.analyze_scores(scores)

    visualizer.visualize_scores(points, scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize anomaly scores on point cloud")

    parser.add_argument(
        "--bin", 
        type=str, 
        default="/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/data/stu_dataset/validation/141/velodyne/000300.bin", 
        help="Path to .bin file (point cloud)"
    )
    parser.add_argument(
        "--score", 
        type=str, 
        default="/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/Mask4Former3D/saved/2025-08-19_155106/prediction_sml/141/000300.txt", 
        help="Path to .txt file (anomaly scores)"
    )
    parser.add_argument(
        "--analyze", 
        action="store_true", 
        help="If set, analyze score statistics"
    )

    args = parser.parse_args()
    main(args.bin, args.score, args.analyze)


#        default="/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/Mask4Former3D/saved/2025-07-14_151856/prediction/141/000320.txt", 

" nuscenes lidar has 5 data fields, KITTI and rosbag have 4 data fields"

import numpy as np
import open3d as o3d

def load_lidar_bin(bin_path):
    # Each point is (x, y, z, intensity) - float32
    point_cloud = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 5)
    xyz = point_cloud[:, :3]  # Discard intensity and ring index if present
    return xyz

def visualize_point_cloud(xyz):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    bin_file = "/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/Mask4Former3D/data/test/210/velodyne/000000.bin"  # Replace with actual path
    points = load_lidar_bin(bin_file)
    visualize_point_cloud(points)





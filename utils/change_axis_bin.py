import os
import numpy as np

def rotate_180_z(points):
    """Rotate point cloud 180 degrees around Z-axis (invert X and Y)."""
    rotated = points.copy()
    rotated[:, 0] = -rotated[:, 0]  # Invert X
    rotated[:, 1] = -rotated[:, 1]  # Invert Y
    return rotated

def process_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    bin_files = [f for f in os.listdir(input_folder) if f.endswith(".bin")]

    for file_name in bin_files:
        input_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder, file_name)

        # Load the bin file (assume x, y, z, intensity, ring)
        points = np.fromfile(input_path, dtype=np.float32).reshape(-1, 4)

        # Apply rotation
        rotated_points = rotate_180_z(points)

        # Save to new .bin file
        rotated_points.astype(np.float32).tofile(output_path)

        print(f"Processed: {file_name}")

# === Change these paths ===
input_dir = "/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/Mask4Former3D/data/test/202/velodyne/"
output_dir = "/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/Mask4Former3D/data/test/202/velodyne_new"

process_folder(input_dir, output_dir)

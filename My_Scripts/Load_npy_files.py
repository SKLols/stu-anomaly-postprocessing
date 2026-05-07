import numpy as np

# file path
#file_path = "/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/Mask4Former3D/validate_stats/pointcloud_mean.npy"
file_path = "/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/validate_stats/stu_calibration_COMBINED_all_sequences_50frames_vars.npy"

# Load the .npy file
data = np.load(file_path, allow_pickle=True)

# Print the type and shape of the data
print("Type:", type(data))
if hasattr(data, "shape"):
    print("Shape:", data.shape)

# Print the actual content (be careful if it's huge)
print("Data:\n", data)

# Save as .txt (one value per line)
output_path = file_path.replace(".npy", ".txt")
# np.savetxt(output_path, data, fmt="%.6f")  # keep 6 decimals

print(f"Data saved to {output_path}")
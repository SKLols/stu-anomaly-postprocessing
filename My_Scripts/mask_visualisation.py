'''
Loads / Visualise single mask for specific query. 
Loads / Visualise all 100 query mask in sequence
Reshape the file to orignal point cloud shape
'''
# #This is working script automatically loads single mask and also gives 

# import numpy as np
# import open3d as o3d
# import matplotlib.pyplot as plt
# from pathlib import Path

# def load_bin_file(bin_path):
#     """Load LiDAR point cloud from .bin file and calculate range"""
#     points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
#     xyz = points[:, :3]
    
#     # Calculate point cloud range
#     min_point = np.min(xyz, axis=0)
#     max_point = np.max(xyz, axis=0)
#     range_size = max_point - min_point
    
#     print("\nLiDAR Point Cloud Range:")
#     print(f"Min (x,y,z): {min_point}")
#     print(f"Max (x,y,z): {max_point}")
#     print(f"Range (dx,dy,dz): {range_size}")
#     print(f"Total points: {len(xyz)}")
    
#     return xyz

# def load_mask_values(mask_path):
#     """Load mask values from text file and calculate range"""
#     mask_values = np.loadtxt(mask_path)
    
#     # Calculate mask value ranges
#     print("\nMask Value Ranges:")
#     for i in range(2):  # First two columns
#         col = mask_values[:, i]
#         print(f"Column {i}: Min={col.min():.2f}, Max={col.max():.2f}, Mean={col.mean():.2f}")
    
#     return mask_values

# def visualize_point_cloud(points, values, column_idx=0, title="Mask Visualization"):
#     """Visualize point cloud colored by mask values"""
#     mask_values = values[:, column_idx]
#     normalized_values = (mask_values - mask_values.min()) / (mask_values.max() - mask_values.min())
    
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points)
    
#     colors = plt.get_cmap("viridis")(normalized_values)[:, :3]
#     pcd.colors = o3d.utility.Vector3dVector(colors)
    
#     # Add coordinate frame for reference
#     coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)
    
#     o3d.visualization.draw_geometries(
#         [pcd, coord_frame],
#         window_name=f"{title} (Column {column_idx})",
#         width=800,
#         height=600
#     )


# # # File paths (UPDATE THESE)
# bin_file_path = "/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/data/stu_dataset/validation/138/velodyne/000000.bin"
# mask_file_path = "/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/batch_0_confid_remapped.txt"


# # Load data with range calculations
# print("=== Data Analysis ===")
# points = load_bin_file(bin_file_path)
# mask_values = load_mask_values(mask_file_path)

# # Verify shapes match
# assert points.shape[0] == mask_values.shape[0], "Point cloud and mask values must have same number of points"

# # Visualizations
# print("\n=== Visualizations ===")
# # Simply change this number to visualize any column (0-99)
# column_to_visualize = 20  # Change this to any column index you want
# visualize_point_cloud(points, mask_values, column_idx=column_to_visualize, 
#                      title=f"Mask Column {column_to_visualize}")

# #This visualises multiple =========================================================================

# import numpy as np
# import open3d as o3d
# import matplotlib.pyplot as plt
# from pathlib import Path

# def load_bin_file(bin_path):
#     """Load LiDAR point cloud from .bin file"""
#     points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
#     return points[:, :3]

# def load_mask_values(mask_path):
#     """Load mask values from text file"""
#     return np.loadtxt(mask_path)

# def visualize_columns_interactive(points, values):
#     """Visualize mask columns with Q key to advance"""
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points)
    
#     vis = o3d.visualization.VisualizerWithKeyCallback()
#     vis.create_window(width=800, height=600)
#     vis.add_geometry(pcd)
    
#     current_col = -1
#     is_single_column = len(values.shape) == 1
    
#     def advance(vis):
#         nonlocal current_col
#         #current_col = (current_col + 1) % values.shape[1] #Multiple columns

#         if is_single_column:
#             # For single column, just show it (no cycling)
#             mask_values = values
#         else:
#             # For multiple columns, cycle through them
#             current_col = (current_col + 1) % values.shape[1]
#             mask_values = values[:, current_col]
        
#         # Update with new column data
#         #mask_values = values[:, current_col]
#         normalized_values = (mask_values - mask_values.min()) / (mask_values.max() - mask_values.min())
#         colors = plt.get_cmap("viridis")(normalized_values)[:, :3]
#         pcd.colors = o3d.utility.Vector3dVector(colors)
        
#         vis.update_geometry(pcd)
#         vis.update_renderer()
#         print(f"Showing column {current_col} - Range: [{mask_values.min():.2f}, {mask_values.max():.2f}]")

#         if is_single_column:
#             print(f"Showing max logit values - Range: [{mask_values.min():.2f}, {mask_values.max():.2f}]")
#         else:
#             print(f"Showing column {current_col} - Range: [{mask_values.min():.2f}, {mask_values.max():.2f}]")

#     # Register Q key callback
#     vis.register_key_callback(ord('Q'), advance)
    
#     # Show first column initially
#     advance(vis)
    
#     #print("\nPress Q to view next mask column")
#     #print("Close window to exit\n")

#     if is_single_column:
#         print("\nShowing single column of max logit values")
#     else:
#         print("\nPress Q to view next mask column")
#     print("Close window to exit\n")
    
#     vis.run()
#     vis.destroy_window()

# # File paths (UPDATE THESE)
# bin_file_path = "/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/data/stu_dataset/validation/141/velodyne/000300.bin"
# #mask_file_path = "/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/batch_0_confid_remapped.txt"
# mask_file_path = "/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/batch_0_confid_maxlogit.txt"


# # Load data
# points = load_bin_file(bin_file_path)
# mask_values = load_mask_values(mask_file_path)

# # Verify shapes match
# assert points.shape[0] == mask_values.shape[0], "Point cloud and mask values must have same number of points"

# # Start interactive visualization
# visualize_columns_interactive(points, mask_values)

# #This visualises multiple with front view initially set=========================================================================

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from pathlib import Path
import math

def load_bin_file(bin_path):
    """Load LiDAR point cloud from .bin file"""
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return points[:, :3]

def load_mask_values(mask_path):
    """Load mask values from text file"""
    return np.loadtxt(mask_path)

def visualize_columns_interactive(points, values):
    """Visualize mask columns with Q key to advance"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Create coordinate frame
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(width=800, height=600)
    vis.add_geometry(pcd)
    vis.add_geometry(coord)
    
    current_col = -1
    is_single_column = len(values.shape) == 1
    
    def advance(vis):
        nonlocal current_col
        #current_col = (current_col + 1) % values.shape[1] #Multiple columns

        if is_single_column:
            # For single column, just show it (no cycling)
            mask_values = values
        else:
            # For multiple columns, cycle through them
            current_col = (current_col + 1) % values.shape[1]
            mask_values = values[:, current_col]
        
        # Update with new column data
        #mask_values = values[:, current_col]
        normalized_values = (mask_values - mask_values.min()) / (mask_values.max() - mask_values.min())
        colors = plt.get_cmap("viridis")(normalized_values)[:, :3]
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        vis.update_geometry(pcd)
        vis.update_renderer()
        print(f"Showing column {current_col} - Range: [{mask_values.min():.2f}, {mask_values.max():.2f}]")

        if is_single_column:
            print(f"Showing max logit values - Range: [{mask_values.min():.2f}, {mask_values.max():.2f}]")
        else:
            print(f"Showing column {current_col} - Range: [{mask_values.min():.2f}, {mask_values.max():.2f}]")

    # Register Q key callback
    vis.register_key_callback(ord('Q'), advance)
    
    # Show first column initially
    advance(vis)

    # === Set custom initial camera view ===
    theta = math.radians(30)  # 30 degrees rotation around Y axis
    front = [-math.cos(theta), 0, math.sin(theta)]
    up = [0, 0, 1]
    lookat = [0, 0, 0]
    zoom = 0.02

    vis.poll_events()
    vis.update_renderer()

    ctr = vis.get_view_control()
    ctr.set_front(front)
    ctr.set_up(up)
    ctr.set_lookat(lookat)
    ctr.set_zoom(zoom)

    
    #print("\nPress Q to view next mask column")
    #print("Close window to exit\n")

    if is_single_column:
        print("\nShowing single column of max logit values")
    else:
        print("\nPress Q to view next mask column")
    print("Close window to exit\n")
    print(f"Initial view set with front vector: {front}\n")
    
    vis.run()
    vis.destroy_window()

# File paths (UPDATE THESE)
bin_file_path = "/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/data/stu_dataset/validation/141/velodyne/000300.bin"
#mask_file_path = "/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/batch_0_confid_remapped.txt"
mask_file_path = "/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/batch_0_confid_maxlogit.txt"


# Load data
points = load_bin_file(bin_file_path)
mask_values = load_mask_values(mask_file_path)

# Verify shapes match
assert points.shape[0] == mask_values.shape[0], "Point cloud and mask values must have same number of points"

# Start interactive visualization
visualize_columns_interactive(points, mask_values)

'''
#Reshape the file==============================================================================================
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from pathlib import Path
import os
import torch


#Currently Saved logits are from sequence 138, 141, 106
#138 has 515 (i.e. 0 to 515) frames, 141 has 696 (516 to 1212) 141 frame number will be calculated as "515+1+frame number" has


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#Load the masks and logits after softmax
pred_masks = np.loadtxt(f'/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/Mask4Former3D/logits_dump/batch_1045_masks_1.txt')
pred_masks = torch.from_numpy(pred_masks).float()  # Convert from Numpy to Tensor 
print(f"Shape of pred_masks (before): {pred_masks.shape}")  # Should be (N, 100)
pred_logits = np.loadtxt(f'/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/Mask4Former3D/logits_dump/batch_1045_logits_sm_1.txt')
pred_logits = torch.from_numpy(pred_logits).float()

pred_masks_inverse = np.loadtxt(f"/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/Mask4Former3D/logits_dump/batch_1045_inverse_maps.txt")
print(f"Shape of pred_masks_inverse: {pred_masks_inverse.shape}")
print(f"Type of pred_masks_inverse: {type(pred_masks_inverse)}")

pred_masks_inverse = pred_masks_inverse.astype(np.int64)
pred_masks_inverse = torch.from_numpy(pred_masks_inverse)
print(f"pred_mask_inverse:",pred_masks_inverse.shape)


#confid = pred_masks[pred_masks_inverse]  # This is just for inverse mapping the mask

confid = pred_masks.sigmoid().matmul(pred_logits)
confid_inverse = confid[pred_masks_inverse]
max_logit = torch.max(confid, dim=1).values[pred_masks_inverse]
max_logit_final = (max_logit * -1) + 1
#confid = confid.argmax(dim=1)[inv_map]
print(f"Shape of pred_masks_inverse (after): {confid.shape}")
print(confid)

#dot_product_matrix = torch.tensor([
#    [0.9,  0.1],
#    [0.2, -0.4],
#    [1.1,  0.3]
#])
#
#Test_=sigmoid(dot_product_matrix)
#print("Test:",Test_)

os.makedirs('/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/Mask4Former3D/logits_dump', exist_ok=True)
filename = f'batch_0_confid_maxlogit.txt'
filename_maxlogit = f'batch_0_max_logit_maxlogit.txt'
filename_maxlogit_final = f'batch_0_max_logit_final_maxlogit.txt'

np.savetxt(filename,  confid_inverse, fmt='%.4f')
print(f"Saved confid {filename} (shape: {confid_inverse.shape})")

np.savetxt(filename_maxlogit,  max_logit, fmt='%.4f')
print(f"Saved confid {filename} (shape: {max_logit.shape})")

np.savetxt(filename_maxlogit_final,  max_logit, fmt='%.4f')
print(f"Saved confid {filename} (shape: {max_logit_final.shape})")
'''
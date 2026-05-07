import numpy as np
import open3d as o3d
import os
import torch


def load_point_cloud(scan_path):
    """Load LiDAR .bin file (x,y,z,intensity)"""
    scan = np.fromfile(scan_path, dtype=np.float32).reshape(-1, 4)
    return scan[:, :3]  # Return only x,y,z

def load_labels(label_path,labels_file = "labels_file.txt"):
    labels = np.fromfile(label_path, dtype=np.uint32).astype(np.int32)
    semantic_label = labels & 0xFFFF
    instance_label = labels >> 16

    print("labels:",instance_label)
    with open(labels_file, 'w') as f:
        f.write("labels:n")
        np.savetxt(f, instance_label, fmt='%d', header='Label Values', comments='')

    return semantic_label, instance_label

def save_labels(labels, save_path):
    # Ensure input arrays have compatible shapes
    instance_label, semantic_label = labels
    assert instance_label.shape == semantic_label.shape, "Instance and semantic labels must have the same shape"

    # Convert to 32-bit unsigned integers for bitwise operations
    instance_upper = (instance_label.astype(np.uint32) & 0xFFFF) << 16  # Upper 16 bits
    semantic_lower = semantic_label.astype(np.uint32) & 0xFFFF           # Lower 16 bits

    # Combine into final 32-bit labels
    combined_labels = instance_upper | semantic_lower

    # Save to binary file
    combined_labels.tofile(save_path)

# def _apply_distance_mask(self, points, labels):
#     """Filter labels based on distance (original logic)."""
#     distances = np.linalg.norm(points, axis=1)
#     labels = np.where(
#         (distances > self.max_eval_distance) | (distances < self.min_eval_distance),
#         -1,
#         labels,
#     )
#     return labels

points = load_point_cloud("/media/ubuntu22/HDD22T/_Sourabh/PMF-master/data/semantic-kitti-fov/sequences/141/velodyne/000000.bin")
sem_preds = np.loadtxt("/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/Mask4Former3D/saved/2025-07-15_145436/prediction/141/sem_preds_000320.txt")
# label_file="/media/ubuntu22/HDD22T/_Sourabh/PMF-master/experiments_semkitti_stu/PMF-SemanticKitti/log_SemanticKitti_PMFNet-resnet34_bs8-lr0.001_baseline_timestamp/Eval-SemanticKitti-PMFNet-best_IOU_model-noKNN-debug_timestamp/preds_all_methods/sequences/141/predictions/000000.label"

# 2. Define the color map (RGB 0-1)
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
    12: [0, 0, 0.5],     # other-ground (dark blue)
    13: [1.0, 0.84, 0.0],       # building (light blue)
    14: [0, 0.5, 0.5],   # fence (teal)
    15: [0, 1, 0],       # vegetation (green)
    16: [0.55, 0.27, 0.07], # trunk (brown)
    17: [0, 1, 0.5],     # terrain (spring green)
    18: [1, 0.75, 0.8],  # pole (pink)
    19: [0.29, 0, 0.51]  # traffic-sign (indigo)
}

# 3. Create color array (unknown classes will remain black)
#colors = np.zeros((len(sem_preds), 3))  # Initialize all to black
colors = np.zeros((len(label_file), 3))

# Apply colors for known classes
for class_id, color in COLOR_MAP.items():
    #mask = (sem_preds == class_id-1)
    mask = (label_file == class_id-1)
    colors[mask] = color

# 4. Create and visualize point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

# 5. Simple visualization
o3d.visualization.draw_geometries([pcd], 
                                 window_name="Semantic Predictions",
                                 width=800,
                                 height=600)













































'''
pred_logits = np.loadtxt(f'/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/Mask4Former3D/logits_dump/batch_0_logits_sm_1.txt')
# 2. Generate a synthetic binary mask for demo (replace with your actual mask)
pred_masks = np.loadtxt(f'/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/Mask4Former3D/logits_dump/batch_0_masks_1.txt')
print(f"Shape of pred_masks: {pred_masks.shape}")  # Should be (N, 100)

pred_masks_inverse = np.loadtxt(f"/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/Mask4Former3D/logits_dump/batch_0_inverse_maps.txt")
print(f"Shape of pred_masks_inverse: {pred_masks_inverse.shape}")
print(f"Type of pred_masks_inverse: {type(pred_masks_inverse)}")
pred_masks_inverse = pred_masks_inverse.astype(np.int64)
#confid = pred_masks[pred_masks_inverse]
#confid = (confid*-1)
#confid = pred_masks.sigmoid().matmul(pred_logits)
#confid = confid.argmax(dim=1)[inv_map]
#print(f"Shape of pred_masks_inverse: {confid.shape}")
#print(confid)

#os.makedirs('confid_dump', exist_ok=True)
#filename = f'batch_confid_remapped.txt'
#
#np.savetxt(filename,  confid, fmt='%.4f')
#print(f"Saved confid {filename} (shape: {confid.shape})")

#Full_pred_masks = pred_masks[inv_map]
#binary_mask = np.zeros(len(points), dtype=np.uint8)
#binary_mask[points[:,0] > 0] = 1  # Simple threshold on x-axis
'''



'''

def visualise_specific_query_mask(points,confid, query_idx, threshold=30):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    #get mask for this query
    query_mask = confid[:,query_idx]

    conf_normailized = (query_mask - query_mask.min()) / (query_mask.max() - query_mask.min() + 1e-6)
    jet = cm.get_cmap("jet")
    colors = jet(conf_normailized)[:, :3]

    ##gray for background, red for mask
    #colors = np.zeros((len(points),3))
    #colors[:, :] = [0.3, 0.3, 0.3] #grey

    #mask = query_mask > threshold
    #colors[mask] = [1,0,0] #red for mask

    pcd.colors = o3d.utility.Vector3dVector(colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name = f'Query {query_idx} Mask')
    vis.add_geometry(pcd)

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0)
    vis.add_geometry(coord_frame)

    vis.run()
    vis.destroy_window()

visualise_specific_query_mask(points,confid,99)
'''


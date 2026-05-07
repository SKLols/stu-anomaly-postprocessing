import torch
import numpy as np
import MinkowskiEngine as ME
from pathlib import Path
from omegaconf import OmegaConf
import hydra
import os

# === Load Model and Config ===
def load_model(config_path, ckpt_path):
    cfg = OmegaConf.load(config_path)
    model = hydra.utils.instantiate(cfg.model)
    state_dict = torch.load(ckpt_path, map_location='cpu')["state_dict"]
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.cuda()
    return model, cfg

# === Load Point Cloud ===
def load_pointcloud(file_path):
    # Example for SemanticKITTI .bin format (N x 4: x,y,z,intensity)
    points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    coords = np.floor(points[:, :3] / 0.05).astype(np.int32)  # quantize (voxel size = 5cm)
    feats = points[:, 3:]  # use intensity only
    return coords, feats, points[:, :3]

# === Inference ===
def inference(model, coords, feats):
    coords_batch = ME.utils.batched_coordinates([coords])
    feats_batch = torch.from_numpy(feats).float().cuda()
    x = ME.SparseTensor(coordinates=coords_batch, features=feats_batch)
    with torch.no_grad():
        output = model(x, raw_coordinates=coords, is_eval=True)

    pred_logits = output["pred_logits"]
    pred_masks = output["pred_masks"]

    logits = torch.softmax(pred_logits[0], dim=-1)[:-1]  # remove no-object class
    masks = pred_masks[0].sigmoid()

    # Compute semantic prediction
    sem_confid = masks.float().matmul(logits)  # [N_points x num_classes]
    sem_preds = torch.argmax(sem_confid, dim=1).cpu().numpy()
    return sem_preds

# === Save to TXT or PLY ===
def save_results(original_xyz, sem_preds, output_path):
    result = np.hstack((original_xyz, sem_preds[:, None]))
    np.savetxt(output_path.with_suffix('.txt'), result, fmt='%.3f %.3f %.3f %d')
    print(f"Saved: {output_path}.txt")

# === Main Function ===
def main():
    config_path = "/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/Mask4Former3D/conf/config_panoptic_3d.yaml"
    ckpt_path = "/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/Mask4Former3D/checkpoints/Ensemble_model_1.ckpt"
    pointcloud_path = "/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/data/stu_dataset/validation/141/velodyne/000320.bin"
    output_path = Path("output/your_result")

    model, cfg = load_model(config_path, ckpt_path)
    coords, feats, xyz = load_pointcloud(pointcloud_path)
    sem_preds = inference(model, coords, feats)
    save_results(xyz, sem_preds, output_path)

if __name__ == "__main__":
    main()


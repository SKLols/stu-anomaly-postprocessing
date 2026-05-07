#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np

try:
    import open3d as o3d
except Exception:
    print("ERROR: Open3D is required (pip install open3d).", file=sys.stderr)
    raise

try:
    import yaml
    _HAS_YAML = True
except Exception:
    _HAS_YAML = False


def read_velodyne_bin(bin_path: Path) -> np.ndarray:
    arr = np.fromfile(str(bin_path), dtype=np.float32)
    if arr.size % 4 != 0:
        raise ValueError(f"Invalid .bin file size not multiple of 4 floats: {bin_path}")
    return arr.reshape(-1, 4)


def read_labels(label_path: Path) -> np.ndarray:
    return np.fromfile(str(label_path), dtype=np.uint32)


def _to_int_tuple_map(d):
    return {int(k): tuple(d[k]) for k in d}


def load_semantickitti_config(cfg_path: Optional[Path]):
    out = {"learning_map": None, "color_map": None, "learning_color_map": None}
    if cfg_path is None:
        return out
    if not cfg_path.exists():
        print(f"WARNING: config not found: {cfg_path}. Using fallback colors.", file=sys.stderr)
        return out
    if not _HAS_YAML:
        print("WARNING: PyYAML not installed. `pip install pyyaml` to use official color map.", file=sys.stderr)
        return out
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    if "learning_map" in cfg:
        out["learning_map"] = {int(k): int(v) for k, v in cfg["learning_map"].items()}
    if "color_map" in cfg:
        out["color_map"] = _to_int_tuple_map(cfg["color_map"])
    if "learning_color_map" in cfg:
        out["learning_color_map"] = _to_int_tuple_map(cfg["learning_color_map"])
    return out


def fallback_palette() -> Dict[int, tuple]:
    return {
        0:  (0, 0, 0),          1:  (0, 0, 255),
        10: (245, 150, 100),    11: (245, 230, 100),
        13: (255, 0, 0),        15: (255, 0, 255),
        16: (200, 200, 100),    18: (150, 60, 30),
        20: (255, 255, 0),      30: (0, 0, 100),
        31: (0, 60, 100),       32: (0, 0, 255),
        40: (255, 100, 255),    44: (255, 100, 0),
        48: (200, 200, 200),    49: (150, 150, 150),
        50: (75, 0, 75),        51: (75, 0, 175),
        52: (0, 200, 255),      60: (50, 120, 255),
        70: (0, 175, 0),        71: (0, 60, 135),
        72: (80, 240, 150),     80: (150, 240, 255),
        81: (0, 0, 255),        99: (255, 255, 255),
        252: (245, 150, 100),   253: (245, 230, 100),
        254: (255, 0, 0),       255: (255, 0, 255),
        256: (0, 0, 100),       257: (0, 60, 100),
        258: (0, 0, 255),
    }


def hashed_color(ids: np.ndarray) -> np.ndarray:
    ids = ids.astype(np.uint32)
    r = (1103515245 * (ids ^ 12345) + 12345) & 0xFFFFFFFF
    g = (1103515245 * (ids ^ 54321) + 98765) & 0xFFFFFFFF
    b = (1103515245 * (ids ^ 22222) + 55555) & 0xFFFFFFFF
    rgb = np.stack([r % 256, g % 256, b % 256], axis=-1).astype(np.uint8)
    return rgb


def colors_from_yaml_or_fallback(sem_ids: np.ndarray,
                                 cfg: Dict[str, Optional[Dict[int, int]]]) -> np.ndarray:
    learning_map = cfg.get("learning_map", None)
    color_map = cfg.get("color_map", None)
    learning_color_map = cfg.get("learning_color_map", None)

    if learning_color_map and learning_map is not None:
        train_ids = np.vectorize(lambda x: learning_map.get(int(x), 0))(sem_ids)
        colors = np.zeros((sem_ids.size, 3), dtype=np.uint8)
        for tid, rgb in learning_color_map.items():
            mask = (train_ids == tid)
            if np.any(mask):
                colors[mask] = np.array(rgb, dtype=np.uint8)
        leftover = (colors.sum(axis=1) == 0)
        if np.any(leftover):
            colors[leftover] = hashed_color(train_ids[leftover])
        return colors.astype(np.float32) / 255.0

    if color_map is not None:
        keys = np.array(list(color_map.keys()), dtype=int)
        if np.any(keys >= 10):
            colors = np.zeros((sem_ids.size, 3), dtype=np.uint8)
            for sid, rgb in color_map.items():
                mask = (sem_ids == sid)
                if np.any(mask):
                    colors[mask] = np.array(rgb, dtype=np.uint8)
            leftover = (colors.sum(axis=1) == 0)
            if np.any(leftover):
                colors[leftover] = hashed_color(sem_ids[leftover])
            return colors.astype(np.float32) / 255.0
        if learning_map is not None:
            train_ids = np.vectorize(lambda x: learning_map.get(int(x), 0))(sem_ids)
            colors = np.zeros((sem_ids.size, 3), dtype=np.uint8)
            for tid, rgb in color_map.items():
                mask = (train_ids == tid)
                if np.any(mask):
                    colors[mask] = np.array(rgb, dtype=np.uint8)
            leftover = (colors.sum(axis=1) == 0)
            if np.any(leftover):
                colors[leftover] = hashed_color(train_ids[leftover])
            return colors.astype(np.float32) / 255.0

    pal = fallback_palette()
    colors = np.zeros((sem_ids.size, 3), dtype=np.uint8)
    for sid, rgb in pal.items():
        mask = (sem_ids == sid)
        if np.any(mask):
            colors[mask] = np.array(rgb, dtype=np.uint8)
    leftover = (colors.sum(axis=1) == 0)
    if np.any(leftover):
        colors[leftover] = hashed_color(sem_ids[leftover])
    return colors.astype(np.float32) / 255.0


def visualize(points_xyz: np.ndarray, colors01: np.ndarray, point_size: float = 1.0,
              save_ply: Optional[Path] = None, save_png: Optional[Path] = None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors01)

    if save_ply is not None:
        o3d.io.write_point_cloud(str(save_ply), pcd, write_ascii=False, compressed=False, print_progress=False)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="SemanticKITTI Labels", width=1280, height=720, visible=True)
    vis.add_geometry(pcd)

    opt = vis.get_render_option()
    opt.point_size = float(point_size)
    opt.background_color = np.array([0, 0, 0])

    ctr = vis.get_view_control()
    ctr.set_front([0.0, -1.0, 0.0])
    ctr.set_lookat([0.0, 0.0, 0.0])
    ctr.set_up([0.0, 0.0, 1.0])
    ctr.set_zoom(0.35)

    vis.poll_events()
    vis.update_renderer()

    if save_png is not None:
        vis.capture_screen_image(str(save_png), do_render=True)

    print("Close the window to finish...")
    vis.run()
    vis.destroy_window()


def resolve_paths(args):
    if args.velodyne and args.label:
        return Path(args.velodyne), Path(args.label)

    if not args.dataset_root or not args.seq or not args.frame:
        print("ERROR: Please provide either (--velodyne & --label) or (--dataset_root, --seq, --frame).", file=sys.stderr)
        sys.exit(1)

    seq_dir = Path(args.dataset_root) / "sequences" / args.seq
    bin_path = seq_dir / "velodyne" / f"{args.frame}.bin"

    if args.pred_root:
        pred_dir = Path(args.pred_root)
        label_path = pred_dir / f"{args.frame}.label"
    else:
        label_path = seq_dir / "labels" / f"{args.frame}.label"

    return bin_path, label_path


def main():
    ap = argparse.ArgumentParser(description="SemanticKITTI label visualizer (GT or predictions)")
    ap.add_argument("--velodyne", type=Path, help="Path to a single frame .bin")
    ap.add_argument("--label", type=Path, help="Path to a single frame .label (GT or prediction)")
    ap.add_argument("--dataset_root", type=Path, help="Root of SemanticKITTI dataset (contains 'sequences' folder).")
    ap.add_argument("--seq", type=str, help="Sequence id, e.g., 11")
    ap.add_argument("--frame", type=str, help="Frame id, e.g., 000000")
    ap.add_argument("--pred_root", type=Path, help="Prediction folder for a sequence, e.g., .../sequences/11/predictions")
    ap.add_argument("--config", type=Path, default=None, help="Path to semantic-kitti.yaml (optional).")
    ap.add_argument("--point_size", type=float, default=1.0)
    ap.add_argument("--downsample", type=int, default=1, help="Keep 1 in N points (uniform).")
    ap.add_argument("--save_ply", type=Path, default=None, help="Optional output colored PLY path.")
    ap.add_argument("--save_png", type=Path, default=None, help="Optional screenshot PNG path.")
    args = ap.parse_args()

    bin_path, label_path = resolve_paths(args)

    if not bin_path.exists():
        print(f"ERROR: .bin not found: {bin_path}", file=sys.stderr)
        sys.exit(1)
    if not label_path.exists():
        print(f"ERROR: .label not found: {label_path}", file=sys.stderr)
        sys.exit(1)

    scan = read_velodyne_bin(bin_path)
    labels = read_labels(label_path)
    if scan.shape[0] != labels.shape[0]:
        print(f"ERROR: Point count mismatch: {scan.shape[0]} pts vs {labels.shape[0]} labels.", file=sys.stderr)
        sys.exit(1)

    sem_ids = (labels & np.uint32(0xFFFF)).astype(np.int32)

    cfg = load_semantickitti_config(args.config)
    colors01 = colors_from_yaml_or_fallback(sem_ids, cfg)

    pts_xyz = scan[:, :3].copy()
    if args.downsample > 1:
        idx = np.arange(pts_xyz.shape[0])[::args.downsample]
        pts_xyz = pts_xyz[idx]
        colors01 = colors01[idx]

    unique_classes, counts = np.unique(sem_ids, return_counts=True)
    print("\nAvailable semantic classes in this frame:")

    # Try to get label names if available in config YAML
    label_names = None
    if args.config is not None and args.config.exists() and _HAS_YAML:
        with open(args.config, "r") as f:
            cfg_yaml = yaml.safe_load(f)
            label_names = cfg_yaml.get("labels", None)

    for cid, cnt in zip(unique_classes, counts):
        name = label_names.get(cid, "unknown") if label_names else "unknown"
        print(f"Class ID {cid:3d} ({name:15s}): {cnt:7d} points")

    visualize(pts_xyz, colors01, point_size=args.point_size,
              save_ply=args.save_ply, save_png=args.save_png)


if __name__ == "__main__":
    main()
"""
kitti GT
python vis_stu_inlier_GT.py \
--velodyne /media/ubuntu22/HDD22T/_Zhiran/3d_anomaly_segmentation/stu_dataset/Mask4Former3D/data/train/07/velodyne/000086.bin \
--label    /media/ubuntu22/HDD22T/_Zhiran/3d_anomaly_segmentation/stu_dataset/Mask4Former3D/data/train/07/labels/000086.label \
--config   /media/ubuntu22/HDD22T/_Zhiran/3d_segmentation/semantic-kitti-api-master/config/semantic-kitti.yaml \
--point_size 1.5 --downsample 2 --save_png pred_000201.png

kitti GT
python vis_stu_inlier_GT.py \
--velodyne /media/ubuntu22/HDD22T/_Sourabh/PMF-master/data/semantic-kitti-fov/sequences/08/velodyne/000753.bin \
--label    /media/ubuntu22/HDD22T/_Sourabh/PMF-master/data/semantic-kitti-fov/sequences/08/labels/000753.label \
--config   /media/ubuntu22/HDD22T/_Zhiran/3d_segmentation/semantic-kitti-api-master/config/semantic-kitti.yaml \
--point_size 1.5 --downsample 2 --save_png pred_000201.png

  STU GT
python vis_stu_inlier_GT.py \
--velodyne /media/ubuntu22/HDD22T/_Zhiran/3d_anomaly_segmentation/stu_dataset/Mask4Former3D/data/train/206/velodyne/000000.bin \
--label    /media/ubuntu22/HDD22T/_Zhiran/3d_anomaly_segmentation/stu_dataset/Mask4Former3D/data/train/206/labels/000000.label \
--config   /media/ubuntu22/HDD22T/_Zhiran/3d_segmentation/semantic-kitti-api-master/config/semantic-kitti.yaml \
--point_size 1.5 --downsample 2 --save_png pred_000201.png

  STU GT FV
python vis_stu_inlier_GT.py \
--velodyne /media/ubuntu22/HDD22T/_Zhiran/3d_anomaly_segmentation/stu_dataset/Mask4Former3D/data/train/206/velodyne/000000.bin \
--label    /media/ubuntu22/HDD22T/_Zhiran/3d_anomaly_segmentation/stu_dataset/Mask4Former3D/data/train/206/labels/000000.label \
--config   /media/ubuntu22/HDD22T/_Zhiran/3d_segmentation/semantic-kitti-api-master/config/semantic-kitti.yaml \
--point_size 1.5 --downsample 2 --save_png pred_000201.png


prediction pmf
python vis_stu_inlier_GT.py \
--velodyne /media/ubuntu22/HDD22T/_Sourabh/PMF-master/data/semantic-kitti-fov/sequences/141/velodyne/000333.bin \
--label    /media/ubuntu22/HDD22T/_Sourabh/PMF-master/experiments_semkitti_stu/PMF-SemanticKitti/log_SemanticKitti_PMFNet-resnet34_bs8-lr0.001_baseline_timestamp/Eval-SemanticKitti-PMFNet-best_IOU_model-noKNN-debug_timestamp/preds_all_methods/sequences/141/predictions/000333.label \
--config   /media/ubuntu22/HDD22T/_Zhiran/3d_segmentation/semantic-kitti-api-master/config/semantic-kitti.yaml \
--point_size 1.5 --downsample 2 --save_png pred_000201.png

prediction mask4former fv
python vis_stu_inlier_GT.py \
--velodyne /media/ubuntu22/HDD22T/_Sourabh/PMF-master/data/semantic-kitti-fov/sequences/141/velodyne/000333.bin \
--label    /media/ubuntu22/HDD22T/_Sourabh/PMF-master/experiments_semkitti_stu/PMF-SemanticKitti/log_SemanticKitti_PMFNet-resnet34_bs8-lr0.001_baseline_timestamp/Eval-SemanticKitti-PMFNet-best_IOU_model-noKNN-debug_timestamp/preds_all_methods/sequences/141/predictions/000333.label \
--config   /media/ubuntu22/HDD22T/_Zhiran/3d_segmentation/semantic-kitti-api-master/config/semantic-kitti.yaml \
--point_size 1.5 --downsample 2 --save_png pred_000201.png



"""

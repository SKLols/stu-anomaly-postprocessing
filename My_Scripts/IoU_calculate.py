import numpy as np
from pathlib import Path
import os

# SemanticKITTI class information
semantic_kitti_classes = [
    'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle', 
    'person', 'bicyclist', 'motorcyclist', 'road', 'parking', 
    'sidewalk', 'other-ground', 'building', 'fence', 'vegetation',
    'trunk', 'terrain', 'pole', 'traffic-sign'
]

# Learning map from SemanticKITTI
learning_map = {
    10: 0,  # "car" -> 0
    11: 1,  # "bicycle" -> 1
    15: 2,  # "motorcycle" -> 2
    18: 3,  # "truck" -> 3
    20: 4,  # "other-vehicle" -> 4
    30: 5,  # "person" -> 5
    31: 6,  # "bicyclist" -> 6
    32: 7,  # "motorcyclist" -> 7
    40: 8,  # "road" -> 8
    44: 9,  # "parking" -> 9
    48: 10, # "sidewalk" -> 10
    49: 11, # "other-ground" -> 11
    50: 12, # "building" -> 12
    51: 13, # "fence" -> 13
    70: 14, # "vegetation" -> 14
    71: 15, # "trunk" -> 15
    72: 16, # "terrain" -> 16
    80: 17, # "pole" -> 17
    81: 18  # "traffic-sign" -> 18
}

def compute_iou(confusion_matrix):
    """
    Compute IoU from confusion matrix 
    IoU = TP / (TP + FP + FN)
    """
    iou_per_class = np.diag(confusion_matrix) / (
        confusion_matrix.sum(axis=1) + confusion_matrix.sum(axis=0) - np.diag(confusion_matrix) + 1e-8
    )
    return iou_per_class

def evaluate_sequence_08():
    """Main evaluation function for sequence 08"""
    
    # PATHS
    pred_base_path = Path("/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/Mask4Former3D/saved/2025-11-17_201856/prediction_sem_preds/08")
    gt_base_path = Path("/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/Mask4Former3D/data/validation/08/labels")
    
    num_classes = 19
    
    # Initialize confusion matrix
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    # Get all prediction files
    pred_files = sorted(list(pred_base_path.glob("sem_preds_*.npy")))
    print(f"Found {len(pred_files)} prediction files")
    
    processed_files = 0
    
    for pred_file in pred_files:
        # Extract frame number from filename
        frame_num = pred_file.stem.replace("sem_preds_", "").zfill(6)
        gt_file = gt_base_path / f"{frame_num}.label"
        
        if not gt_file.exists():
            continue
        
        # Load predictions
        pred_semantic = np.load(pred_file).astype(np.int64)
        
        # Load ground truth
        gt_data = np.fromfile(gt_file, dtype=np.uint32)
        gt_semantic = gt_data & 0xFFFF
        
        # Remap GT to 0-18 range
        gt_remapped = np.full_like(gt_semantic, -1)
        for orig_label, new_label in learning_map.items():
            gt_remapped[gt_semantic == orig_label] = new_label
        
        # Ensure same length
        min_length = min(len(pred_semantic), len(gt_remapped))
        pred_valid = pred_semantic[:min_length]
        gt_valid = gt_remapped[:min_length]
        
        # Only consider valid points
        valid_mask = gt_valid != -1
        pred_final = pred_valid[valid_mask]
        gt_final = gt_valid[valid_mask]
        
        if len(pred_final) == 0:
            continue
        
        # MEMORY-EFFICIENT: Update confusion matrix directly without bincount
        for gt_class in range(num_classes):
            for pred_class in range(num_classes):
                count = np.sum((gt_final == gt_class) & (pred_final == pred_class))
                confusion_matrix[gt_class, pred_class] += count
        
        processed_files += 1
        
        if processed_files % 100 == 0:
            print(f"Processed {processed_files} files...")
    
    print(f"Successfully processed {processed_files} files out of {len(pred_files)}")
    return confusion_matrix

def print_results_table(iou_per_class, miou):
    """Print LaTeX table format"""
    print("\\textbf{Mask4Former-3D} & L & ", end="")
    for i in range(19):
        print(f"{iou_per_class[i]*100:.2f} & ", end="")
    print(f"{miou*100:.2f} \\\\")

def main():
    print("Evaluating Sequence 08...")
    
    confusion_matrix = evaluate_sequence_08()
    
    # Check if we have any data
    if np.sum(confusion_matrix) == 0:
        print("ERROR: No valid data processed. Check paths and file formats.")
        return
    
    print(f"\nConfusion matrix total points: {np.sum(confusion_matrix)}")
    
    iou_per_class = compute_iou(confusion_matrix)
    miou = np.mean(iou_per_class)
    
    print("\n" + "="*80)
    print("Semantic Segmentation Results - Sequence 08")
    print("="*80)
    
    for i, class_name in enumerate(semantic_kitti_classes):
        print(f"{class_name:15}: {iou_per_class[i]*100:6.2f}%")
    
    print(f"{'mIoU':15}: {miou*100:6.2f}%")
    
    print("\n" + "="*80)
    print("LaTeX Table Format:")
    print("="*80)
    print_results_table(iou_per_class, miou)
    
    # Save results
    results = {
        'per_class_iou': {name: float(iou_per_class[i]) for i, name in enumerate(semantic_kitti_classes)},
        'mIoU': float(miou)
    }
    
    import json
    with open('sequence_08_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to sequence_08_results.json")

if __name__ == "__main__":
    main()
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from pathlib import Path
from sklearn.metrics import roc_curve

def load_point_cloud(filename):
    """Load point cloud from binary file"""
    scan = np.fromfile(filename, dtype=np.float32)
    return scan.reshape((-1, 4))[:, :3], scan.reshape((-1, 4))[:, 3]

def load_labels(filename):
    """Load semantic labels from file"""
    if not os.path.exists(filename):
        return np.array([]), np.array([])
    label = np.fromfile(filename, dtype=np.uint32).reshape(-1)
    sem_label = label & 0xFFFF
    return sem_label, label

def load_data_for_plotting(sequences, method="sml", base_pred_dir=""):
    """
    Load scores, labels, and predicted classes for plotting
    """
    all_scores = []
    all_labels = []
    all_pred_classes = []
    
    for seq in sequences:
        print(f"Loading sequence {seq}...")
        data_dir = f"/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/data/stu_dataset/validation/{seq}"
        
        # Use exact paths based on your structure
        pred_dir = f"{base_pred_dir}/prediction_{method}/{seq}"  # SML scores
        class_dir = f"{base_pred_dir}/prediction_sem_preds/{seq}"  # Your class labels path
        
        # Check if directories exist
        if not os.path.exists(pred_dir):
            print(f"❌ Prediction directory not found: {pred_dir}")
            continue
            
        if not os.path.exists(class_dir):
            print(f"❌ Class directory not found: {class_dir}")
            continue
        
        lidar_files = sorted(glob.glob(f"{data_dir}/velodyne/*.bin"))
        print(f"Found {len(lidar_files)} lidar files in sequence {seq}")
        
        files_loaded = 0
        for pcd_file in lidar_files:
            points, _ = load_point_cloud(pcd_file)
            file_id = os.path.basename(pcd_file).replace('.bin', '')
            label_file = f"{data_dir}/labels/{file_id}.label"
            gt_sem, _ = load_labels(label_file)

            # Score file - SML anomaly scores
            score_file = f"{pred_dir}/{file_id}.txt"
            
            # Class file - semantic predictions
            class_file = f"{class_dir}/{file_id}.txt"  # Using your exact path
            
            if os.path.exists(score_file) and os.path.exists(class_file):
                try:
                    # Load SML anomaly scores
                    scores = np.loadtxt(score_file).astype(np.float32)
                    # Load semantic class predictions
                    pred_classes = np.loadtxt(class_file).astype(int)
                    
                    # Filter points based on distance (same as your evaluation)
                    distances = np.linalg.norm(points, axis=1)
                    inlier_labels = np.where(gt_sem != 0, 0, -1)
                    processed_labels = np.where(gt_sem == 2, 1, inlier_labels)
                    processed_labels = np.where(
                        (distances > 50) | (distances < 2.5), -1, processed_labels
                    )
                    ignore_mask = processed_labels != -1
                    
                    # Check if arrays have same length
                    if len(scores) != len(points) or len(pred_classes) != len(points):
                        print(f"  ⚠️  Array length mismatch for {file_id}:")
                        print(f"     Points: {len(points)}, Scores: {len(scores)}, Classes: {len(pred_classes)}")
                        # Use the minimum length
                        min_length = min(len(points), len(scores), len(pred_classes))
                        scores = scores[:min_length]
                        pred_classes = pred_classes[:min_length]
                        ignore_mask = ignore_mask[:min_length]
                    
                    # Store valid points
                    all_scores.append(scores[ignore_mask])
                    all_labels.append(processed_labels[ignore_mask])
                    all_pred_classes.append(pred_classes[ignore_mask])
                    
                    files_loaded += 1
                    print(f"  ✅ Loaded {file_id}: {np.sum(ignore_mask)} valid points")
                    
                except Exception as e:
                    print(f"  ❌ Error loading files for {file_id}: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                if not os.path.exists(score_file):
                    print(f"  ❌ Score file missing: {score_file}")
                if not os.path.exists(class_file):
                    print(f"  ❌ Class file missing: {class_file}")
        
        print(f"Loaded {files_loaded} files from sequence {seq}")
    
    # Combine all data
    if all_scores:
        final_scores = np.concatenate(all_scores)
        final_labels = np.concatenate(all_labels)
        final_pred_classes = np.concatenate(all_pred_classes)
        
        print(f"\n✅ Successfully loaded {len(final_scores)} total points")
        print(f"   ID points: {np.sum(final_labels == 0)}, OOD points: {np.sum(final_labels == 1)}")
        print(f"   Unique predicted classes: {np.unique(final_pred_classes)}")
        
        return final_scores, final_labels, final_pred_classes
    else:
        print("❌ No data loaded - check file paths and naming conventions")
        return None, None, None

def plot_sml_boxplots(scores, labels, pred_classes, output_path="sml_boxplot.png"):
    """
    Plot box plots for SML method
    """
    # DEBUG: Check score distributions by label
    print(f"\nDEBUG: Score statistics by label:")
    print(f"  ID (0) scores - Mean: {np.mean(scores[labels==0]):.3f}, Median: {np.median(scores[labels==0]):.3f}")
    print(f"  OOD (1) scores - Mean: {np.mean(scores[labels==1]):.3f}, Median: {np.median(scores[labels==1]):.3f}")
    
    # Calculate FPR@95
    fpr_vals, tpr_vals, thresholds = roc_curve(y_true=labels, y_score=scores)
    global_fpr = 0
    for tpr_val, fpr_val, thr in zip(tpr_vals, fpr_vals, thresholds):
        if tpr_val >= 0.95:
            global_fpr = fpr_val
            break
    
    print(f"DEBUG: Global FPR@95: {global_fpr:.3f}")

    # Define classes (0-18 for your dataset)
    classes = list(range(19))
    
    # Calculate class frequencies from predictions
    class_counts = []
    for cls in classes:
        class_counts.append(np.sum(pred_classes == cls))
    
    # Sort classes by frequency (most to least frequent)
    sorted_indices = np.argsort(class_counts)[::-1]
    sorted_classes = [classes[i] for i in sorted_indices]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Global calculations
    global_id_scores = scores[labels == 0]
    global_ood_scores = scores[labels == 1]
    
    # Global overlap calculation
    global_id_10, global_id_90 = np.percentile(global_id_scores, [10, 90])
    global_ood_10, global_ood_90 = np.percentile(global_ood_scores, [10, 90])
    
    global_overlap_min = max(global_id_10, global_ood_10)
    global_overlap_max = min(global_id_90, global_ood_90)
    
    # Draw the continuous gray area across all classes
    if global_overlap_min < global_overlap_max:
        opacity = min(global_fpr, 1.0)
        ax.axhspan(global_overlap_min, global_overlap_max, 
                   alpha=opacity * 0.3, color='gray', 
                   label=f'Global Overlap (10-90%, FPR@95: {global_fpr:.3f})')
        print(f"Gray area drawn: [{global_overlap_min:.3f}, {global_overlap_max:.3f}] with opacity {opacity:.3f}")
    
    # Per-class calculations
    fpr_values = []
    class_info = []
    
    for cls_idx, cls in enumerate(sorted_classes):
        # Skip classes with no predictions
        cls_mask = pred_classes == cls
        if not np.any(cls_mask):
            continue
            
        cls_scores = scores[cls_mask]
        cls_gt_labels = labels[cls_mask]
        
        # Split into ID (label == 0) and OOD (label == 1)
        id_scores = cls_scores[cls_gt_labels == 0]
        ood_scores = cls_scores[cls_gt_labels == 1]
        
        if len(id_scores) == 0 or len(ood_scores) == 0:
            continue
        
        # Calculate quartiles and mean
        id_q1, id_q3 = np.percentile(id_scores, [25, 75])
        id_mean = np.mean(id_scores)
        ood_q1, ood_q3 = np.percentile(ood_scores, [25, 75])
        ood_mean = np.mean(ood_scores)
        
        # Calculate FPR at TPR 95%
        if len(np.unique(cls_gt_labels)) > 1:
            cls_fpr_vals, cls_tpr_vals, cls_thresholds = roc_curve(
                y_true=cls_gt_labels, y_score=cls_scores
            )
            cls_fpr = 0
            for tpr_val, fpr_val, thr in zip(cls_tpr_vals, cls_fpr_vals, cls_thresholds):
                if tpr_val >= 0.95:
                    cls_fpr = fpr_val
                    break
        else:
            cls_fpr = 1.0
        
        fpr_values.append(cls_fpr)
        
        # Store class information
        class_info.append({
            'class': cls, 'id_count': len(id_scores), 
            'ood_count': len(ood_scores), 'fpr': cls_fpr
        })
        
        # Plot boxes
        box_width = 0.3
        ax.bar(cls_idx - box_width/2, id_q3 - id_q1, bottom=id_q1, 
               width=box_width, color='red', alpha=0.7, label='ID' if cls_idx == 0 else "")
        ax.plot([cls_idx - box_width/2, cls_idx - box_width/2], [id_q1, id_q3], 'k-', lw=2)
        ax.bar(cls_idx + box_width/2, ood_q3 - ood_q1, bottom=ood_q1,
               width=box_width, color='blue', alpha=0.7, label='OOD' if cls_idx == 0 else "")
        ax.plot([cls_idx + box_width/2, cls_idx + box_width/2], [ood_q1, ood_q3], 'k-', lw=2)
        ax.scatter(cls_idx - box_width/2, id_mean, color='darkred', s=50, zorder=3, 
                  label='Mean ID' if cls_idx == 0 else "")
        ax.scatter(cls_idx + box_width/2, ood_mean, color='darkblue', s=50, zorder=3, 
                  label='Mean OOD' if cls_idx == 0 else "")
    
    # Calculate class-average FPR
    class_avg_fpr = np.mean(fpr_values) if fpr_values else 0.0
    
    # Customize the plot
    ax.set_xlabel('Classes (sorted by frequency)', fontsize=12)
    ax.set_ylabel('SML Score (higher = more anomalous)', fontsize=12)
    ax.set_title(f'SML: In-Distribution vs Out-of-Distribution Score Distributions\n'
                f'Global FPR@95: {global_fpr:.3f} | Class-average FPR@95: {class_avg_fpr:.3f}', 
                fontsize=14)
    
    ax.set_xticks(range(len(sorted_classes)))
    ax.set_xticklabels([f'C{c}' for c in sorted_classes], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print(f"\n=== SML Box Plot Summary ===")
    print(f"Plot saved to: {output_path}")
    print(f"Global FPR@95: {global_fpr:.3f}")
    print(f"Class-average FPR@95: {class_avg_fpr:.3f}")
    print(f"Number of classes with data: {len(class_info)}")
    print(f"Total points: {len(scores)} (ID: {len(global_id_scores)}, OOD: {len(global_ood_scores)})")
    
    return fig, global_fpr, class_avg_fpr

def main():
    """Main function to load data and create SML box plots"""
    # Configuration
    sequences = [125]
    method = "maxlogit"
    base_pred_dir = "/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/Mask4Former3D/saved/2025-11-14_202142"
    output_path = "ml_boxplot.png"
    
    print("=== Loading Data for SML Box Plot ===")
    print(f"Sequences: {sequences}")
    print(f"Method: {method}")
    print(f"Prediction directory: {base_pred_dir}")
    
    # Load data
    scores, labels, pred_classes = load_data_for_plotting(
        sequences=sequences,
        method=method,
        base_pred_dir=base_pred_dir
    )
    
    if scores is not None:
        # Create the box plot
        print("\n=== Creating SML Box Plot ===")
        plot_sml_boxplots(scores, labels, pred_classes, output_path)
    else:
        print("\n❌ Failed to load data. Possible issues:")
        print("1. Check if both SML score and class prediction directories exist")
        print("2. Verify file naming conventions")
        print("3. Check array lengths match")

if __name__ == "__main__":
    main()
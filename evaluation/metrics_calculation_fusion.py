import numpy as np
from sklearn.metrics import auc, average_precision_score, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import os
import glob

class PointOODMetricsCalculator:
    min_eval_distance = 2.5
    max_eval_distance = 50
    min_num_points_to_eval = 5

    def __init__(self):
        self.all_scores = []
        self.all_labels = []

    def update(self, points, scores, target):
        distances = np.linalg.norm(points, axis=1)
        inlier_labels = np.where(target != 0, 0, -1)
        processed_labels = np.where(target == 2, 1, inlier_labels)
        processed_labels = np.where(
            (distances > self.max_eval_distance) | (distances < self.min_eval_distance),
            -1,
            processed_labels,
        )
        ignore_mask = processed_labels != -1
        labels = processed_labels[ignore_mask]

        if np.sum(labels) < self.min_num_points_to_eval:
            return
        if len(scores) != len(target):
            raise ValueError("Prediction and label count mismatch")

        prediction = scores[ignore_mask]
        self.all_scores.append(prediction)
        self.all_labels.append(labels)

    def compute_metrics(self):
        if not self.all_scores:
            return {}

        targets = np.concatenate(self.all_labels, axis=0)
        predictions = np.concatenate(self.all_scores, axis=0)

        AP = average_precision_score(y_true=targets, y_score=predictions)
        roc_auc, fpr, threshold = self._calculate_auroc(predictions, targets)

        return {
            "AP": AP * 100,
            "FPR95": fpr * 100,
            "AUROC": roc_auc * 100,
            "threshold": threshold,
            "scores": predictions,
            "labels": targets
        }

    @staticmethod
    def _calculate_auroc(predictions, targets):
        fpr, tpr, thresholds = roc_curve(y_true=targets, y_score=predictions)
        roc_auc = auc(fpr, tpr)
        fpr_best = 0
        optimal_threshold = 0

        for tpr_val, fpr_val, thr in zip(tpr, fpr, thresholds):
            if tpr_val > 0.95:
                fpr_best = fpr_val
                optimal_threshold = thr
                break

        return roc_auc, fpr_best, optimal_threshold

def load_point_cloud(filename):
    scan = np.fromfile(filename, dtype=np.float32)
    return scan.reshape((-1, 4))[:, :3], scan.reshape((-1, 4))[:, 3]

def load_labels(filename):
    if not os.path.exists(filename):
        return np.array([]), np.array([])
    label = np.fromfile(filename, dtype=np.uint32).reshape(-1)
    sem_label = label & 0xFFFF
    return sem_label, label

def main():
    sequences = [125, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 169]
    # sequences = [125]
    # List all sequences you want to process
    # methods = ["msp", "mask_variance", "mask_variance_energy", "mask_variance_msp"]
    methods = [
        # "prediction_query_based_energy", 
        # "prediction_query_based_entropy", 
        # "prediction_query_based_rba",
        "prediction_query_based_msp",
        # "prediction_query_based_maxlogit"
        ]
    # methods = ["ml"]
    
    all_best_scores = []
    all_best_labels = []

    for Seq in sequences:
        print(f"\n=== Processing sequence {Seq} ===")
        data_dir = f"/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/data/stu_dataset/validation/{Seq}"
        
        method_metrics = {}
        # Process each method
        for method in methods:
            # pred_dir = f"/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/Mask4Former3D/saved/2025-11-29_174432/prediction_query_based_energy/{Seq}"
            # pred_dir = f"/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/Mask4Former3D/saved/2025-11-29_174432/prediction_query_based_entropy/{Seq}"
            # pred_dir = f"/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/Mask4Former3D/saved/2025-11-29_174432/prediction_query_based_rba/{Seq}"
            # pred_dir = f"/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/Mask4Former3D/saved/2025-11-30_023324/prediction_query_based_msp/{Seq}"
            pred_dir = f"/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/Mask4Former3D/saved/2025-12-05_043553/prediction_query_based_sml/{Seq}"
            
            metrics_calculator = PointOODMetricsCalculator()
            
            lidar_files = sorted(glob.glob(f"{data_dir}/velodyne/*.bin"))
            for pcd_file in lidar_files:
                points, _ = load_point_cloud(pcd_file)
                file_id = os.path.basename(pcd_file).replace('.bin', '')
                label_file = f"{data_dir}/labels/{file_id}.label"
                gt_sem, _ = load_labels(label_file)

                pred_file = f"{pred_dir}/{file_id}.txt"
                if os.path.exists(pred_file):
                    scores = np.loadtxt(pred_file).astype(np.float32)
                    metrics_calculator.update(points, scores, gt_sem)
                else:
                    print(f"Warning: Prediction file not found: {pred_file}")
            
            metrics = metrics_calculator.compute_metrics()
            if metrics:
                method_metrics[method] = metrics
                print(f"{method} AP: {metrics['AP']:.2f}, AUROC: {metrics['AUROC']:.2f}, FPR95: {metrics['FPR95']:.2f}")

        # Select best method based on AP
        # best_method = max(method_metrics.items(), key=lambda x: x[1]['AP'])[0]
        # Select best method based on FPR
        best_method = min(method_metrics.items(), key=lambda x: x[1]['FPR95'])[0]
        print(f"Best method for sequence {Seq}: {best_method}")
        all_best_scores.append(method_metrics[best_method]['scores'])
        all_best_labels.append(method_metrics[best_method]['labels'])

    # Compute global metrics using best-per-sequence selection
    final_scores = np.concatenate(all_best_scores)
    final_labels = np.concatenate(all_best_labels)

    final_ap = average_precision_score(final_labels, final_scores) * 100
    final_auroc = roc_auc_score(final_labels, final_scores) * 100

    # FPR95 calculation
    fpr, tpr, thresholds = roc_curve(final_labels, final_scores)
    fpr95 = fpr[np.searchsorted(tpr, 0.95)] * 100

    print("\n=== FINAL METRICS (Best-per-sequence) ===")
    print(f"AP: {final_ap:.2f}")
    print(f"AUROC: {final_auroc:.2f}")
    print(f"FPR95: {fpr95:.2f}")

if __name__ == "__main__":
    main()

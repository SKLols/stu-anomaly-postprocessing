"""
PQ Trainer — Panoptic Segmentation with Anomaly Scoring
=========================================================
This module extends the original STU PQ trainer with:

    1. Query Selection Framework: Filters Mask4Former-3D object queries
       based on spatial area thresholds before anomaly scoring, achieving
       ~62% query reduction while improving detection performance.

    2. Extended Uncertainty Methods: Implements MSP, SML, Energy, Entropy
       in addition to the original STU baselines (MaxLogit, RbA, Deep Ensemble).

Original baseline trainer: Mask4Former3D/trainer/pq_trainer.py
Thesis: "Query-Optimized 3D Anomaly Segmentation: Comprehensive Uncertainty
         Evaluation for Autonomous Driving", Sourabh Lolge, THI, 2025.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import statistics
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import hydra
import MinkowskiEngine as ME
import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.cluster import DBSCAN

from utils.utils import associate_instances
from query_selection.filter import select_queries
from uncertainty_methods.scoring import (
    score_maxlogit,
    score_msp,
    score_rba,
    score_entropy,
    score_energy,
    score_sml,
)

# ---------------------------------------------------------------------------
# Query selection configuration
# ---------------------------------------------------------------------------

USE_QUERY_SELECTION = True   # Toggle query selection on/off
QUERY_MIN_POINTS = 30        # Optimal threshold from ablation study (thesis Table 6.2)
QUERY_ACT_THRESHOLD = 0.1    # Sigmoid activation threshold for point counting


class PanopticSegmentation(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # SML per-class statistics (computed on training set)
        # TODO: move these paths to hydra config for portability
        # mean_dict = np.load("/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/Mask4Former3D/validate_stats_query/pointcloud_mean.npy", allow_pickle=True).item()
        # var_dict  = np.load("/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/Mask4Former3D/validate_stats_query/pointcloud_var.npy", allow_pickle=True).item()
        # mean_dict = np.load("/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/Mask4Former3D/validate_stats/pointcloud_mean.npy", allow_pickle=True).item()
        # var_dict  = np.load("/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/Mask4Former3D/validate_stats/pointcloud_var.npy", allow_pickle=True).item()
        #mean_dict = np.load("/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/Mask4Former3D/train_stats_frames_50_class_means.npy", allow_pickle=True).item()
        #var_dict  = np.load("/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/Mask4Former3D/train_stats_frames_50_class_vars.npy", allow_pickle=True).item()
        # mean_dict = np.load("/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/validate_stats/stu_calibration_COMBINED_all_sequences_50frames_means.npy", allow_pickle=True).item()
        # var_dict  = np.load("/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/validate_stats/stu_calibration_COMBINED_all_sequences_50frames_vars.npy", allow_pickle=True).item()

        #self.class_mean = torch.tensor(list(mean_dict.values())).float().to(self.device)
        #self.class_var  = torch.tensor(list(var_dict.values())).float().to(self.device)

        self.analyze_mode = True
        self.all_object_stats = []
        self.analyzed_frames = 0

        self.save_hyperparameters()

        # Model
        self.model = hydra.utils.instantiate(config.model)
        self.optional_freeze = nullcontext

        # Loss
        matcher = hydra.utils.instantiate(config.matcher)
        weight_dict = {
            "loss_ce": matcher.cost_class,
            "loss_mask": matcher.cost_mask,
            "loss_dice": matcher.cost_dice,
            "loss_box": matcher.cost_box,
        }
        aux_weight_dict = {}
        for i in range(self.model.num_levels * self.model.num_decoders):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
        self.criterion = hydra.utils.instantiate(
            config.loss, matcher=matcher, weight_dict=weight_dict
        )

        # Metrics
        self.class_evaluator = hydra.utils.instantiate(config.metric)
        self.last_seq = None

    # -----------------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------------

    def forward(self, x, raw_coordinates=None, is_eval=False):
        with self.optional_freeze():
            x = self.model(x, raw_coordinates=raw_coordinates, is_eval=is_eval)
        return x

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _save_score(
        score: torch.Tensor,
        method_name: str,
        sequences: list,
        b_idx: int,
        frame_str: str,
        save_dir: str,
    ) -> None:
        """
        Save anomaly scores for a single frame to disk.

        Saves both a .txt file (for STU evaluation protocol) and a .npy
        file (for downstream visualization and analysis).

        Args:
            score:       Anomaly score tensor, shape [N_raw_points].
            method_name: Name of the scoring method (e.g. 'msp', 'sml').
            sequences:   Batch sequence metadata list.
            b_idx:       Index within the current batch.
            frame_str:   Zero-padded frame ID string (e.g. '000125').
            save_dir:    Root directory for saving predictions.
        """
        save_path = Path(save_dir) / f"prediction_{method_name}" / sequences[b_idx][0]
        save_path.mkdir(parents=True, exist_ok=True)

        score_np = score.detach().cpu().numpy()
        np.savetxt(save_path / f"{sequences[b_idx][1]}.txt", score_np)
        np.save(
            save_path / f"{method_name}_{frame_str}.npy",
            score_np.astype(np.float32),
        )

    def save_query_masks(self, masked, inv_map, sequences, b_idx, batch_idx):
        """
        Save query masks mapped to raw point space as .txt files.

        Used for debugging and analysis of query activation patterns.

        Args:
            masked:     Sigmoid-activated mask tensor, shape [N_voxels, N_queries].
            inv_map:    Inverse map from voxels to raw points.
            sequences:  Batch sequence metadata.
            b_idx:      Batch index.
            batch_idx:  Global batch index.
        """
        import os
        os.makedirs("query_masks_dump", exist_ok=True)

        masked_raw = masked[inv_map]
        masked_np = masked_raw.detach().cpu().numpy()

        seq_name, frame_id = sequences[b_idx]
        frame_str = f"{int(frame_id):06d}"

        mask_filename = f"query_masks_dump/{seq_name}_{frame_str}_masks.txt"
        np.savetxt(mask_filename, masked_np, fmt="%.6f")
        print(f"[QueryMasks] Saved to: {mask_filename} (shape: {masked_np.shape})")

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        data, target = batch
        raw_coordinates = data.raw_coordinates
        data = ME.SparseTensor(
            coordinates=data.coordinates, features=data.features, device=self.device
        )
        output = self.forward(data, raw_coordinates=raw_coordinates)
        losses = self.criterion(output, target)

        for k in list(losses.keys()):
            if k in self.criterion.weight_dict:
                losses[k] *= self.criterion.weight_dict[k]
            else:
                losses.pop(k)

        logs = {f"train_{k}": v.detach().cpu().item() for k, v in losses.items()}
        logs["train_mean_loss_ce"] = statistics.mean(
            [v for k, v in logs.items() if "loss_ce" in k]
        )
        logs["train_mean_loss_mask"] = statistics.mean(
            [v for k, v in logs.items() if "loss_mask" in k]
        )
        logs["train_mean_loss_dice"] = statistics.mean(
            [v for k, v in logs.items() if "loss_dice" in k]
        )
        logs["train_mean_loss_box"] = statistics.mean(
            [v for k, v in logs.items() if "loss_box" in k]
        )
        self.log_dict(logs)
        return sum(losses.values())

    # -----------------------------------------------------------------------
    # Test — anomaly scoring
    # -----------------------------------------------------------------------

    def test_step(self, batch, batch_idx):
        """
        Run inference and compute anomaly scores for a single batch.

        Pipeline:
            1. Mask4Former-3D forward pass.
            2. Optional query selection (select_queries) — filters queries
               with fewer than QUERY_MIN_POINTS active raw points.
            3. Anomaly scoring using selected uncertainty method(s).
            4. Save scores to disk for STU evaluation protocol.

        Args:
            batch:     Tuple of (data, target) from the dataloader.
            batch_idx: Index of the current batch.

        Returns:
            Empty dict (scores saved to disk).
        """
        torch.cuda.empty_cache()

        data, target = batch
        inverse_maps = data.inverse_maps
        raw_coordinates = data.raw_coordinates
        sequences = data.sequences

        data = ME.SparseTensor(
            coordinates=data.coordinates,
            features=data.features,
            device=self.device,
        )
        output = self.forward(data, raw_coordinates=raw_coordinates, is_eval=True)

        pred_logits = output["pred_logits"]
        pred_logits = torch.functional.F.softmax(pred_logits, dim=-1)[..., :-1]
        pred_masks = output["pred_masks"]

        for b_idx in range(len(pred_logits)):
            frame_str = f"{int(sequences[b_idx][1]):06d}"
            inv_map = inverse_maps[b_idx]

            # Keep original (full queries) always
            mask_full = pred_masks[b_idx].float().sigmoid()
            logit_full = pred_logits[b_idx]

            # Selected queries
            mask_qs, logit_qs = select_queries(
                mask=mask_full,
                logit=logit_full,
                inv_map=inv_map,
                min_points=QUERY_MIN_POINTS,
                activation_threshold=QUERY_ACT_THRESHOLD,
            )

            # ------------------------------------------------------------------
            # Anomaly scoring — runs BOTH full and selected queries
            # Saved separately: "maxlogit" vs "maxlogit_qs"
            # ------------------------------------------------------------------

            # --- MaxLogit ---
            maxlogit = score_maxlogit(mask_full, logit_full, inv_map)
            self._save_score(maxlogit, "maxlogit", sequences, b_idx, frame_str,
                             self.config.general.save_dir)

            maxlogit_qs = score_maxlogit(mask_qs, logit_qs, inv_map)
            self._save_score(maxlogit_qs, "maxlogit_qs", sequences, b_idx, frame_str,
                             self.config.general.save_dir)

            # --- MSP ---
            msp = score_msp(mask_full, logit_full, inv_map)
            self._save_score(msp, "msp", sequences, b_idx, frame_str,
                             self.config.general.save_dir)

            msp_qs = score_msp(mask_qs, logit_qs, inv_map)
            self._save_score(msp_qs, "msp_qs", sequences, b_idx, frame_str,
                             self.config.general.save_dir)

            # --- RbA ---
            rba = score_rba(mask_full, logit_full, inv_map)
            self._save_score(rba, "rba", sequences, b_idx, frame_str,
                             self.config.general.save_dir)

            rba_qs = score_rba(mask_qs, logit_qs, inv_map)
            self._save_score(rba_qs, "rba_qs", sequences, b_idx, frame_str,
                             self.config.general.save_dir)

            # --- Entropy ---
            entropy = score_entropy(mask_full, logit_full, inv_map)
            self._save_score(entropy, "entropy", sequences, b_idx, frame_str,
                             self.config.general.save_dir)

            entropy_qs = score_entropy(mask_qs, logit_qs, inv_map)
            self._save_score(entropy_qs, "entropy_qs", sequences, b_idx, frame_str,
                             self.config.general.save_dir)

            # --- Energy ---
            energy = score_energy(mask_full, logit_full, inv_map)
            self._save_score(energy, "energy", sequences, b_idx, frame_str,
                             self.config.general.save_dir)

            energy_qs = score_energy(mask_qs, logit_qs, inv_map)
            self._save_score(energy_qs, "energy_qs", sequences, b_idx, frame_str,
                             self.config.general.save_dir)

            # --- SML ---
            # sml = score_sml(mask_full, logit_full, inv_map,
            #                 class_mean=self.class_mean, class_var=self.class_var)
            # self._save_score(sml, "sml", sequences, b_idx, frame_str,
            #                  self.config.general.save_dir)

            # sml_qs = score_sml(mask_qs, logit_qs, inv_map,
            #                    class_mean=self.class_mean, class_var=self.class_var)
            # self._save_score(sml_qs, "sml_qs", sequences, b_idx, frame_str,
            #                  self.config.general.save_dir)

        return {}

    # -----------------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------------

    def validation_step(self, batch, batch_idx):
        data, target = batch
        inverse_maps = data.inverse_maps
        original_labels = data.original_labels
        raw_coordinates = data.raw_coordinates
        num_points = data.num_points
        sequences = data.sequences

        data = ME.SparseTensor(
            coordinates=data.coordinates, features=data.features, device=self.device
        )
        output = self.forward(data, raw_coordinates=raw_coordinates, is_eval=True)
        losses = self.criterion(output, target)

        for k in list(losses.keys()):
            if k in self.criterion.weight_dict:
                losses[k] *= self.criterion.weight_dict[k]
            else:
                losses.pop(k)

        pred_logits = output["pred_logits"]
        pred_logits = torch.functional.F.softmax(pred_logits, dim=-1)[..., :-1]
        pred_masks = output["pred_masks"]

        offset_coords_idx = 0
        for logit, mask, map, label, n_point, seq in zip(
            pred_logits,
            pred_masks,
            inverse_maps,
            original_labels,
            num_points,
            sequences,
        ):
            seq = seq[0]
            if seq != self.last_seq:
                self.last_seq = seq
                self.previous_instances = None
                self.max_instance_id = self.config.model.num_queries
                self.scene = 0

            class_confidence, classes = torch.max(logit.detach().cpu(), dim=1)
            foreground_confidence = mask.detach().cpu().float().sigmoid()
            confidence = class_confidence[None, ...] * foreground_confidence
            confidence = confidence[map].numpy()
            ins_preds = np.argmax(confidence, axis=1)
            sem_preds = classes[ins_preds].numpy() + 1
            ins_preds += 1
            ins_preds[
                np.isin(
                    sem_preds,
                    range(1, self.config.data.min_stuff_cls_id),
                    invert=True,
                )
            ] = 0

            sem_labels = self.validation_dataset._remap_model_output(label[:, 0])
            ins_labels = label[:, 1] >> 16

            db_max_instance_id = self.config.model.num_queries
            if self.config.general.dbscan_eps is not None:
                curr_coords_idx = mask.shape[0]
                curr_coords = raw_coordinates[
                    offset_coords_idx : curr_coords_idx + offset_coords_idx, :3
                ]
                curr_coords = curr_coords[map].detach().cpu().numpy()
                offset_coords_idx += curr_coords_idx

                ins_ids = np.unique(ins_preds)
                for ins_id in ins_ids:
                    if ins_id != 0:
                        instance_mask = ins_preds == ins_id
                        clusters = (
                            DBSCAN(
                                eps=self.config.general.dbscan_eps,
                                min_samples=1,
                                n_jobs=-1,
                            )
                            .fit(curr_coords[instance_mask])
                            .labels_
                        )
                        new_mask = np.zeros(ins_preds.shape, dtype=np.int64)
                        new_mask[instance_mask] = clusters + 1
                        for cluster_id in np.unique(new_mask):
                            if cluster_id != 0:
                                db_max_instance_id += 1
                                ins_preds[new_mask == cluster_id] = db_max_instance_id

            self.max_instance_id = max(db_max_instance_id, self.max_instance_id)

            for i in range(len(n_point) - 1):
                indices = range(n_point[i], n_point[i + 1])
                if i == 0 and self.previous_instances is not None:
                    current_instances = ins_preds[indices]
                    associations = associate_instances(
                        self.previous_instances, current_instances
                    )
                    for id in np.unique(ins_preds):
                        if associations.get(id) is None:
                            self.max_instance_id += 1
                            associations[id] = self.max_instance_id
                    ins_preds = np.vectorize(associations.__getitem__)(ins_preds)
                else:
                    self.class_evaluator.addBatch(
                        sem_preds, ins_preds, sem_labels, ins_labels, indices, seq
                    )

            if i > 0:
                self.previous_instances = ins_preds[indices]

        return {f"val_{k}": v.detach().cpu().item() for k, v in losses.items()}

    # -----------------------------------------------------------------------
    # Epoch end hooks
    # -----------------------------------------------------------------------

    def training_epoch_end(self, outputs):
        train_loss = sum([out["loss"].cpu().item() for out in outputs]) / len(outputs)
        self.log_dict({"train_loss_mean": train_loss})

    def validation_epoch_end(self, outputs):
        self.last_seq = None
        class_names = self.config.data.class_names
        pq, sq, rq, all_pq, all_sq, all_rq = self.class_evaluator.getPQ()
        self.class_evaluator.reset()

        results = {
            "val_mean_pq": pq,
            "val_mean_sq": sq,
            "val_mean_rq": rq,
        }
        for i, (pq, sq, rq) in enumerate(zip(all_pq, all_sq, all_rq)):
            results[f"val_{class_names[i-1]}_pq"] = pq.item()
            results[f"val_{class_names[i-1]}_sq"] = sq.item()
            results[f"val_{class_names[i-1]}_rq"] = rq.item()

        self.log_dict(results)
        print(results)

        dd = defaultdict(list)
        for output in outputs:
            for key, val in output.items():
                dd[key].append(val)
        dd = {k: statistics.mean(v) for k, v in dd.items()}
        dd["val_mean_loss_ce"] = statistics.mean(
            [v for k, v in dd.items() if "loss_ce" in k]
        )
        dd["val_mean_loss_mask"] = statistics.mean(
            [v for k, v in dd.items() if "loss_mask" in k]
        )
        dd["val_mean_loss_dice"] = statistics.mean(
            [v for k, v in dd.items() if "loss_dice" in k]
        )
        dd["val_mean_loss_box"] = statistics.mean(
            [v for k, v in dd.items() if "loss_box" in k]
        )
        self.log_dict(dd)

    def test_epoch_end(self, outputs):
        return {}

    # -----------------------------------------------------------------------
    # Optimizers and dataloaders
    # -----------------------------------------------------------------------

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(
            self.config.optimizer, params=self.parameters()
        )
        if "steps_per_epoch" in self.config.scheduler.scheduler.keys():
            self.config.scheduler.scheduler.steps_per_epoch = len(
                self.train_dataloader()
            )
        lr_scheduler = hydra.utils.instantiate(
            self.config.scheduler.scheduler, optimizer=optimizer
        )
        scheduler_config = {"scheduler": lr_scheduler}
        scheduler_config.update(self.config.scheduler.pytorch_lightning_params)
        return [optimizer], [scheduler_config]

    def prepare_data(self):
        self.train_dataset = hydra.utils.instantiate(self.config.data.train_dataset)
        self.validation_dataset = hydra.utils.instantiate(
            self.config.data.validation_dataset
        )
        self.test_dataset = hydra.utils.instantiate(self.config.data.test_dataset)

    def train_dataloader(self):
        c_fn = hydra.utils.instantiate(self.config.data.train_collation)
        return hydra.utils.instantiate(
            self.config.data.train_dataloader,
            self.train_dataset,
            collate_fn=c_fn,
        )

    def val_dataloader(self):
        c_fn = hydra.utils.instantiate(self.config.data.validation_collation)
        return hydra.utils.instantiate(
            self.config.data.validation_dataloader,
            self.validation_dataset,
            collate_fn=c_fn,
        )

    def test_dataloader(self):
        c_fn = hydra.utils.instantiate(self.config.data.test_collation)
        return hydra.utils.instantiate(
            self.config.data.test_dataloader,
            self.test_dataset,
            collate_fn=c_fn,
        )
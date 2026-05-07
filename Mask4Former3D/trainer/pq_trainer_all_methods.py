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

import os
import math

import time


class PanopticSegmentation(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config

        # Load dictionary from .npy (query logic during stats)
        # mean_dict = np.load("/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/Mask4Former3D/validate_stats_query/pointcloud_mean.npy", allow_pickle=True).item()
        # var_dict  = np.load("/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/Mask4Former3D/validate_stats_query/pointcloud_var.npy", allow_pickle=True).item()
        # Load dictionary from .npy (confid logic during stats)
        # mean_dict = np.load("/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/Mask4Former3D/validate_stats/pointcloud_mean.npy", allow_pickle=True).item()
        # var_dict  = np.load("/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/Mask4Former3D/validate_stats/pointcloud_var.npy", allow_pickle=True).item()
        
        #Load dictionary from .npy (confid logic during stats)
        mean_dict = np.load("/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/Mask4Former3D/train_stats_frames_50_class_means.npy", allow_pickle=True).item()
        var_dict  = np.load("/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/Mask4Former3D/train_stats_frames_50_class_vars.npy", allow_pickle=True).item()

        # #Load dictionary from .npy (confid logic during stats)
        # mean_dict_no_inv = np.load("/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/class_means_test_19_v9.npy", allow_pickle=True).item()
        # var_dict_no_inv  = np.load("/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/class_variances_test_19_v9.npy", allow_pickle=True).item()


        
        # #Load dictionary from .npy (confid calibrated sample all combined)
        # mean_dict = np.load("/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/validate_stats/stu_calibration_COMBINED_all_sequences_50frames_means.npy", allow_pickle=True).item()
        # var_dict  = np.load("/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/validate_stats/stu_calibration_COMBINED_all_sequences_50frames_vars.npy", allow_pickle=True).item()
 
        
        
        # Convert to tensors in the correct order
        self.class_mean = torch.tensor(list(mean_dict.values())).float().to(self.device)
        self.class_var  = torch.tensor(list(var_dict.values())).float().to(self.device)

        # self.class_mean_no_inv = torch.tensor(list(mean_dict_no_inv.values())).float().to(self.device)
        # self.class_var_no_inv  = torch.tensor(list(var_dict_no_inv.values())).float().to(self.device)

        self.analyze_mode = True
        self.all_object_stats = []  # Global statistics storage
        self.analyzed_frames = 0

        self.save_hyperparameters()
        # model
        self.model = hydra.utils.instantiate(config.model)
        self.optional_freeze = nullcontext

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
        # metrics
        self.class_evaluator = hydra.utils.instantiate(config.metric)
        self.last_seq = None

    def forward(self, x, raw_coordinates=None, is_eval=False):
        with self.optional_freeze():
            x = self.model(x, raw_coordinates=raw_coordinates, is_eval=is_eval)
        return x

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
                # remove this loss if not specified in `weight_dict`
                losses.pop(k)

        logs = {f"train_{k}": v.detach().cpu().item() for k, v in losses.items()}

        logs["train_mean_loss_ce"] = statistics.mean(
            [item for item in [v for k, v in logs.items() if "loss_ce" in k]]
        )

        logs["train_mean_loss_mask"] = statistics.mean(
            [item for item in [v for k, v in logs.items() if "loss_mask" in k]]
        )

        logs["train_mean_loss_dice"] = statistics.mean(
            [item for item in [v for k, v in logs.items() if "loss_dice" in k]]
        )

        logs["train_mean_loss_box"] = statistics.mean(
            [item for item in [v for k, v in logs.items() if "loss_box" in k]]
        )

        self.log_dict(logs)
        return sum(losses.values())  

    def save_pred_logits(self, pred_logits, batch_idx):
        """Save raw logits to a text file with batch index"""
        os.makedirs('logits_full_dump', exist_ok=True)
        filename = f'logits_full_dump/batch_{batch_idx}_logits_1.txt'

        # Convert to numpy and save
        np.savetxt(filename, pred_logits.cpu().numpy().reshape(-1, pred_logits.shape[-1]))
        print(f"Saved raw logits to {filename}")
    
    def save_pred_logits_sm(self, pred_logits, batch_idx):
        """Save raw logits to a text file with batch index"""
        os.makedirs('logits_full_dump', exist_ok=True)
        filename = f'logits_full_dump/batch_{batch_idx}_logits_sm_1.txt'

        # Convert to numpy and save
        np.savetxt(filename, pred_logits.cpu().numpy().reshape(-1, pred_logits.shape[-1]))
        print(f"Saved raw logit_sm to {filename}")

    def save_pred_mask(self, pred_masks, batch_idx):
        """Save raw logits to a text file with batch index"""
        os.makedirs('logits_full_dump', exist_ok=True)
        filename = f'logits_full_dump/batch_{batch_idx}_masks_1.txt'

        if isinstance(pred_masks, list):
            # Case 1: pred_masks is a list of tensors
            mask_array = torch.cat([m.cpu() for m in pred_masks], dim=0).numpy()
        elif isinstance(pred_masks, torch.Tensor):
            # Case 2: pred_masks is already a tensor
            mask_array = pred_masks.cpu().numpy()
        else:
            raise ValueError(f"Unsupported mask type: {type(pred_masks)}")
    
        # Save with 4 decimal places
        np.savetxt(filename, mask_array, fmt='%.4f')
        print(f"Saved masks to {filename} (shape: {mask_array.shape})")

    def save_remapping_data(self, batch_data, batch_idx):
        """Save inverse maps in .txt format for short-term visualization"""
        os.makedirs('logits_full_dump', exist_ok=True)
        
        # Save inverse maps as .txt (same format as your other files)
        inverse_maps = batch_data.inverse_maps[0].cpu().numpy()
        np.savetxt(f'logits_full_dump/batch_{batch_idx}_inverse_maps.txt', 
                   inverse_maps, fmt='%d')  # Save as integers
        
        # Save raw coordinates (optional but recommended)
        raw_coords = batch_data.raw_coordinates.cpu().numpy()
        np.savetxt(f'logits_full_dump/batch_{batch_idx}_raw_coords.txt', 
                   raw_coords, fmt='%.6f')  # 6 decimal places for coordinates
        
        print(f"Saved remapping data for batch {batch_idx} (TXT format)")

    def save_query_masks(self, masked, inv_map, sequences, b_idx, batch_idx):
        """Save query masks as [N_raw_points, 100] .txt files"""
        os.makedirs('query_masks_dump', exist_ok=True)
        
        # Map to raw points: [N_voxels, 100] → [N_raw_points, 100]
        masked_raw = masked[inv_map]  # This is what you want!
        masked_np = masked_raw.detach().cpu().numpy()  # [N_raw_points, 100]
        
        # Get sequence info
        seq_name, frame_id = sequences[b_idx]
        frame_str = f"{int(frame_id):06d}"
        
        # Save as .txt
        mask_filename = f'query_masks_dump/{seq_name}_{frame_str}_masks.txt'
        np.savetxt(mask_filename, masked_np, fmt='%.6f')
        
        print(f"Saved query masks to: {mask_filename} (shape: {masked_np.shape})")

    def test_step(self, batch, batch_idx):
        torch.cuda.empty_cache()

        data, target = batch
        # self.save_remapping_data(data, batch_idx)
        inverse_maps = data.inverse_maps
        raw_coordinates = data.raw_coordinates
        sequences = data.sequences

        data = ME.SparseTensor(
            coordinates=data.coordinates, features=data.features, device=self.device
        )

        output = self.forward(data, raw_coordinates=raw_coordinates, is_eval=True)

        def normalize(
                x
                ):
            """
            Normalize tensor to [0, 1] range with numerical stability.
            
            Args:
                x: Input tensor to normalize
                
            Returns:
                Tensor normalized to [0, 1] range
            """
            return (x - x.min()) / (x.max() - x.min() + 1e-8)

        def get_maxlogit(
                logit, 
                mask, 
                inv_map
                ):
            """
            Calculate maximum logit-based anomaly scores.
            
            Args:
                logit: Raw output logits from model [C, H, W] or [B, C, H, W]
                mask: Binary mask tensor for filtering relevant regions
                inv_map: Tensor for reordering/remapping the output to original space
                
            Returns:
                Calibrated anomaly scores between 0 and 1, where higher values indicate more anomalous
            """
            confid = mask.float().sigmoid().matmul(logit)
            max_logit = torch.max(confid, dim=1).values[inv_map]
            max_logit = (max_logit * -1) + 1
            # base_min, base_max = max_logit.min(), max_logit.max()
            # base_mean, base_std = max_logit.mean(), max_logit.std()
            # print(f"Max logit scores range - Min: {base_min:.3f}, Max: {base_max:.3f}, Mean: {base_mean:.3f}, Std: {base_std:.3f}")
            # max_logit_calib = normalize(max_logit)
            # return max_logit_calib
            return max_logit       
        
        def get_maxlogit_pre(
                self,
                logit, 
                mask, 
                inv_map,
                sequences,
                b_idx,
                batch_idx
                ):
            """
            Calculate maximum logit-based anomaly scores.
            """
            masked = mask.float().sigmoid()  # [N_voxels, 100]
            
            # ✅ SAVE QUERY MASKS MAPPED TO RAW POINTS
            self.save_query_masks(masked, inv_map, sequences, b_idx, batch_idx)
            
            # Continue with original logic
            confid = masked.matmul(logit)
            # self.save_confid_query_masks(confid, inv_map, sequences, b_idx, batch_idx)
            max_logit = torch.max(confid, dim=1).values[inv_map]
            max_logit_final = (max_logit * -1) + 1
            max_logit_calib = normalize(max_logit_final)
            
            return max_logit_calib 
        
        def get_msp(
                logit, 
                mask, 
                inv_map
                ):
            """
            Calculate Maximum Softmax Probability (MSP) based anomaly detection.
            
            Args:
                logit: Raw output logits from model [C, H, W] or [B, C, H, W]
                mask: Binary mask tensor for filtering relevant regions  
                inv_map: Tensor for reordering/remapping the output to original space
                
            Returns:
                Anomaly scores between 0 and 1, where 1 represents maximum uncertainty/anomaly
            """
            confid = mask.float().sigmoid().matmul(logit)
            probs = torch.functional.F.softmax(confid, dim=1)
            msp = torch.max(probs, dim=1).values
            msp = msp[inv_map]
            anomaly_score = 1 - msp
            # base_min, base_max = anomaly_score.min(), anomaly_score.max()
            # base_mean, base_std = anomaly_score.mean(), anomaly_score.std()
            # print(f"rba logit scores range - Min: {base_min:.3f}, Max: {base_max:.3f}, Mean: {base_mean:.3f}, Std: {base_std:.3f}")
            
            return anomaly_score      
  

        def get_rba(
                logit, 
                mask, 
                inv_map
                ):
            """
            Calculate Reject by Acceptance based anomaly scores.
            
            Args:
                logit: Raw output logits from model [C, H, W] or [B, C, H, W]
                mask: Binary mask tensor for filtering relevant regions
                inv_map: Tensor for reordering/remapping the output to original space
                
            Returns:
                Calibrated RbA anomaly scores normalized between 0 and 1
            """
            confid = mask.float().sigmoid().matmul(logit)
            rba = -confid.tanh().sum(dim=1)[inv_map]
            rba[rba < -1] = -1
            rba = rba + 1
            # base_min, base_max = rba.min(), rba.max()
            # base_mean, base_std = rba.mean(), rba.std()
            # print(f"rba logit scores range - Min: {base_min:.3f}, Max: {base_max:.3f}, Mean: {base_mean:.3f}, Std: {base_std:.3f}")
            # rba_calib = normalize(rba)
            # return rba_calib
            return rba
        
        def get_entropy(
                logit, 
                mask, 
                inv_map
                ):
            """
            Calculate entropy based uncertainty scores for anomaly detection.
            
            Args:
                logit: Raw output logits from model [C, H, W] or [B, C, H, W]
                mask: Binary mask tensor for filtering
                inv_map: Tensor for reordering/remapping the output
                
            Returns:
                Calibrated energy scores normalized between 0 and 1
            """
            confid = mask.float().sigmoid().matmul(logit)
            probs = torch.functional.F.softmax(confid, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)[inv_map]
            base_min, base_max = entropy.min(), entropy.max()
            base_mean, base_std = entropy.mean(), entropy.std()
            print(f"entropy logit scores range - Min: {base_min:.3f}, Max: {base_max:.3f}, Mean: {base_mean:.3f}, Std: {base_std:.3f}")
            # entropy_calib = normalize(entropy)
            # return entropy_calib
            return entropy
        
        def get_energy(
                logit, 
                mask, 
                inv_map
                ):
            """
            Calculate calibrated energy scores from logits and mask.
            
            Args:
                logit: Raw output logits from model [C, H, W] or [B, C, H, W]
                mask: Binary mask tensor for filtering
                inv_map: Tensor for reordering/remapping the output
                
            Returns:
                Calibrated energy scores normalized between 0 and 1
            """
            confid = mask.float().sigmoid().matmul(logit)
            energy = -torch.logsumexp(confid, dim=1)[inv_map]
            base_min, base_max = energy.min(), energy.max()
            base_mean, base_std = energy.mean(), energy.std()
            print(f"energy logit scores range - Min: {base_min:.3f}, Max: {base_max:.3f}, Mean: {base_mean:.3f}, Std: {base_std:.3f}")
            energy_calib = normalize(energy)
            return energy_calib
            # return energy
        
        def get_sml(
                logit, 
                mask, 
                inv_map, 
                class_mean, 
                class_var
                ):
            """
            Calculate Standardized Max Logits (SML) for anomaly detection.
            
            Args:
                logit: Raw output logits from model [C, H, W] or [B, C, H, W]
                mask: Binary mask tensor for filtering relevant regions
                inv_map: Tensor for reordering/remapping the output to original space
                class_mean: Mean values for each class for standardization
                class_var: Variance values for each class for standardization
                
            Returns:
                Standardized max logit scores for anomaly detection
            """
            confid = mask.float().sigmoid().matmul(logit)       # [N_points, num_classes]
            max_logit, prediction = torch.max(confid, dim=1)    # max logit per point and predicted class
            max_logit = max_logit[inv_map]                      # map back to raw points
            prediction = prediction[inv_map]                    # same for classes

            # max_logit = (max_logit * -1) + 1

            # sml = max_logit.clone()
            sml = torch.zeros_like(max_logit)

            for c in range(len(class_mean)):
                mask_c = prediction == c
                if mask_c.any():
                    eps = 1e-6
                    # sml[mask_c] = ((max_logit[mask_c] - class_mean[c]) / torch.sqrt(class_var[c]+eps))
                    
                    z = (max_logit[mask_c] - class_mean[c]) / torch.sqrt(class_var[c] + eps)

                    y = (max_logit[mask_c] - class_mean[c])** 2 / 19
        
                    # Clip POSITIVE z-scores (normal) to +1
                    # z = torch.clamp(z, max=1.0)  # Clip positive side!
                    sml[mask_c] = z + y
                    # sml_calib = normalize(sml)
            sml = (sml * -1) + 1

            return sml   
        
        def get_proper_sml(logit, mask, inv_map, class_mean, class_var):
            confid = mask.float().sigmoid().matmul(logit)       
            max_logit, prediction = torch.max(confid, dim=1)    
            max_logit = max_logit[inv_map]                      
            prediction = prediction[inv_map]                    

            # Standardize
            sml = torch.zeros_like(max_logit)
            
            for c in range(len(class_mean)):
                mask_c = prediction == c
                if mask_c.any():
                    eps = 1e-6
                    sml[mask_c] = ((max_logit[mask_c] - class_mean[c]) / torch.sqrt(class_var[c] + eps))
            
            # Apply same inversion as your MaxLogit for consistent comparison
            sml = (sml * -1) + 1  # Higher = more anomalous
            return sml

        def get_sml_calib(
                logit, 
                mask, 
                inv_map, 
                class_mean, 
                class_var
                ):
            """
            Calculate Standardized Max Logits (SML) for anomaly detection.
            
            Args:
                logit: Raw output logits from model [C, H, W] or [B, C, H, W]
                mask: Binary mask tensor for filtering relevant regions
                inv_map: Tensor for reordering/remapping the output to original space
                class_mean: Mean values for each class for standardization
                class_var: Variance values for each class for standardization
                
            Returns:
                Standardized max logit scores for anomaly detection
            """
            confid = mask.float().sigmoid().matmul(logit)       # [N_points, num_classes]
            max_logit, prediction = torch.max(confid, dim=1)    # max logit per point and predicted class
            max_logit = max_logit[inv_map]                      # map back to raw points
            prediction = prediction[inv_map]                    # same for classes

            max_logit = (max_logit * -1) + 1

            sml = max_logit.clone()

            for c in range(len(class_mean)):
                mask_c = prediction == c
                if mask_c.any():
                    eps = 1e-6
                    sml[mask_c] = ((max_logit[mask_c] - class_mean[c]) / torch.sqrt(class_var[c]+eps))
                    # sml = (sml * -1)
                    sml_calib = normalize(sml)

            return sml_calib     

        def min_max_calibrate(scores, min_quantile=0.001, max_quantile=0.999, eps=1e-6):
            
            q_min = torch.quantile(scores, min_quantile)
            q_max = torch.quantile(scores, max_quantile)
            
            calibrated = (scores - q_min) / (q_max - q_min + eps)
            calibrated = torch.clamp(calibrated, 0, 1)
            
            return calibrated
        
        def get_ensemble(
                logit, 
                mask, 
                inv_map
                ):
            confid = mask.float().sigmoid().matmul(logit)
            ensemble = confid[inv_map]
            print(f"Ensemble shape: {ensemble.shape}")
         
            return ensemble           

        def save_anomaly_score(
                score, 
                method_name, 
                sequences, 
                b_idx, 
                frame_str, 
                save_dir
                ):
            """
            Save anomaly scores for a given method.
            
            Args:
                score: The computed score tensor
                method_name: Name of the method (maxlogit, msp, entropy, etc.)
                sequences: Sequences data
                b_idx: Batch index
                frame_str: Formatted frame string
                save_dir: Base save directory
            """
            save_path = Path(save_dir) / f"prediction_{method_name}"
            base_path = save_path / f"{sequences[b_idx][0]}"
            original_save_path = base_path / f"{sequences[b_idx][1]}.txt"
            
            if not original_save_path.parent.exists():
                original_save_path.parent.mkdir(exist_ok=True, parents=True)
                
            np.savetxt(original_save_path, score.detach().cpu().numpy())
            np.save(base_path / f"{method_name}_{frame_str}.npy", 
                    score.detach().cpu().numpy().astype(np.float32))
            
        #====================================================================================0

        # Modified into large groups but normal maxlogit process no objectomaly
        # Query_maxlogit
        
        def get_query_based_anomaly_maxlogit(
                logit, 
                mask, 
                inv_map,
                min_points=1000,  # Only queries with 1000+ points
                threshold=0.1   
            ):
            """
            Use only large queries (1000+ points) for inlier scoring
            """
            masked = mask.float().sigmoid()
            
            # Map voxel masks to raw points
            masked_np = masked.detach().cpu().numpy()
            inv_map_np = inv_map.cpu().numpy()
            
            N_raw_points = len(inv_map_np)
            masked_raw = np.zeros((N_raw_points, masked_np.shape[1]))
            
            for raw_point_idx, voxel_idx in enumerate(inv_map_np):
                masked_raw[raw_point_idx, :] = masked_np[voxel_idx, :]
            
            # Select only queries with 1000+ points
            large_query_indices = []
            for query_idx in range(masked_raw.shape[1]):
                active_points = np.where(masked_raw[:, query_idx] > threshold)[0]
                num_points = len(active_points)
                if num_points >= min_points:  # Only 1000+ points
                    large_query_indices.append(query_idx)
            
            print(f"Selected {len(large_query_indices)} large queries (>=1000 points)")
            
            # Use only large queries for confidence calculation
            if len(large_query_indices) > 0:
                selected_masked = masked[:, large_query_indices]
                selected_logits = logit[large_query_indices]
                confid = selected_masked.matmul(selected_logits)
                print("Large queries working")
            else:
                # Fallback: use all queries if no large ones found
                confid = masked.matmul(logit)
                print("No large queries found. Using all queries as fallback")
            
            #Maxlogit
            max_logit = torch.max(confid, dim=1).values
            base_anomaly = max_logit[inv_map]
            base_anomaly_np = base_anomaly.detach().cpu().numpy()
        

            # Debug scores
            base_min, base_max = base_anomaly_np.min(), base_anomaly_np.max()
            base_mean, base_std = base_anomaly_np.mean(), base_anomaly_np.std()
            print(f"Large queries scores - Min: {base_min:.3f}, Max: {base_max:.3f}, Mean: {base_mean:.3f}, Std: {base_std:.3f}")

            # No overlap processing - use raw scores directly
            final_anomaly_raw = base_anomaly_np.copy()
            
            # Simple coverage calculation
            coverage_mask = final_anomaly_raw > 0
            coverage_percentage = (np.sum(coverage_mask) / N_raw_points) * 100
            print(f"Coverage: {np.sum(coverage_mask)}/{N_raw_points} points ({coverage_percentage:.1f}%)")
            
            final_anomaly_tensor = torch.from_numpy(final_anomaly_raw).float().to(logit.device)
            final_anomaly_inverted = (final_anomaly_tensor * -1) + 1
            # final_anomaly_inverted = normalize(final_anomaly_inverted)
            # final_anomaly_inverted = (final_anomaly_tensor)
            
            return final_anomaly_inverted
        
        #=====================================================================================================
        
        # Modified into large groups but normal maxlogit process no objectomaly
        # Query_maxlogit
        def get_query_based_anomaly_rba(
                logit, 
                mask, 
                inv_map,
                min_points=1000,  # Only queries with 1000+ points
                threshold=0.1   
            ):
            """
            Use only large queries (1000+ points) for inlier scoring
            """
            masked = mask.float().sigmoid()
            
            # Map voxel masks to raw points
            masked_np = masked.detach().cpu().numpy()
            inv_map_np = inv_map.cpu().numpy()
            
            N_raw_points = len(inv_map_np)
            masked_raw = np.zeros((N_raw_points, masked_np.shape[1]))
            
            for raw_point_idx, voxel_idx in enumerate(inv_map_np):
                masked_raw[raw_point_idx, :] = masked_np[voxel_idx, :]
            
            # Select only queries with 1000+ points
            large_query_indices = []
            for query_idx in range(masked_raw.shape[1]):
                active_points = np.where(masked_raw[:, query_idx] > threshold)[0]
                num_points = len(active_points)
                if num_points >= min_points:  # Only 1000+ points
                    large_query_indices.append(query_idx)
            
            print(f"Selected {len(large_query_indices)} large queries (>=1000 points)")
            
            # Use only large queries for confidence calculation
            if len(large_query_indices) > 0:
                selected_masked = masked[:, large_query_indices]
                selected_logits = logit[large_query_indices]
                confid = selected_masked.matmul(selected_logits)
                print("Large queries working")
            else:
                # Fallback: use all queries if no large ones found
                confid = masked.matmul(logit)
                print("No large queries found. Using all queries as fallback")
            
            # #Maxlogit
            # max_logit = torch.max(confid, dim=1).values
            # base_anomaly = max_logit[inv_map]
            # base_anomaly_np = base_anomaly.detach().cpu().numpy()

            # # Debug scores
            # base_min, base_max = base_anomaly_np.min(), base_anomaly_np.max()
            # base_mean, base_std = base_anomaly_np.mean(), base_anomaly_np.std()
            # print(f"Large queries scores - Min: {base_min:.3f}, Max: {base_max:.3f}, Mean: {base_mean:.3f}, Std: {base_std:.3f}")

            #RBA
            rba = -confid.tanh().sum(dim=1)[inv_map]  # RBA scoring
            rba[rba < -1] = -1
            rba = rba + 1
            # rba_calib = normalize(rba)
            base_anomaly_np = rba.detach().cpu().numpy()

            # Debug scores
            base_min, base_max = base_anomaly_np.min(), base_anomaly_np.max()
            base_mean, base_std = base_anomaly_np.mean(), base_anomaly_np.std()
            print(f"RBA scores - Min: {base_min:.3f}, Max: {base_max:.3f}, Mean: {base_mean:.3f}, Std: {base_std:.3f}")
            
            # No overlap processing - use raw scores directly
            final_anomaly_raw = base_anomaly_np.copy()
            
            # Simple coverage calculation
            coverage_mask = final_anomaly_raw > 0
            coverage_percentage = (np.sum(coverage_mask) / N_raw_points) * 100
            print(f"Coverage: {np.sum(coverage_mask)}/{N_raw_points} points ({coverage_percentage:.1f}%)")
            
            final_anomaly_tensor = torch.from_numpy(final_anomaly_raw).float().to(logit.device)
            # final_anomaly_inverted = (final_anomaly_tensor * -1) + 1
            final_anomaly_inverted = (final_anomaly_tensor)
            
            return final_anomaly_inverted
        
        #==============================================================================================00
        # Modified into large groups but normal maxlogit process no objectomaly
        # Query_msp
        def get_query_based_anomaly_msp(
                logit, 
                mask, 
                inv_map,
                min_points=1000,  # Only queries with 1000+ points
                threshold=0.1   
            ):
            """
            Use only large queries (1000+ points) for inlier scoring
            """
            masked = mask.float().sigmoid()
            
            # Map voxel masks to raw points
            masked_np = masked.detach().cpu().numpy()
            inv_map_np = inv_map.cpu().numpy()
            
            N_raw_points = len(inv_map_np)
            masked_raw = np.zeros((N_raw_points, masked_np.shape[1]))
            
            for raw_point_idx, voxel_idx in enumerate(inv_map_np):
                masked_raw[raw_point_idx, :] = masked_np[voxel_idx, :]
            
            # Select only queries with 1000+ points
            large_query_indices = []
            for query_idx in range(masked_raw.shape[1]):
                active_points = np.where(masked_raw[:, query_idx] > threshold)[0]
                num_points = len(active_points)
                if num_points >= min_points:  # Only 1000+ points
                    large_query_indices.append(query_idx)
            
            print(f"Selected {len(large_query_indices)} large queries (>=1000 points)")
    
            # Step 2: Further filter queries based on high confidence points (>=0.9)
            high_confidence_query_indices = []
            high_confidence_point_counts = []
            
            for query_idx in large_query_indices:
                # Count points with confidence >= 0.9 in this query
                high_conf_points = np.where(masked_raw[:, query_idx] >= 0.5)[0]
                num_high_conf = len(high_conf_points)
                
                # Only keep queries that have at least some high-confidence points
                if num_high_conf > 0:
                    high_confidence_query_indices.append(query_idx)
                    high_confidence_point_counts.append(num_high_conf)
            
            # print(f"Step 2 - Confidence filter: Selected {len(high_confidence_query_indices)} queries (with >= {0.5} confidence points)")
            
            # # Print details about high-confidence points
            # if len(high_confidence_point_counts) > 0:
            #     print(f"High-confidence points per query - Min: {min(high_confidence_point_counts)}, "
            #         f"Max: {max(high_confidence_point_counts)}, "
            #         f"Mean: {np.mean(high_confidence_point_counts):.1f}")
            #     print(f"Total high-confidence points across all selected queries: {sum(high_confidence_point_counts)}")
            
            # # Step 3: Use only high-confidence queries for scoring
            # if len(high_confidence_query_indices) > 0:
            #     selected_masked = masked[:, high_confidence_query_indices]
            #     selected_logits = logit[high_confidence_query_indices]
            #     confid = selected_masked.matmul(selected_logits)
                
            #     print(f"Using {len(high_confidence_query_indices)} high-confidence queries for scoring")
            
            # Use only large queries for confidence calculation
            if len(large_query_indices) > 0:
                selected_masked = masked[:, large_query_indices]
                selected_logits = logit[large_query_indices]
                confid = selected_masked.matmul(selected_logits)
                print("Large queries working")
                
            else:
                # Fallback: use all queries if no large ones found
                confid = masked.matmul(logit)
                print("No large queries found. Using all queries as fallback")
            
            #MSP
            probs = torch.functional.F.softmax(confid, dim=1)
            msp = torch.max(probs, dim=1).values
            base_anomaly = msp[inv_map]
            base_anomaly_np = base_anomaly.detach().cpu().numpy()   

            # Debug scores
            base_min, base_max = base_anomaly_np.min(), base_anomaly_np.max()
            base_mean, base_std = base_anomaly_np.mean(), base_anomaly_np.std()
            print(f"msp queries scores - Min: {base_min:.3f}, Max: {base_max:.3f}, Mean: {base_mean:.3f}, Std: {base_std:.3f}")
            
            # No overlap processing - use raw scores directly
            final_anomaly_raw = base_anomaly_np.copy()
            
            # Simple coverage calculation
            coverage_mask = final_anomaly_raw > 0
            coverage_percentage = (np.sum(coverage_mask) / N_raw_points) * 100
            print(f"Coverage: {np.sum(coverage_mask)}/{N_raw_points} points ({coverage_percentage:.1f}%)")
            
            final_anomaly_tensor = torch.from_numpy(final_anomaly_raw).float().to(logit.device)
            final_anomaly_inverted = (final_anomaly_tensor * -1) + 1
            # final_anomaly_inverted = normalize(final_anomaly_inverted)
            # final_anomaly_inverted = (final_anomaly_tensor)
            
            return final_anomaly_inverted
        
        #======================================================================================================

        # Modified into large groups but normal maxlogit process no objectomaly
        # Query_energy
        def get_query_based_anomaly_energy(
                logit, 
                mask, 
                inv_map,
                min_points=1000,  # Only queries with 1000+ points
                threshold=0.1   
            ):
            """
            Use only large queries (1000+ points) for inlier scoring
            """
            masked = mask.float().sigmoid()
            
            # Map voxel masks to raw points
            masked_np = masked.detach().cpu().numpy()
            inv_map_np = inv_map.cpu().numpy()
            
            N_raw_points = len(inv_map_np)
            masked_raw = np.zeros((N_raw_points, masked_np.shape[1]))
            
            for raw_point_idx, voxel_idx in enumerate(inv_map_np):
                masked_raw[raw_point_idx, :] = masked_np[voxel_idx, :]
            
            # Select only queries with 1000+ points
            large_query_indices = []
            for query_idx in range(masked_raw.shape[1]):
                active_points = np.where(masked_raw[:, query_idx] > threshold)[0]
                num_points = len(active_points)
                if num_points >= min_points:  # Only 1000+ points
                    large_query_indices.append(query_idx)
            
            print(f"Selected {len(large_query_indices)} large queries (>=1000 points)")
            
            # Use only large queries for confidence calculation
            if len(large_query_indices) > 0:
                selected_masked = masked[:, large_query_indices]
                selected_logits = logit[large_query_indices]
                confid = selected_masked.matmul(selected_logits)
                print("Large queries working")
            else:
                # Fallback: use all queries if no large ones found
                confid = masked.matmul(logit)
                print("No large queries found. Using all queries as fallback")
            
            #energy
            energy = -torch.logsumexp(confid, dim=1)[inv_map]
            base_anomaly = energy
            base_anomaly_np = base_anomaly.detach().cpu().numpy()

            

            # Debug scores
            base_min, base_max = base_anomaly_np.min(), base_anomaly_np.max()
            base_mean, base_std = base_anomaly_np.mean(), base_anomaly_np.std()
            print(f"energy queries scores - Min: {base_min:.3f}, Max: {base_max:.3f}, Mean: {base_mean:.3f}, Std: {base_std:.3f}")
            
            # No overlap processing - use raw scores directly
            final_anomaly_raw = base_anomaly_np.copy()
            
            # Simple coverage calculation
            coverage_mask = final_anomaly_raw > 0
            coverage_percentage = (np.sum(coverage_mask) / N_raw_points) * 100
            print(f"Coverage: {np.sum(coverage_mask)}/{N_raw_points} points ({coverage_percentage:.1f}%)")
            
            final_anomaly_tensor = torch.from_numpy(final_anomaly_raw).float().to(logit.device)
            # final_anomaly_inverted = (final_anomaly_tensor * -1) + 1
            final_anomaly_inverted_ = (final_anomaly_tensor)
            final_anomaly_inverted = normalize(final_anomaly_inverted_)

            
            return final_anomaly_inverted
        
        #======================================================================================

        # Modified into large groups but normal maxlogit process no objectomaly
        # Query_entropy
        def get_query_based_anomaly_entropy(
                logit, 
                mask, 
                inv_map,
                min_points=1000,  # Only queries with 1000+ points
                threshold=0.1   
            ):
            """
            Use only large queries (1000+ points) for inlier scoring
            """
            masked = mask.float().sigmoid()
            
            # Map voxel masks to raw points
            masked_np = masked.detach().cpu().numpy()
            inv_map_np = inv_map.cpu().numpy()
            
            N_raw_points = len(inv_map_np)
            masked_raw = np.zeros((N_raw_points, masked_np.shape[1]))
            
            for raw_point_idx, voxel_idx in enumerate(inv_map_np):
                masked_raw[raw_point_idx, :] = masked_np[voxel_idx, :]
            
            # Select only queries with 1000+ points
            large_query_indices = []
            for query_idx in range(masked_raw.shape[1]):
                active_points = np.where(masked_raw[:, query_idx] > threshold)[0]
                num_points = len(active_points)
                if num_points >= min_points:  # Only 1000+ points
                    large_query_indices.append(query_idx)
            
            print(f"Selected {len(large_query_indices)} large queries (>=1000 points)")
            
            # Use only large queries for confidence calculation
            if len(large_query_indices) > 0:
                selected_masked = masked[:, large_query_indices]
                selected_logits = logit[large_query_indices]
                confid = selected_masked.matmul(selected_logits)
                print("Large queries working")
            else:
                # Fallback: use all queries if no large ones found
                confid = masked.matmul(logit)
                print("No large queries found. Using all queries as fallback")
            
            #Maxlogit
            probs = torch.functional.F.softmax(confid, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)[inv_map]
            base_anomaly = entropy
            base_anomaly_np = base_anomaly.detach().cpu().numpy()

            # Debug scores
            base_min, base_max = base_anomaly_np.min(), base_anomaly_np.max()
            base_mean, base_std = base_anomaly_np.mean(), base_anomaly_np.std()
            print(f"entropy queries scores - Min: {base_min:.3f}, Max: {base_max:.3f}, Mean: {base_mean:.3f}, Std: {base_std:.3f}")
            
            # No overlap processing - use raw scores directly
            final_anomaly_raw = base_anomaly_np.copy()
            
            # Simple coverage calculation
            coverage_mask = final_anomaly_raw > 0
            coverage_percentage = (np.sum(coverage_mask) / N_raw_points) * 100
            print(f"Coverage: {np.sum(coverage_mask)}/{N_raw_points} points ({coverage_percentage:.1f}%)")
            
            final_anomaly_tensor = torch.from_numpy(final_anomaly_raw).float().to(logit.device)
            # final_anomaly_inverted = (final_anomaly_tensor * -1) + 1
            final_anomaly_inverted = (final_anomaly_tensor)
            
            return final_anomaly_inverted
        
        #=================================================================================================

        # Query_entropy
        def get_query_based_anomaly_ensemble(
                logit, 
                mask, 
                inv_map,
                min_points=1000,  # Only queries with 1000+ points
                threshold=0.1   
            ):
            """
            Use only large queries (1000+ points) for inlier scoring
            """
            masked = mask.float().sigmoid()
            
            # Map voxel masks to raw points
            masked_np = masked.detach().cpu().numpy()
            inv_map_np = inv_map.cpu().numpy()
            
            N_raw_points = len(inv_map_np)
            masked_raw = np.zeros((N_raw_points, masked_np.shape[1]))
            
            for raw_point_idx, voxel_idx in enumerate(inv_map_np):
                masked_raw[raw_point_idx, :] = masked_np[voxel_idx, :]
            
            # Select only queries with 1000+ points
            large_query_indices = []
            for query_idx in range(masked_raw.shape[1]):
                active_points = np.where(masked_raw[:, query_idx] > threshold)[0]
                num_points = len(active_points)
                if num_points >= min_points:  # Only 1000+ points
                    large_query_indices.append(query_idx)
            
            print(f"Selected {len(large_query_indices)} large queries (>=1000 points)")
            
            # Use only large queries for confidence calculation
            if len(large_query_indices) > 0:
                selected_masked = masked[:, large_query_indices]
                selected_logits = logit[large_query_indices]
                confid = selected_masked.matmul(selected_logits)
                print("Large queries working")
            else:
                # Fallback: use all queries if no large ones found
                confid = masked.matmul(logit)
                print("No large queries found. Using all queries as fallback")
            
            ensemble = confid[inv_map]
            print(f"Ensemble shape: {ensemble.shape}")
            # # No overlap processing - use raw scores directly
            # final_anomaly_raw = confid.copy()
            
            # # Simple coverage calculation
            # coverage_mask = final_anomaly_raw > 0
            # coverage_percentage = (np.sum(coverage_mask) / N_raw_points) * 100
            # print(f"Coverage: {np.sum(coverage_mask)}/{N_raw_points} points ({coverage_percentage:.1f}%)")
            
            # final_anomaly_tensor = torch.from_numpy(final_anomaly_raw).float().to(logit.device)
            # # final_anomaly_inverted = (final_anomaly_tensor * -1) + 1
            # final_anomaly_inverted = (final_anomaly_tensor)
            
            return ensemble
        
        #================================================================================================0

        def get_query_based_anomaly_sml(
                logit, 
                mask, 
                inv_map,
                class_mean,
                class_var,
                min_points=1000,  # Only queries with 1000+ points
                threshold=0.1   
            ):
            """
            Use only large queries (1000+ points) for inlier scoring
            """
            masked = mask.float().sigmoid()
            
            # Map voxel masks to raw points
            masked_np = masked.detach().cpu().numpy()
            inv_map_np = inv_map.cpu().numpy()
            
            N_raw_points = len(inv_map_np)
            masked_raw = np.zeros((N_raw_points, masked_np.shape[1]))
            
            for raw_point_idx, voxel_idx in enumerate(inv_map_np):
                masked_raw[raw_point_idx, :] = masked_np[voxel_idx, :]
            
            # Select only queries with 1000+ points
            large_query_indices = []
            for query_idx in range(masked_raw.shape[1]):
                active_points = np.where(masked_raw[:, query_idx] > threshold)[0]
                num_points = len(active_points)
                if num_points >= min_points:  # Only 1000+ points
                    large_query_indices.append(query_idx)
            
            print(f"Selected {len(large_query_indices)} large queries (>=1000 points)")
            
            # Use only large queries for confidence calculation
            if len(large_query_indices) > 0:
                selected_masked = masked[:, large_query_indices]
                selected_logits = logit[large_query_indices]
                confid = selected_masked.matmul(selected_logits)
                print("Large queries working")
            else:
                # Fallback: use all queries if no large ones found
                confid = masked.matmul(logit)
                print("No large queries found. Using all queries as fallback")
            
            #SML
            max_logit, prediction = torch.max(confid, dim=1)    # max logit per point and predicted class
            max_logit = max_logit[inv_map]                      # map back to raw points
            prediction = prediction[inv_map]                    # same for classes


            # sml = max_logit.clone()
            sml = torch.zeros_like(max_logit)

            for c in range(len(class_mean)):
                mask_c = prediction == c
                if mask_c.any():
                    eps = 1e-6
                    # sml[mask_c] = ((max_logit[mask_c] - class_mean[c]) / torch.sqrt(class_var[c]+eps))
                    z = (max_logit[mask_c] - class_mean[c]) / torch.sqrt(class_var[c] + eps)
                    y = (max_logit[mask_c] - class_mean[c])** 2 / 19
        
                    # Clip POSITIVE z-scores (normal) to +1
                    # z = torch.clamp(z, max=1.0)  # Clip positive side!
                    sml[mask_c] = z + y

            base_anomaly = sml
            base_anomaly_np = base_anomaly.detach().cpu().numpy()
        

            # Debug scores
            base_min, base_max = base_anomaly_np.min(), base_anomaly_np.max()
            base_mean, base_std = base_anomaly_np.mean(), base_anomaly_np.std()
            print(f"SML queries scores - Min: {base_min:.3f}, Max: {base_max:.3f}, Mean: {base_mean:.3f}, Std: {base_std:.3f}")

            # No overlap processing - use raw scores directly
            final_anomaly_raw = base_anomaly_np.copy()
            
            # Simple coverage calculation
            coverage_mask = final_anomaly_raw > 0
            coverage_percentage = (np.sum(coverage_mask) / N_raw_points) * 100
            print(f"Coverage: {np.sum(coverage_mask)}/{N_raw_points} points ({coverage_percentage:.1f}%)")
            
            final_anomaly_tensor = torch.from_numpy(final_anomaly_raw).float().to(logit.device)
            final_anomaly_inverted = (final_anomaly_tensor * -1) + 1
            # final_anomaly_inverted = normalize(final_anomaly_inverted)
            
            
            return final_anomaly_inverted
        
        #============================================================================================
        
            
        pred_logits_raw = output["pred_logits"] #Accessing via keys so output is dictionary
        # self.save_pred_logits(pred_logits_raw, batch_idx)
        pred_logits = torch.functional.F.softmax(pred_logits_raw, dim=-1)[..., :-1]
        # self.save_pred_logits_sm(pred_logits, batch_idx)
        pred_masks = output["pred_masks"] 
        # self.save_pred_mask(pred_masks, batch_idx) 
        

        for b_idx in range(len(pred_logits)):
            sequence_name = sequences[b_idx][0]
            frame_id = int(sequences[b_idx][1])
            frame_str = f"{frame_id:06d}"
            inv_map = inverse_maps[b_idx]

            # maxlogit = get_maxlogit(
            #     pred_logits[b_idx], 
            #     pred_masks[b_idx], 
            #     inv_map
            #     )
            
            # maxlogit_pre = get_maxlogit_pre(
            #     self,
            #     pred_logits[b_idx], 
            #     pred_masks[b_idx], 
            #     inv_map,
            #     sequences,
            #     b_idx,
            #     batch_idx
            #     )
            
            # msp = get_msp(
            #     pred_logits[b_idx], 
            #     pred_masks[b_idx], 
            #     inv_map
            #     )

            # rba = get_rba(
            #     pred_logits[b_idx], 
            #     pred_masks[b_idx], 
            #     inv_map
            #     )          
           
            # entropy = get_entropy(
            #     pred_logits[b_idx], 
            #     pred_masks[b_idx], 
            #     inv_map
            #     )
           
            # energy = get_energy(
            #     pred_logits[b_idx], 
            #     pred_masks[b_idx], 
            #     inv_map
            #     )

            # ensemble = get_ensemble(
            #     pred_logits[b_idx], 
            #     pred_masks[b_idx], 
            #     inv_map
            #     ) 

            sml = get_sml(
                pred_logits[b_idx], 
                pred_masks[b_idx], 
                inv_map,
                self.class_mean, 
                self.class_var
                )
            
            # sml_p = get_proper_sml(
            #     pred_logits[b_idx], 
            #     pred_masks[b_idx], 
            #     inv_map,
            #     self.class_mean_no_inv, 
            #     self.class_var_no_inv
            #     )
            
            # sml_calib = get_sml_calib(
            #     pred_logits[b_idx], 
            #     pred_masks[b_idx], 
            #     inv_map,
            #     self.class_mean, 
            #     self.class_var
            #     )

            # query_anomaly_calib = get_query_based_anomaly_maxlogit(
            #     pred_logits[b_idx], 
            #     pred_masks[b_idx], 
            #     inv_map,
            #     min_points=30,  # Changed to 1000 for large queries
            #     threshold=0.1,
            #     # save_intermediate=True,
            #     # sequence_info=(sequences[b_idx][0], sequences[b_idx][1])  # Add this line
            # )

            # query_anomaly_calib_rba = get_query_based_anomaly_rba(
            #     pred_logits[b_idx], 
            #     pred_masks[b_idx], 
            #     inv_map,
            #     min_points=30,  # Changed to 1000 for large queries
            #     threshold=0.1,
            #     # save_intermediate=True,
            #     # sequence_info=(sequences[b_idx][0], sequences[b_idx][1])  # Add this line
            # )

            # query_anomaly_calib_msp = get_query_based_anomaly_msp(
            #     pred_logits[b_idx], 
            #     pred_masks[b_idx], 
            #     inv_map,
            #     min_points=30,  # Changed to 1000 for large queries
            #     threshold=0.1,
            #     # save_intermediate=True,
            #     # sequence_info=(sequences[b_idx][0], sequences[b_idx][1])  # Add this line
            # )

            # query_anomaly_calib_energy = get_query_based_anomaly_energy(
            #     pred_logits[b_idx], 
            #     pred_masks[b_idx], 
            #     inv_map,
            #     min_points=30,  # Changed to 1000 for large queries
            #     threshold=0.1,
            #     # save_intermediate=True,
            #     # sequence_info=(sequences[b_idx][0], sequences[b_idx][1])  # Add this line
            # )

            # query_anomaly_calib_entropy = get_query_based_anomaly_entropy(
            #     pred_logits[b_idx], 
            #     pred_masks[b_idx], 
            #     inv_map,
            #     min_points=5,  # Changed to 1000 for large queries
            #     threshold=0.1,
            #     # save_intermediate=True,
            #     # sequence_info=(sequences[b_idx][0], sequences[b_idx][1])  # Add this line
            # )

            # query_anomaly_calib_ensemble = get_query_based_anomaly_ensemble(
            #     pred_logits[b_idx], 
            #     pred_masks[b_idx], 
            #     inv_map,
            #     min_points=30,  # Changed to 1000 for large queries
            #     threshold=0.1,
            #     # save_intermediate=True,
            #     # sequence_info=(sequences[b_idx][0], sequences[b_idx][1])  # Add this line
            # )

            query_anomaly_calib_sml = get_query_based_anomaly_sml(
                pred_logits[b_idx], 
                pred_masks[b_idx], 
                inv_map,
                self.class_mean, 
                self.class_var,
                min_points=30,  # Changed to 1000 for large queries
                threshold=0.1,
                # save_intermediate=True,
                # sequence_info=(sequences[b_idx][0], sequences[b_idx][1])  # Add this line
            )
        
            # Normalize the result
            # query_anomaly_calib = normalize(query_anomaly)
            
            # Save inlier predictions
            confid = pred_masks[b_idx].float().sigmoid().matmul(pred_logits[b_idx])
            # sem_preds = torch.argmax(confid, dim=1)[inv_map].cpu().numpy()
            sem_preds = torch.argmax(confid, dim=1)[inv_map]

            # =======================================================================
            # 3. Save the scores
            # =======================================================================

            # scores_dict = {
                # "maxlogit": maxlogit,
                # "msp": msp,
                # "rba": rba, 
                # "entropy": entropy,
                # "energy": energy,
                # "sml": sml,
                # "sml_calib": sml_calib,
                # "inlier": sem_preds
                # }

            # for method_name, score in scores_dict.items():
            #     save_anomaly_score(
            #         score, 
            #         method_name, 
            #         sequences, 
            #         b_idx, 
            #         frame_str, 
            #         self.config.general.save_dir
            #         )

            # # Save inlier predictions
            # confid = pred_masks[b_idx].float().sigmoid().matmul(pred_logits[b_idx])
            # sem_preds = torch.argmax(confid, dim=1)[inv_map].cpu().numpy()
            # # sem_preds_path = base_path_maxlogit / f"sem_preds_{sequences[b_idx][1]}.txt"
            # sem_preds_path = base_path_msp / f"classes_{frame_str}.txt"  # CHANGED NAME
            # np.savetxt(sem_preds_path, sem_preds, fmt='%d')
            # #========================================
            
            # # For Maxlogit
            # save_path_maxlogit = Path(self.config.general.save_dir) / "prediction_maxlogit"
            # base_path_maxlogit = save_path_maxlogit / f"{sequences[b_idx][0]}"
                
            # # Save original MaxLogit
            # original_save_path_maxlogit = base_path_maxlogit / f"{sequences[b_idx][1]}.txt"          
            # if not original_save_path_maxlogit.parent.exists():
            #     original_save_path_maxlogit.parent.mkdir(exist_ok=True, parents=True)
            # np.savetxt(original_save_path_maxlogit, maxlogit.detach().cpu().numpy())
            # np.save(base_path_maxlogit / f"ml_{frame_str}.npy", maxlogit.detach().cpu().numpy().astype(np.float32))
            # # #=========================================================================
            # # Save the result query based max logit
            # save_path_query = Path(self.config.general.save_dir) / "prediction_query_based_maxlogit"
            # base_path_query = save_path_query / f"{sequences[b_idx][0]}"
            # base_path_query.mkdir(parents=True, exist_ok=True)
            
            # np.savetxt(base_path_query / f"{sequences[b_idx][1]}.txt", 
            #         query_anomaly_calib.detach().cpu().numpy())
            # np.save(base_path_query / f"query_anomaly_{frame_str}.npy", 
            #         query_anomaly_calib.detach().cpu().numpy().astype(np.float32))
            # #============================================================================
        
            # For Maxlogit
            # save_path_maxlogit_ = Path(self.config.general.save_dir) / "prediction_maxlogit_pre"
            # base_path_maxlogit_ = save_path_maxlogit_ / f"{sequences[b_idx][0]}"
                
            # # Save original MaxLogit
            # original_save_path_maxlogit_ = base_path_maxlogit_ / f"{sequences[b_idx][1]}.txt"          
            # if not original_save_path_maxlogit_.parent.exists():
            #     original_save_path_maxlogit_.parent.mkdir(exist_ok=True, parents=True)
            # np.savetxt(original_save_path_maxlogit_, maxlogit_pre.detach().cpu().numpy())
            # np.save(base_path_maxlogit_ / f"ml_{frame_str}.npy", maxlogit_pre.detach().cpu().numpy().astype(np.float32))
            # #=========================================================================  
            
            # # For MSP
            # save_path_msp = Path(self.config.general.save_dir) / "prediction_msp"
            # base_path_msp = save_path_msp / f"{sequences[b_idx][0]}"
                
            # # Save original msp
            # original_save_path_msp = base_path_msp / f"{sequences[b_idx][1]}.txt"          
            # if not original_save_path_msp.parent.exists():
            #     original_save_path_msp.parent.mkdir(exist_ok=True, parents=True)
            # np.savetxt(original_save_path_msp, msp.detach().cpu().numpy())
            # np.save(base_path_msp / f"msp_{frame_str}.npy", msp.detach().cpu().numpy().astype(np.float32))
            # #=========================================================================

            # # Save the result query based msp
            # save_path_query_msp = Path(self.config.general.save_dir) / "prediction_query_based_msp"
            # base_path_query_msp = save_path_query_msp / f"{sequences[b_idx][0]}"
            # base_path_query_msp.mkdir(parents=True, exist_ok=True)
            
            # np.savetxt(base_path_query_msp / f"{sequences[b_idx][1]}.txt", 
            #         query_anomaly_calib_msp.detach().cpu().numpy())
            # np.save(base_path_query_msp / f"query_anomaly_msp{frame_str}.npy", 
            #         query_anomaly_calib_msp.detach().cpu().numpy().astype(np.float32))
            # # ============================================================================

            # # For Entropy
            # save_path_entropy = Path(self.config.general.save_dir) / "prediction_entropy"
            # base_path_entropy = save_path_entropy / f"{sequences[b_idx][0]}"
                        
            # # Save original Entropy
            # original_save_path_entropy = base_path_entropy / f"{sequences[b_idx][1]}.txt"          
            # if not original_save_path_entropy.parent.exists():
            #     original_save_path_entropy.parent.mkdir(exist_ok=True, parents=True)
            # np.savetxt(original_save_path_entropy, entropy.detach().cpu().numpy())
            # np.save(base_path_entropy / f"entropy_{frame_str}.npy", entropy.detach().cpu().numpy().astype(np.float32))
            # #=========================================================================

            # # Save the result query based entropy
            # save_path_query_entropy = Path(self.config.general.save_dir) / "prediction_query_based_entropy"
            # base_path_query_entropy = save_path_query_entropy / f"{sequences[b_idx][0]}"
            # base_path_query_entropy.mkdir(parents=True, exist_ok=True)
            
            # np.savetxt(base_path_query_entropy / f"{sequences[b_idx][1]}.txt", 
            #         query_anomaly_calib_entropy.detach().cpu().numpy())
            # np.save(base_path_query_entropy / f"query_anomaly_entropy{frame_str}.npy", 
            #         query_anomaly_calib_entropy.detach().cpu().numpy().astype(np.float32))
            # #============================================================================

            # # For Energy
            # save_path_energy = Path(self.config.general.save_dir) / "prediction_energy"
            # base_path_energy = save_path_energy / f"{sequences[b_idx][0]}"
                        
            # # Save original Energy
            # original_save_path_energy = base_path_energy / f"{sequences[b_idx][1]}.txt"          
            # if not original_save_path_energy.parent.exists():
            #     original_save_path_energy.parent.mkdir(exist_ok=True, parents=True)
            # np.savetxt(original_save_path_energy, energy.detach().cpu().numpy())
            # np.save(base_path_energy / f"energy_{frame_str}.npy", energy.detach().cpu().numpy().astype(np.float32))
            # # =========================================================================

            # # Save the result query based energy
            # save_path_query_energy = Path(self.config.general.save_dir) / "prediction_query_based_energy"
            # base_path_query_energy = save_path_query_energy / f"{sequences[b_idx][0]}"
            # base_path_query_energy.mkdir(parents=True, exist_ok=True)
            
            # np.savetxt(base_path_query_energy / f"{sequences[b_idx][1]}.txt", 
            #         query_anomaly_calib_energy.detach().cpu().numpy())
            # np.save(base_path_query_energy / f"query_anomaly_energy{frame_str}.npy", 
            #         query_anomaly_calib_energy.detach().cpu().numpy().astype(np.float32))
            # #============================================================================
            
            # # For query based ensemble
            # save_path_query_ensemble = Path(self.config.general.save_dir) / "prediction_query_based_ensemble"
            # base_path_query_ensemble = save_path_query_ensemble / f"{sequences[b_idx][0]}"
                
            # # Save original ensemble
            # original_save_path_query_ensemble = base_path_query_ensemble / f"{sequences[b_idx][1]}.txt"          
            # if not original_save_path_query_ensemble.parent.exists():
            #     original_save_path_query_ensemble.parent.mkdir(exist_ok=True, parents=True)
            # np.savetxt(original_save_path_query_ensemble, query_anomaly_calib_ensemble.detach().cpu().numpy())
            # np.save(base_path_query_ensemble / f"query_anomaly_ensemble{frame_str}.npy", ensemble.detach().cpu().numpy().astype(np.float32))
            # # #=========================================================================

            # # For Ensemble
            # save_path_ensemble = Path(self.config.general.save_dir) / "prediction_ensemble"
            # base_path_ensemble = save_path_ensemble / f"{sequences[b_idx][0]}"
                
            # # Save original ensemble
            # original_save_path_ensemble = base_path_ensemble / f"{sequences[b_idx][1]}.txt"          
            # if not original_save_path_ensemble.parent.exists():
            #     original_save_path_ensemble.parent.mkdir(exist_ok=True, parents=True)
            # np.savetxt(original_save_path_ensemble, ensemble.detach().cpu().numpy())
            # np.save(base_path_ensemble / f"ensemble_{frame_str}.npy", ensemble.detach().cpu().numpy().astype(np.float32))
            # # #=========================================================================

            # For SML
            save_path_sml = Path(self.config.general.save_dir) / "prediction_sml"
            base_path_sml = save_path_sml / f"{sequences[b_idx][0]}"
                        
            # Save original SML
            original_save_path_sml = base_path_sml / f"{sequences[b_idx][1]}.txt"          
            if not original_save_path_sml.parent.exists():
                original_save_path_sml.parent.mkdir(exist_ok=True, parents=True)
            np.savetxt(original_save_path_sml, sml.detach().cpu().numpy())
            np.save(base_path_sml / f"sml_{frame_str}.npy", sml.detach().cpu().numpy().astype(np.float32))
            # #=========================================================================

             # Save the result query based sml
            save_path_query_sml = Path(self.config.general.save_dir) / "prediction_query_based_sml"
            base_path_query_sml = save_path_query_sml / f"{sequences[b_idx][0]}"
            base_path_query_sml.mkdir(parents=True, exist_ok=True)
            
            np.savetxt(base_path_query_sml / f"{sequences[b_idx][1]}.txt", 
                    query_anomaly_calib_sml.detach().cpu().numpy())
            np.save(base_path_query_sml / f"query_anomaly_sml{frame_str}.npy", 
                    query_anomaly_calib_sml.detach().cpu().numpy().astype(np.float32))

            #==================================================================================

            # # For SML Proper
            # save_path_sml_p = Path(self.config.general.save_dir) / "prediction_sml_p"
            # base_path_sml_p = save_path_sml_p / f"{sequences[b_idx][0]}"
                        
            # # Save original SML
            # original_save_path_sml_p = base_path_sml_p / f"{sequences[b_idx][1]}.txt"          
            # if not original_save_path_sml_p.parent.exists():
            #     original_save_path_sml_p.parent.mkdir(exist_ok=True, parents=True)
            # np.savetxt(original_save_path_sml_p, sml_p.detach().cpu().numpy())
            # np.save(base_path_sml_p / f"sml_p_{frame_str}.npy", sml_p.detach().cpu().numpy().astype(np.float32))
            # #=========================================================================

            # # For SML Calib
            # save_path_sml_calib = Path(self.config.general.save_dir) / "prediction_sml_calib"
            # base_path_sml_calib = save_path_sml_calib / f"{sequences[b_idx][0]}"
                        
            # # Save original SML
            # original_save_path_sml_calib = base_path_sml_calib / f"{sequences[b_idx][1]}.txt"          
            # if not original_save_path_sml_calib.parent.exists():
            #     original_save_path_sml_calib.parent.mkdir(exist_ok=True, parents=True)
            # np.savetxt(original_save_path_sml_calib, sml_calib.detach().cpu().numpy())
            # np.save(base_path_sml_calib / f"sml_calib_{frame_str}.npy", sml_calib.detach().cpu().numpy().astype(np.float32))
            # # =========================================================================

            # # For RBA
            # save_path_rba = Path(self.config.general.save_dir) / "prediction_rba"
            # base_path_rba = save_path_rba / f"{sequences[b_idx][0]}"
                        
            # # Save original RBA
            # original_save_path_rba = base_path_rba / f"{sequences[b_idx][1]}.txt"          
            # if not original_save_path_rba.parent.exists():
            #     original_save_path_rba.parent.mkdir(exist_ok=True, parents=True)
            # np.savetxt(original_save_path_rba, rba.detach().cpu().numpy())
            # np.save(base_path_rba / f"rba_{frame_str}.npy", rba.detach().cpu().numpy().astype(np.float32))
            # #==========================================================================

            # # Save the result query based rba
            # save_path_query_rba = Path(self.config.general.save_dir) / "prediction_query_based_rba"
            # base_path_query_rba = save_path_query_rba / f"{sequences[b_idx][0]}"
            # base_path_query_rba.mkdir(parents=True, exist_ok=True)
            
            # np.savetxt(base_path_query_rba / f"{sequences[b_idx][1]}.txt", 
            #         query_anomaly_calib_rba.detach().cpu().numpy())
            # np.save(base_path_query_rba / f"query_anomaly_rba{frame_str}.npy", 
            #         query_anomaly_calib_rba.detach().cpu().numpy().astype(np.float32))
            # # ============================================================================

            # # For inlier
            # save_path_sem_preds = Path(self.config.general.save_dir) / "prediction_sem_preds"
            # base_path_sem_preds = save_path_sem_preds / f"{sequences[b_idx][0]}"
                
            # # Save original sem_preds
            # original_save_path_sem_preds = base_path_sem_preds / f"{sequences[b_idx][1]}.txt"          
            # if not original_save_path_sem_preds.parent.exists():
            #     original_save_path_sem_preds.parent.mkdir(exist_ok=True, parents=True)
            # np.savetxt(original_save_path_sem_preds, sem_preds.detach().cpu().numpy())
            # np.save(base_path_sem_preds / f"sem_preds_{frame_str}.npy", sem_preds.detach().cpu().numpy().astype(np.float32))
            #=========================================================================

            

        return {}

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
                # remove this loss if not specified in `weight_dict`
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
                    sem_preds, range(1, self.config.data.min_stuff_cls_id), invert=True
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

    def training_epoch_end(self, outputs):
        train_loss = sum([out["loss"].cpu().item() for out in outputs]) / len(outputs)
        results = {"train_loss_mean": train_loss}
        self.log_dict(results)

    def validation_epoch_end(self, outputs):
        self.last_seq = None
        class_names = self.config.data.class_names
        pq, sq, rq, all_pq, all_sq, all_rq = self.class_evaluator.getPQ()
        self.class_evaluator.reset()
        results = {}
        results["val_mean_pq"] = pq
        results["val_mean_sq"] = sq
        results["val_mean_rq"] = rq
        # print(class_names)
        # print(all_pq)
        # print(all_sq)
        # print(all_rq)
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
            [item for item in [v for k, v in dd.items() if "loss_ce" in k]]
        )
        dd["val_mean_loss_mask"] = statistics.mean(
            [item for item in [v for k, v in dd.items() if "loss_mask" in k]]
        )
        dd["val_mean_loss_dice"] = statistics.mean(
            [item for item in [v for k, v in dd.items() if "loss_dice" in k]]
        )
        dd["val_mean_loss_box"] = statistics.mean(
            [item for item in [v for k, v in dd.items() if "loss_box" in k]]
        )
        self.log_dict(dd)

    def test_epoch_end(self, outputs):
        return {}

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

# Query-Optimized 3D Anomaly Segmentation
### Comprehensive Uncertainty Evaluation for Autonomous Driving

**Master's Thesis** · Technische Hochschule Ingolstadt · 2025  
**Author:** Sourabh Lolge · `00135092`  
**Supervisor:** M.Sc. Yan Zhiran  
**Examiners:** Prof. Dr. Gordon Elger · Prof. Dr. rer. nat. Armin Arnold  
**Research Group:** Sensor Technology and Data Fusion for Environmental Perception, Institute of Innovative Mobility

---

> **Built upon the CVPR 2025 paper:**  
> *Spotting the Unexpected (STU): A 3D LiDAR Dataset for Anomaly Segmentation in Autonomous Driving*  
> Nekrasov et al. — [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/html/Nekrasov_Spotting_the_Unexpected_STU_A_3D_LiDAR_Dataset_for_Anomaly_Segmentation_CVPR_2025_paper.html) [[Original Repo]](https://github.com/kumuji/stu_dataset)

---

## What This Work Contributes

The STU benchmark evaluated only 5 of 8 major uncertainty methods and processed all 100 object queries indiscriminately despite clear computational redundancy. This thesis addresses both gaps:

**1. Novel Post-Processing Query Selection Framework**  
Filters Mask4Former-3D object queries by spatial area threshold before anomaly scoring — no architectural changes, no retraining. Achieves **62% query reduction** (100 → ~38 queries) while *improving* detection performance, with up to **+16.70% AUROC** at the critical 10–20m detection range.

**2. First Comprehensive Uncertainty Method Evaluation in 3D**  
Extends STU's evaluation to all major output-based methods: MSP, SML, Energy, and Entropy — previously validated only in 2D contexts. Key finding: **MSP achieves 92.94% AUROC vs Deep Ensemble's 90.93%**, with 3× lower computational cost (single forward pass vs. three models).

---

## Key Results

### Comprehensive Method Comparison (STU Validation Set)

| Method | Queries | AUROC ↑ | FPR@95 ↓ | AP ↑ | RecallQ ↑ | UQ ↑ |
|--------|---------|---------|----------|------|-----------|------|
| MC Dropout ⋆ | Full | 65.76 | 79.82 | 0.17 | 3.54 | 2.63 |
| Void Classifier ⋆ | Full | 89.77 | 79.50 | 2.62 | 17.35 | 14.10 |
| RbA ⋆ | Full | 73.00 | 100.00 | 1.64 | 21.84 | 17.16 |
| MaxLogit ⋆ | Full | 87.27 | 68.76 | 2.02 | 26.64 | 21.12 |
| Deep Ensemble ⋆ | Full | 90.93 | 37.34 | 6.94 | 17.70 | 14.15 |
| **MSP (Ours)** | Full | 91.24 | 41.01 | 6.42 | 7.51 | 6.75 |
| **SML (Ours)** | Full | 94.10 | 19.63 | 2.04 | 21.69 | 17.51 |
| **MSP + QS (Ours)** | Selected | **92.94** | 33.55 | 6.21 | 7.51 | 6.75 |
| **SML + QS (Ours)** | Selected | **95.18** | **17.40** | 2.06 | 22.31 | 17.91 |
| **MaxLogit + QS (Ours)** | Selected | 92.52 | 36.33 | 6.11 | **36.93** | **29.30** |

⋆ Results officially reported in the STU benchmark paper. QS = Query Selection (Area > 30).

### Practical Deployment Recommendations

| Use Case | Recommended Method | Why |
|----------|-------------------|-----|
| General balanced detection | **MSP + Query Selection** | Best AUROC/AP balance, single forward pass |
| Safety-critical (minimize false alarms) | **SML + Query Selection** | Lowest FPR@95 (17.40%) |
| Object-level localization | **MaxLogit + Query Selection** | Highest RecallQ (36.93%) and UQ (29.30%) |

---

## Repository Structure

```
stu-anomaly-postprocessing/
│
├── query_selection/                ← [CONTRIBUTION 1] Query filtering framework
│   ├── filter.py                   ← select_queries(): core area-based filtering
│   ├── query_mask_analysis.py
│   └── query_mask_reordering.py
│
├── uncertainty_methods/            ← [CONTRIBUTION 2] Extended uncertainty scoring
│   ├── scoring.py                  ← score_maxlogit/msp/rba/entropy/energy/sml()
│   ├── analyse_anomaly_score.py
│   └── create_ensemble.py
│
├── evaluation/                     ← Metrics and evaluation scripts
│   ├── metrics_calculation_fusion.py
│   ├── IoU_calculate.py
│   ├── bar_graph_plotting.py
│   └── Box_plot_plotting.py
│
├── visualization/                  ← Visualization scripts
│   └── ...
│
├── utils/                          ← Shared utilities and STU evaluation
│   ├── combined_metrics_computation.py
│   └── common.py
│
├── Mask4Former3D/                  ← STU baseline (attributed)
│   ├── trainer/
│   │   ├── pq_trainer.py           ← YOUR modified trainer (imports scoring + filter)
│   │   └── pq_trainer_orignal.py   ← STU original trainer (unchanged)
│   ├── models/
│   ├── conf/
│   └── BASELINE_README.md          ← Original STU documentation
│
├── compute_object_level_ood.py     ← STU evaluation script
├── compute_point_level_ood.py      ← STU evaluation script
├── README.md
├── ATTRIBUTION.md
└── LICENSE
```

---

## Getting Started

### Prerequisites

```bash
# Python 3.9, CUDA 11.x
pip install torch==2.4.0
pip install MinkowskiEngine==0.5.4
pip install pytorch-lightning hydra-core scikit-learn
```

### Dataset Setup

| Dataset | Role | Sequences | Scans |
|---------|------|-----------|-------|
| SemanticKITTI | ID Training | 00–07, 09–10 | 19,130 |
| Panoptic-CUDA | ID Training | 30, 31, 36, 40, 41 | 4,200 |
| STU-Inlier | ID Training | 206 | 450 |
| STU Anomaly | OOD Evaluation | 19 scenes | 350–650/scene |

Download the STU dataset from the [original repository](https://github.com/kumuji/stu_dataset).

---

## Running Inference

Run from inside `Mask4Former3D/` with `PYTHONPATH` pointing to repo root:

```bash
cd Mask4Former3D

PYTHONPATH=/path/to/stu-anomaly-postprocessing python main_panoptic.py \
    model=mask4former3d \
    data/datasets=semantic_kitti_206 \
    general.ckpt_path=/path/to/checkpoints/Ensemble_model_1.ckpt \
    general.mode=test \
    general.save_dir=/path/to/saved/results
```

**Toggle methods** in `Mask4Former3D/trainer/pq_trainer.py` — uncomment the method(s) you want. Both full queries and query-selected versions run simultaneously and save to separate folders:

```
saved/results/
├── prediction_maxlogit/       ← full queries
├── prediction_maxlogit_qs/    ← query selected (Area > 30)
├── prediction_msp/
├── prediction_msp_qs/
├── prediction_rba/
├── prediction_rba_qs/
├── prediction_entropy/
├── prediction_entropy_qs/
├── prediction_energy/
└── prediction_energy_qs/
```

---

## Running Evaluation

Run from repo root:

```bash
python3 -m utils.combined_metrics_computation \
    --prediction_type query_anomaly \
    --data_path /path/to/stu_dataset/validation \
    --prediction_path /path/to/saved/results/prediction_msp_qs \
    --output_path /path/to/saved/results/metrics_msp_qs.json
```

---

## How Query Selection Works

```
LiDAR Point Cloud
        │
        ▼
┌─────────────────────┐
│   Mask4Former-3D    │  ← Pre-trained backbone
└─────────────────────┘
        │
        ▼ 100 object queries + predicted masks
┌────────────────────────────────────┐
│   Query Selection (filter.py)      │  ← Filter: mask area > 30 points
│   ~62% reduction, no retraining    │  ← 100 → ~38 queries
└────────────────────────────────────┘
        │
        ├── Full queries (100)     ──► scored → prediction_<method>/
        └── Selected queries (~38) ──► scored → prediction_<method>_qs/
```

---

## Implemented Uncertainty Methods

| Method | Reference | Best Metric |
|--------|-----------|-------------|
| MaxLogit | Hendrycks et al., 2019 | RecallQ: 36.93% (with QS) |
| MSP | Hendrycks & Gimpel, 2016 | AUROC: 92.94% (with QS) |
| RbA | Nayal et al., ICCV 2023 | UQ: 29.87% (with QS) |
| Entropy | Chan et al., ICCV 2021 | — |
| Energy | Tian et al., ECCV 2022 | — |
| SML | Jung et al., ICCV 2021 | FPR@95: 17.40% (with QS) |

All methods implemented in `uncertainty_methods/scoring.py`.

---

## Citation

```bibtex
@mastersthesis{lolge2025queryoptimized,
  title     = {Query-Optimized 3D Anomaly Segmentation:
               Comprehensive Uncertainty Evaluation for Autonomous Driving},
  author    = {Lolge, Sourabh},
  school    = {Technische Hochschule Ingolstadt},
  year      = {2025},
  type      = {Master's Thesis}
}

@inproceedings{nekrasov2025stu,
  title     = {Spotting the Unexpected (STU): A 3D LiDAR Dataset
               for Anomaly Segmentation in Autonomous Driving},
  author    = {Nekrasov, Alexey and Burdorf, Malcolm and Worrall, Stewart
               and Leibe, Bastian and Berrio Perez, Julie Stephany},
  booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2025}
}
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.  
The baseline code from the STU repository is also MIT licensed — see [ATTRIBUTION.md](ATTRIBUTION.md).
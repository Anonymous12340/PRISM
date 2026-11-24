# PRISM: Orchestrating Contract Normalization and Multi-Expert Candidate Fusion for Task-Aware Tabular Data Quality

[![Venue](https://img.shields.io/badge/Venue-VLDB%20Submission-blue)](#)
[![Artifact](https://img.shields.io/badge/Artifact-Reproducibility%20Package-brightgreen)](#)
[![Python](https://img.shields.io/badge/Python-3.12-informational)](#)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](#)

Official anonymous artifact repository for the VLDB submission:
> **PRISM: Orchestrating Contract Normalization and Multi-Expert Candidate Fusion for Task-Aware Tabular Data Quality**

---

## Overview

Preparing dirty, multi-source relational tables for downstream machine learning remains challenging in practice:
- **Task-agnostic cleaning** often overwrites informative missingness and valid extreme values without verifying downstream predictive risk.
- **Unconstrained LLM code agents** consume large token budgets, suffer from execution failures, and fail to invert documented multi-source unit scaling.
- **Single-family predictors** struggle to generalize across contrasting tabular regimes (e.g., compact smooth signals vs. skewed high-variance or categorical-heavy tables).

**PRISM** is a four-stage framework under strict **group-disjoint train–test isolation** that decouples deterministic schema recovery from bounded multi-expert synthesis and fold-stability validation:

1. **Deterministic Contract Normalization ($\psi_C$)**: Inverts documented multi-source unit scaling (e.g., $100\times/1000\times$) and string casing/whitespace drift prior to modeling while preserving native missingness.
2. **Bounded Local-LLM Constrained Planning**: Uses a local LLM planner (`Qwen3-4B`, `temperature=0`, JSON-constrained decoding over a closed operator library of $\le 160$ verified operators) to synthesize task-conditioned repair and feature pipelines for gradient-boosted trees (`XGBoost`).
3. **Task-Switched Frozen Neural Prior Experts & Parameter-Free Fusion**: Constructs a three-family candidate pool uniting (i) the tree-only planned prefix, (ii) a task-switched frozen tabular in-context neural prior (**TabICL** for classification, **TabPFN** for regression; 8 estimators), and (iii) a parameter-free equal-weight ($0.5/0.5$) blend.
4. **Support-Augmented Cross-Validation Selector**: Gates candidate adoption by inner 3-fold cross-validation stability ($m_{\min}=2$, $1.0\,\widehat{\text{SE}}$, 1,000 bootstrap resamples at $\alpha=0.05$) and freezes holdout predictions before sealed test labels are read once.

### Key Empirical Highlights
- **Shared-Table Multi-Task (E1)**: Wins **18/20** endpoints across 10 shared relational tables ($-12.29\%$ mean regression RMSE over the tree-only prefix; $9/9$ classification task-seed wins).
- **Clean & Export-Corrupted Benchmarks (E2)**: Ranks #1 on **3/4** clean regression tables and **4/4** composite export-corrupted classification tables (up to **$+21.16$ pp ROC-AUC** over the strongest baseline, with zero fallbacks).
- **Controlled Error Injection (E3)**: Achieves the lowest RMSE on **15/15** error-type/rate cells on `diamonds` and the only positive net win margin across all five error categories.
- **Cross-Domain Authentic Dirty Transfer (E5)**: Only system simultaneously improving over unmodified `Raw` on both `telco_churn` ($0.8481$ AUC) and `retinopathy` ($0.8186$ AUC).
- **Scale Invariance & Efficiency (E6–E7)**: Sustains full accuracy down to 4-bit quantized `Qwen3-1.7B/4B` ($2.4$–$3.9$ GiB VRAM) while cutting LLM token usage by **$74.1\%$–$84.6\%$** vs. `LLM-FE`.

---

## Repository Structure

```text
PRISM/
├── README.md                          # Overview, artifact mapping, and reproduction guide
├── system/                            # Final frozen PRISM runtime and expert system
│   ├── CONFIRMED_VERSION.json         # Frozen SHA-256 manifest (formal_version_fixed: true)
│   ├── requirements.txt               # Verified Python 3.12 / CUDA dependencies
│   ├── core/
│   │   └── taskdq/                    # Frozen core planner, operator library, CV selector, and SFT/GRPO modules
│   └── expert_system/                 # Complete Expert System (contract recovery, TabICL, TabPFN, CV blend)
├── results/                           # Strictly filtered final paper results (E1–E7)
│   ├── PROVENANCE.json                # Version provenance (historical baselines vs. frozen PRISM reruns)
│   ├── E1/                            # Shared-table multi-task evaluation (Figure 3)
│   ├── E2/                            # Clean regression & composite corruption (Table 3)
│   ├── E3/                            # Controlled ground-truth error injection (Table 4)
│   ├── E4/                            # Uncontracted core component ablation (Figure 4)
│   ├── E5/                            # Cross-domain authentic dirty transfer (Figure 5)
│   ├── E6/                            # Planner scale, NF4 quantization & post-training audit (Table 5)
│   └── E7/                            # Quality–cost & active GPU scaling trade-off (Figure 6)
└── reproduction/                      # Frozen experiment configurations and runner scripts
    ├── configs/                       # JSON configs for E1–E6
    └── scripts/                       # Runner scripts for E1–E7
```

---

## Artifact-to-Paper Mapping & Version Provenance

This repository includes **only** the datasets, baselines, and metrics reported in the final manuscript. Because selected baseline evaluations were completed prior to the final PRISM system freeze while PRISM was re-evaluated with the frozen system (`system/CONFIRMED_VERSION.json`), both sources are explicitly separated and documented in `results/PROVENANCE.json`.

| Paper Ref. | Suite | Retained Datasets | Retained Methods | Result Files (`results/`) | Runner Script (`reproduction/scripts/`) |
| :--- | :---: | :--- | :--- | :--- | :--- |
| **Figure 3** (Sec. 5.2) | **E1** | 10 shared tables (20 endpoints) | `Raw`, `Standard` (historical); `Full Prefix`, `PRISM` (frozen) | `E1/historical_baselines_mean_sd.csv`, `E1/frozen_prism_per_seed.csv` | `run_e1_frozen.py` |
| **Table 3** (Sec. 5.3) | **E2** | Clean Reg: `cpu_act`, `Ailerons`, `houses`, `diamonds`; Composite Cls: `german_credit`, `default_credit`, `bank_marketing`, `qsar-biodeg` | 9 baselines (+ `Old Full` on clean reg) & `PRISM` ($N \in \{5, 10, 20\}$) | `E2/clean_regression_summary.csv`, `E2/composite_corruption_summary.csv` (+ per-seed CSVs) | `run_e2_clean_regression.py`, `run_e2_composite_corruption.py` |
| **Table 4** (Sec. 5.4) | **E3** | `airline_satisfaction`, `diamonds` | `Raw`, `Standard`, `CAAFE`, `LLM-FE`, `Classical Clean`, `PRISM` ($N=10$, 5 errors × 3 rates) | `E3/downstream_mean_sd.csv`, `E3/downstream_per_seed.csv` | `run_e3_error_injection.py` |
| **Figure 4** (Sec. 5.5) | **E4** | `german_credit`, `default_credit`, `cpu_act` | `PRISM (Full)` vs. `w/o Fusion`, `w/o LLM`, `w/o Native Int.`, `Rule-Only`, `w/o Expert` | `E4/ablation_mean_sd.csv`, `E4/ablation_per_seed.csv` | `run_e4_core_ablation.py` |
| **Figure 5** (Sec. 5.6) | **E5** | `telco_churn`, `retinopathy` | 7 historical baselines (`Raw`, `Standard`, `OpenFE`, `AutoFeat`, `CAAFE`, `LLM-FE`, `No-LM`) + frozen `PRISM` | `E5/historical_baselines_per_seed.csv`, `E5/frozen_prism_per_seed.csv` | `run_e5_transfer.py` |
| **Table 5** (Sec. 5.7) | **E6** | `cpu_act` | 11 planner configs (`Qwen3-0.6B/1.7B/4B/8B` FP16/NF4, `Qwen3-4B Trained` FP16/NF4, `Phi-3.5-mini` FP16) | `E6/scale_quantization_summary.csv`, `E6/condition_costs.csv`, `E6/proposal_quality.csv` | `run_e6_scale_quantization.py` |
| **Figure 6** (Sec. 5.7) | **E7** | `qsar-biodeg`, `cpu_act`, `houses` | `PRISM`, `Standard`, `LLM-FE`, `Clean + OpenFE` ($N=5$) | `E7/quality_cost_summary.csv`, `E7/quality_per_seed.csv`, `E7/cost_per_seed.csv` | `run_e7_quality_cost.py` |

---

## System Implementation & Expert Modules (`system/`)

The frozen implementation is verified by SHA-256 digests in `system/CONFIRMED_VERSION.json`:

- **Core Runtime (`system/core/taskdq/`)**:
  - Constrained JSON planner & dependency decoder: `planner.py`, `json_decode.py`, `full_dependency_decode_20261114.py`
  - Closed operator library & execution engine: `operations.py`, `e2_library.py`, `e2_operations.py`, `e2_classical.py`, `pipeline.py`
  - Group-disjoint evaluation & CV risk gating: `e2_unified.py`, `e2_data.py`, `e2_risk.py`, `e2_selection.py`, `experiment.py`
  - Optional SFT / GRPO post-training modules: `learning.py`, `e2_aligned_sft.py`, `e2_trained.py`
- **Expert System (`system/expert_system/`)**:
  - **Contract Normalization**: `export_shift_schema_stability_20261118.py`, `export_shift_mechanism_20261118.py`, `qualify_export_schema_recovery_20261118.py`
  - **Classification Neural Prior (TabICL)**: `TabICL_expert_comparison_20261118.py`, `tabicl_local_worker_20261118.py`
  - **Regression Neural Prior (TabPFN)**: `TabPFN_full_expert_20261118.py`, `e2_frozen_neural_worker_20261118.py`, `tabpfn_power_finite_repair_20261118.py`, `neural_feature_worker_20261118.py`
  - **CV-Gated Candidate Pool & 0.5/0.5 Blend**: `full_novelty_return_runtime_20261117.py`, `export_shift_full_pilot_20261118.py`

---

## Hardware, Software & Installation

### Experimental Environment
- **OS**: Linux (tested on Python 3.12)
- **GPU**: NVIDIA RTX 4090 (24 GB VRAM; 4-bit NF4 `Qwen3-4B` requires $\approx 3.9$ GiB VRAM)
- **Primary Downstream Model**: `XGBoost` (`n_estimators=200`, `max_depth=6`, `learning_rate=0.1`, `tree_method=hist`, `random_state=0`)
- **Default Local Planner**: `Qwen3-4B` (`temperature=0`, `max_new_tokens=1536`, 3 attempts)

### Setup
```bash
git clone https://github.com/Anonymous12340/PRISM.git
cd PRISM
python3 -m venv .venv
source .venv/bin/activate
pip install -r system/requirements.txt
export PYTHONPATH="$(pwd)/system/core:$(pwd)/system/expert_system:$PYTHONPATH"
```

---

## External Datasets & Pretrained Weights

To respect third-party licenses and repository size limits, raw third-party datasets and pretrained model checkpoints are **not** bundled:
- **Public Tabular Datasets**: All benchmark tables are publicly available from [OpenML](https://www.openml.org/) (e.g., `airline_satisfaction` #46920, `diamonds` #42225, `cpu_act`, `Ailerons`, `houses`, `qsar-biodeg`, `german_credit`, `default_credit`, `bank_marketing`, `telco_churn`, `retinopathy`) and the [UCI Machine Learning Repository](https://archive.ics.uci.edu/) (`credit`, `medical`, `credit_card`, `bike`, `student`, `parkinsons`, `appliances`, `air_quality`, `residential`, `sml2010`).
- **Pretrained Checkpoints**: Local planner weights (`Qwen/Qwen3-0.6B`, `1.7B`, `4B`, `8B`; `microsoft/Phi-3.5-mini-instruct`) and frozen tabular neural priors (`TabICL`, `TabPFN v2`) are downloaded from their official Hugging Face repositories.

---

## Reproduction Instructions

1. **Configure Local Paths**: Replace `<PROJECT_ROOT>` placeholders in `reproduction/configs/*.json` with your local dataset and Hugging Face cache directories.
2. **Run Experiment Suites**:
   ```bash
   # E1: Shared-Table Multi-Task (Figure 3)
   python reproduction/scripts/run_e1_frozen.py

   # E2: Clean Regression & Composite Export Corruption (Table 3)
   python reproduction/scripts/run_e2_clean_regression.py
   python reproduction/scripts/run_e2_composite_corruption.py

   # E3: Controlled Error Injection (Table 4)
   python reproduction/scripts/run_e3_error_injection.py

   # E4: Uncontracted Core Ablation (Figure 4)
   python reproduction/scripts/run_e4_core_ablation.py

   # E5: Cross-Domain Authentic Dirty Transfer (Figure 5)
   python reproduction/scripts/run_e5_transfer.py

   # E6: Model Scale & 4-bit NF4 Quantization Audit (Table 5)
   python reproduction/scripts/run_e6_scale_quantization.py

   # E7: Quality–Cost Trade-Off (Figure 6)
   python reproduction/scripts/run_e7_quality_cost.py
   ```
3. **Verify Published Tables Directly from `results/`**:
   All aggregated mean $\pm$ SD tables and per-seed records in `results/E1`–`results/E7` correspond directly to Figures 3–6 and Tables 3–5 of the manuscript and can be audited independently without GPU execution.

---

## Anonymity & License

This repository is anonymized for double-blind VLDB review (`Anonymous12340`). Code and experimental result tables are released under the MIT License; third-party datasets and pretrained checkpoints remain subject to their respective upstream licenses.

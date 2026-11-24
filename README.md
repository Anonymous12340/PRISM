# PRISM: Planned Repair and In-Context Synthesis for Tabular Machine Learning

Official artifact repository for the VLDB submission **PRISM** (*Planned Repair and In-Context Synthesis for Tabular Machine Learning*).

PRISM unifies deterministic public export-contract normalization, constrained local-LLM data-quality planning, bounded boosted-tree feature synthesis, and frozen in-context neural tabular experts (TabICL / TabPFN) under cross-validation-gated candidate fusion.

---

## 1. Repository Structure

```text
PRISM-VLDB/
├── README.md                      # Project overview, provenance, and reproduction guide
├── VERIFICATION_REPORT.md         # Full Phase-1 paper-vs-server data audit report
├── system/
│   ├── CONFIRMED_VERSION.json     # Frozen SHA-256 manifest of the final PRISM system
│   ├── requirements.txt           # Python & CUDA environment dependencies
│   ├── core/
│   │   └── taskdq/                # Final frozen core TaskDQ / PRISM runtime package
│   └── expert_system/             # Complete Expert System (TabICL, TabPFN, export contract, CV blend)
├── results/
│   ├── PROVENANCE.json            # Machine-readable mapping of all E1–E7 files & version provenance
│   ├── E1/                        # Shared-table multi-task (10 tables, 20 endpoints)
│   ├── E2/                        # Clean regression (4 tables) & Composite corruption (4 tables), N=5/10/20
│   ├── E3/                        # Controlled error injection (airline_satisfaction, diamonds; 5 errors × 3 rates)
│   ├── E4/                        # Core component ablation (german_credit, default_credit, cpu_act)
│   ├── E5/                        # Cross-domain authentic dirty transfer (telco_churn, retinopathy)
│   ├── E6/                        # Planner scale, 4-bit NF4 quantization & post-training audit (cpu_act)
│   └── E7/                        # Quality-cost trade-off (qsar-biodeg, cpu_act, houses; N=5)
└── reproduction/
    ├── configs/                   # Frozen JSON experiment configurations (E1–E7)
    └── scripts/                   # Experiment runner scripts (E1–E7)
```

---

## 2. Final Frozen System & Expert System (`system/`)

The final frozen system corresponds to manifest `system/CONFIRMED_VERSION.json` (`formal_version_fixed: true`, `method: "mandatory public export normalization, Full N3, fixed task expert and half-blend"`).

### Core Package (`system/core/taskdq/`)
- **Constrained Planner & JSON Decoding**: `planner.py`, `json_decode.py`, `full_dependency_decode_20261114.py`
- **Operator Library & Execution**: `operations.py`, `e2_library.py`, `e2_operations.py`, `e2_classical.py`, `pipeline.py`
- **Unified Evaluation & Leakage-Free Splits**: `e2_unified.py`, `e2_data.py`, `e2_risk.py`, `e2_selection.py`, `experiment.py`
- **Optional Offline Post-Training (SFT/GRPO)**: `learning.py`, `e2_aligned_sft.py`, `e2_trained.py`

### Expert System (`system/expert_system/`)
- **Stage (a) Public Export Normalization**: `export_shift_schema_stability_20261118.py`, `export_shift_mechanism_20261118.py`, `qualify_export_schema_recovery_20261118.py` — deterministically normalizes multi-source unit scaling and string formatting drift using public export contracts prior to planning.
- **Stage (b) Frozen In-Context Neural Experts**:
  - Classification Prior (**TabICL**): `TabICL_expert_comparison_20261118.py`, `tabicl_local_worker_20261118.py`
  - Regression Prior (**TabPFN**): `TabPFN_full_expert_20261118.py`, `e2_frozen_neural_worker_20261118.py`, `tabpfn_power_finite_repair_20261118.py`, `neural_feature_worker_20261118.py`
- **Stage (c) CV-Gated Candidate Pool & Fixed 0.5/0.5 Fusion**: `full_novelty_return_runtime_20261117.py`, `export_shift_full_pilot_20261118.py` — selects among the tree-only prefix, standalone frozen neural prior, and equal-weight 0.5/0.5 blend strictly via inner 3-fold cross-validation (`m_min=2`, `1.0 SE`).

---

## 3. Experimental Results & Version Provenance (`results/`)

This repository contains **only** the datasets, methods, and budgets selected in the final paper, strictly filtered from full server runs. Full provenance is recorded in `results/PROVENANCE.json`.

Because existing baselines were evaluated prior to the final PRISM system freeze while PRISM was re-evaluated with the final frozen system, each experiment preserves explicit version separation:

| Exp. | Paper Ref. | Datasets Retained | Methods Retained | Provenance / Version |
| :--- | :--- | :--- | :--- | :--- |
| **E1** | Fig. 3 (Sec. 5.2) | 10 shared tables (20 endpoints) | `Raw`, `Standard` (historical); `Full`, `PRISM` (frozen) | Baselines: `e1_final_system_20261108`; PRISM: `e1_current_frozen_full_20261118` |
| **E2** | Table 3 (Sec. 5.3) | Clean Reg: `cpu_act`, `Ailerons`, `houses`, `diamonds`; Composite Cls: `german_credit`, `default_credit`, `bank_marketing`, `qsar-biodeg` | 9 baselines (+ `old_Full` in Clean Reg) & `PRISM` (`N=5/10/20`) | Clean Reg baselines: `e2_unified_independent_20261107`; PRISM: `e2_frozen_system_20261118` & `e2_composite_four_class_20261118` |
| **E3** | Table 4 (Sec. 5.4) | `airline_satisfaction`, `diamonds` | `Raw`, `Standard`, `CAAFE`, `LLM-FE`, `Classical Clean`, `PRISM` (`N=10`, 5 errors × 3 rates) | Unified frozen run: `e3_four_dataset_20261119` |
| **E4** | Fig. 4 (Sec. 5.5) | `german_credit`, `default_credit`, `cpu_act` | `Full (PRISM)` vs. `w/o Fusion`, `w/o LLM`, `w/o Native Int.`, `Rule-Only`, `w/o Expert` | Unified frozen run: `e4_core_ablation_20261119` |
| **E5** | Fig. 5 (Sec. 5.6) | `telco_churn`, `retinopathy` | 7 historical baselines (`Raw`, `Standard`, `OpenFE`, `AutoFeat`, `CAAFE`, `LLM-FE`, `No-LM`) + frozen `PRISM` | Baselines: `e5_telco_retinopathy_three_seed_20261111`; PRISM: `e5_current_full_rerun_20261119` |
| **E6** | Table 5 (Sec. 5.7) | `cpu_act` | 11 planner configs (`Qwen3-0.6B/1.7B/4B/8B` FP16/NF4, `Qwen3-4B Trained` FP16/NF4, `Phi-3.5-mini` FP16) | Unified frozen run: `e6_core_pressure_20261120` |
| **E7** | Fig. 6 (Sec. 5.7) | `qsar-biodeg`, `cpu_act`, `houses` | `PRISM (Ours)`, `Standard`, `LLM-FE`, `Clean + OpenFE` (`N=5`) | Unified frozen run: `e7_system_quality_cost_20261120_retry3` |

---

## 4. Datasets & Public Availability

Raw third-party datasets are **not** redistributed in this repository to respect third-party OpenML / UCI licenses. All benchmark tables are publicly available from [OpenML](https://www.openml.org/) and the [UCI Machine Learning Repository](https://archive.ics.uci.edu/):
- OpenML IDs (e.g., `airline_satisfaction` #46920, `diamonds` #42225, `cpu_act`, `Ailerons`, `houses`, `qsar-biodeg`, `german_credit`, `default_credit`, `bank_marketing`, `telco_churn`, `retinopathy`) and UCI shared-table benchmarks (`credit`, `medical`, `credit_card`, `bike`, `student`, `parkinsons`, `appliances`, `air_quality`, `residential`, `sml2010`) can be downloaded directly via OpenML/UCI APIs.
- Local pretrained model weights (`Qwen3-0.6B/1.7B/4B/8B`, `Phi-3.5-mini`, `TabICL`, `TabPFN v2`) are obtained from their official Hugging Face repositories and are not bundled here.

---

## 5. Environment & Reproduction

### Hardware & Software
- **OS**: Linux (tested on Python 3.12)
- **GPU**: NVIDIA RTX 4090 (24 GB VRAM; 4-bit NF4 Qwen3-4B requires ~3.9 GiB VRAM)
- **Downstream Predictor**: `XGBoost` (`n_estimators=200`, `max_depth=6`, `learning_rate=0.1`, `tree_method=hist`, `random_state=0`)

### Installation
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r system/requirements.txt
export PYTHONPATH="$(pwd)/system/core:$(pwd)/system/expert_system:$PYTHONPATH"
```

### Running Reproduction Scripts
All frozen experiment configurations and runners are located under `reproduction/configs/` and `reproduction/scripts/`:
- **E1**: `python reproduction/scripts/run_e1_frozen.py`
- **E2 (Clean Regression)**: `python reproduction/scripts/run_e2_clean_regression.py`
- **E2 (Composite Corruption)**: `python reproduction/scripts/run_e2_composite_corruption.py`
- **E3 (Error Injection)**: `python reproduction/scripts/run_e3_error_injection.py`
- **E4 (Core Ablation)**: `python reproduction/scripts/run_e4_core_ablation.py`
- **E5 (Cross-Domain Transfer)**: `python reproduction/scripts/run_e5_transfer.py`
- **E6 (Scale & Quantization)**: `python reproduction/scripts/run_e6_scale_quantization.py`
- **E7 (Quality-Cost)**: `python reproduction/scripts/run_e7_quality_cost.py`

> **Note**: The reproduction scripts preserve the exact frozen code used during the VLDB evaluation without altering core algorithms or experimental logic. Before running on a new machine, update dataset and local Hugging Face cache paths in the corresponding JSON config under `reproduction/configs/`.

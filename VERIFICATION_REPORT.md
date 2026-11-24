# VLDB 论文（PRISM）最终实验数据真实性核验报告

**核验日期**：2026-11-24  
**核验范围**：用户提供的 13 份最终实验数据汇总表（E1–E7）、论文最终稿 `paper_vldb_20261124_v8` 中的全部实验图表（Table 2–5、Figure 3–6 及 Section 5 正文引用数值），以及服务器 `<PROJECT_ROOT>/results/` 与 `deliverables/` 下的真实原始运行输出。

---

## 一、总体核验结论

1. **真实性与可追溯性**：论文中 E1–E7 全部核心实验结果均**真实来自服务器上的实验运行记录与 CSV 汇总文件**，未发现任何凭空编造、估算或篡改数据的情况。
2. **发现 3 处具体不一致/排版误差**（均已精确定位原因与原始文件路径，**未擅自修改论文图表或原始数据**）：
   - **问题 1（E1 汇总图片存在前缀匹配合并 Bug，论文 Figure 3 已修正）**
   - **问题 2（E1 论文正文 Section 5.2 中 `Residential Sale Price` 相对降幅数值笔误）**
   - **问题 3（E2 论文 Table 3 中 `Old Full` 在 `Ailerons` 上末两位数字差异）**
3. **其余 E2（75/76 格）、E3（180/180 格）、E4（全部 14 项）、E5（全部 16 格）、E6（全部 11 行 × 4 列）、E7（全部 3 数据集 × 4 方法 × 质量与成本指标）均与服务器原始运行结果完全一致。**

---

## 二、逐表逐图核验明细

### 1. E1：共享表多任务评估（用户图 1 / 论文 Figure 3 & Sec. 5.2）

- **服务器原始来源**：
  - 历史基线（Raw / Standard）：`results/e1_final_system_20261108/summary/mean_sd.csv`
  - 最终冻结系统（Full / PRISM 专家系统）：`results/e1_current_frozen_full_20261118/deliverables/SCORE_REPORT.csv`
- **核验结果**：19/20 个任务在用户图 1、论文 Figure 3 与服务器原始 CSV 之间完全一致；发现 **2 处问题**：
  1. **[问题 1 — 用户提供的 E1 汇总图第 1 行计算 Bug，论文 Figure 3 正确]**：
     - 用户 E1 汇总图中，`信贷 未完全偿还`（`credit_default`）显示：`当前Full = 0.682040 ± 0.047239`，`当前专家系统 = 0.704195 ± 0.036606`。
     - 经查服务器原始 `SCORE_REPORT.csv`，`credit_default` 三个种子的真实均值为 **`original_Full = 0.640252 ± 0.005725`**，**`Full_neural_experts = 0.671311 ± 0.007071`**。
     - **问题定位**：生成中文 E1 汇总图时使用了前缀匹配，将 `credit_default`（3 种子）与 `credit_card_default`（3 种子）共 6 条记录合并求均值（经复算，6 条合并恰好为 `0.682040 ± 0.047239` 与 `0.704195 ± 0.036606`）。
     - **论文状态**：论文 Figure 3(a) 与 Section 5.2 正文（`$0.6713 \pm 0.0071$` vs. `$0.6403$`）**已经修正该 Bug，使用了正确的 3 种子真实结果**。
  2. **[问题 2 — 论文 Section 5.2 正文笔误]**：
     - `paper_vldb_20261124_v8/sections/05_experiments.tex` 第 33 行写有：`and Residential Sale Price ($-25.25\%$)`。
     - 经核对服务器原始数据，`residential_sale_price` 的 Raw RMSE 为 `304.645640`，PRISM RMSE 为 `271.247188`，相对 Raw 的 RMSE 降幅应为 **`-10.96%`**（论文 Figure 3(b) 中 `Res.: Sale` 柱状图正确绘制在约 `+11.0%` 处），而非 `-25.25%`。

---

### 2. E2：系统级对比（用户图 2–4 / 论文 Table 3 & Sec. 5.3）

- **服务器原始来源**：
  - 干净回归（`cpu_act`, `Ailerons`, `houses`, `diamonds`；N=5/10/20）：`results/e2_frozen_system_20261118/deliverables/summary.csv`（历史基线源自 `results/e2_unified_independent_20261107`，本系统 `ours` 为冻结版重跑）
  - 组合导出污染分类（`german_credit`, `default_credit`, `bank_marketing`, `qsar-biodeg`；N=5/10/20）：`results/e2_composite_four_class_20261118/deliverables/summary.csv`
  - 正文提及的 4 个干净分类表（`kr-vs-kp 0.9995`, `tic-tac-toe 0.9997`, `vote 0.9902`, `qsar-biodeg 0.9330`）：均在 `results/e2_frozen_system_20261118/deliverables/summary.csv` 中核验通过。
- **核验结果**：用户图 2–4（N=5/10/20）与服务器 CSV **100% 一致**；论文 Table 3 共 76 格中 **75/76 格完全一致**，发现 **1 处细微差异**：
  3. **[问题 3 — 论文 Table 3 中 `Old Full` 在 `Ailerons` 数值尾差]**：
     - 论文 Table 3（`tables/tab2_e2_main.tex` 第 22 行）中，`Old Full` 在 `Ailerons` 上记为 `1.58096 ± 0.04462`（$\times 10^{-4}$）。
     - 服务器原始 `results/e2_frozen_system_20261118/deliverables/summary.csv` 中对应记录为 `0.0001581066341736953 ± 4.462395512809193e-06`，四舍五入应为 **`1.58107 ± 0.04462`**（相差 `0.00011 × 10^-4`）。另注：`Old Full` 未出现在用户 E2 中文汇总图中，但在论文 Table 3 及服务器 CSV 中存在。

---

### 3. E3：受控错误注入（用户图 5–9 / 论文 Table 4 & Sec. 5.4）

- **服务器原始来源**：`results/e3_four_dataset_20261119/summary/downstream_mean_sd.csv` 与 `downstream_all_seeds.csv`
- **核验结果**：**完全核验通过（180/180 格一致）**。
  - 用户图 5–9（单位不一致、类别拼写、重复、离群、缺失 5 张表）与论文 Table 4 全部 180 个单元格、以及回退标记（$\dagger k$）均与服务器 CSV 逐格一致。
  - 仅 5 处存在 4 位小数修约边界差异（如 `0.000950` 记作 `0.0009`、`0.983550` 记作 `0.9836`、`0.002650` 记作 `0.0027`），属正常舍入。

---

### 4. E4：关键组件消融（用户图 10 / 论文 Figure 4 & Sec. 5.5）

- **服务器原始来源**：`results/e4_core_ablation_20261119/summary/mean_sd.csv` 与 `all_seeds.csv`
- **核验结果**：**完全核验通过（14/14 项一致）**。
  - 用户图 10 全部 14 项指标均值与样本标准差均与服务器 CSV 精确吻合至小数点后 6 位；论文 Figure 4 标注的 3 位小数（`0.490`, `0.464`, `0.547`, `0.531`, `0.535`, `0.544`, `0.275`, `0.258`, `0.249`）全部准确无误。

---

### 5. E5：跨领域真实脏表迁移（用户图 11 / 论文 Figure 5 & Sec. 5.6）

- **服务器原始来源**：
  - 7 个历史基线（`Raw`, `Standard`, `OpenFE`, `AutoFeat`, `CAAFE`, `LLM-FE`, `无 LM 搜索`）：`deliverables/e5_telco_retinopathy_three_seed_20261111/all15_per_seed.csv`
  - 最终冻结系统（`FULL`）：`results/e5_current_full_rerun_20261119/deliverables.csv`
- **核验结果**：**完全核验通过（16/16 格一致）**。
  - 历史基线与冻结系统重跑结果均精确吻合至小数点后 5 位（例如 `telco_churn` PRISM `0.84808 ± 0.01009`，`retinopathy` PRISM `0.81861 ± 0.00000`）。

---

### 6. E6：模型规模与量化敏感性（用户图 12 / 论文 Table 5 & Sec. 5.7）

- **服务器原始来源**：
  - 下游分数：`results/e6_core_pressure_20261120/test_summary.csv`
  - 显存与耗时：`results/e6_core_pressure_20261120/condition_costs.csv`
  - 提案有效率：`results/e6_core_pressure_20261120/proposal_quality.csv`
- **核验结果**：**完全核验通过（11/11 条件 × 4 列指标全部精确一致）**。
  - 包括 `Qwen3-0.6B/1.7B/4B/8B`（FP16/NF4）、`Qwen3-4B Trained`（FP16/NF4）及 `Phi-3.5-mini FP16`（`0/120` 有效触发确定性回退）的 RMSE、SD、有效提案数、峰值显存（GiB）与每单元耗时（s）全部核验无误。

---

### 7. E7：质量—成本对比（用户图 13 / 论文 Figure 6 & Sec. 5.7）

- **服务器原始来源**：`results/e7_system_quality_cost_20261120_retry3/report/quality_cost_summary.csv`、`quality_by_seed.csv`、`cost_by_seed.csv`
- **核验结果**：**完全核验通过**。
  - `qsar-biodeg`、`cpu_act`、`houses` 上 4 种方法（`Ours`, `Standard`, `LLM-FE`, `Clean + OpenFE`）的质量指标、物理 LM 调用数（3 vs. 5）、总 Token 数（`18,572` / `12,140` / `7,752` vs. `120,889` / `66,031` / `29,926`）、LM 生成秒数与神经先验耗时（`4.95s` / `57.32s` / `122.18s`，总活跃 GPU 时间 `94.9s` / `118.0s` / `154.5s`）全部与服务器 CSV 精确一致。

---

## 三、无法验证真实性的数据项

- **无**。论文所有主表、主图以及正文中引用的关键补充数据均已追溯到服务器真实文件并完成复算。

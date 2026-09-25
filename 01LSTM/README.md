# 模型 01 · LSTM 时序趋势建模 —— 目录索引

> 本目录是**数学模型清单**的第 01 号条目：模型卡 + 可复用代码 + 交付数据 + 文档 + AI 协作提示。
> 目标：**在不同背景、不同数据集下都能直接复用**。

## 目录结构

```
01LSTM/
├─ 01_LSTM_模型清单.md          ★ 模型卡（编号/名称/解决的问题/优缺点/代码提示/已犯错误/可复用文件/其他）
├─ README.md                    本索引
├─ 代码/
│  ├─ lstm_template.py          ★ 通用模板（配置驱动，换数据集只改 JSON）
│  ├─ metrics_utils.py          ★ 设计工具（输入向量审计 / 损失权重重对齐 / WMAE 与 N_eff / 死项检测）
│  ├─ config_mcm2025C.json      本项目配置样例（已在 MCM 2025 C 上验证）
│  ├─ requirements.txt          依赖清单
│  ├─ lstm_312.py               本项目专用实现（含调参网格 / p̂ 消融 / 派生验证）
│  ├─ export_delivery.py        交付数据生成脚本（含 SHA256 清单）
│  ├─ verify_patience_from_single_run.py  省算力工具（一次训练派生整条 patience 扫描）
│  ├─ probe_epoch_budget.py     训练预算敏感性探针
│  └─ 模板自检输出/             模板在 MCM 数据上的验证结果（指标 / 趋势值 / 曲线）
├─ 数据/                        ⭐ 交付给下游 XGBoost-Bootstrap 的数据（含 manifest 校验和）
│  ├─ lstm_trend_history.csv    1312 行：82 实体 × 16 目标期次的趋势值（5 种子均值±std）
│  ├─ lstm_trend_2028.csv       82 行：2028 年趋势值
│  ├─ base_panel_country_year.csv  1558 行：82 × 19 面板 + IsHost/Prep/Alpha_c
│  ├─ host_years_by_country.csv 82 行：各国主办年份
│  ├─ lstm_run_config.json      交付模型配置与数据口径
│  ├─ lstm_metrics.csv          5 种子评估指标
│  └─ manifest.json             文件清单 + 行数 + SHA256
├─ 文档/
│  ├─ LSTM模型总结.pdf                  6 页总结（适合论文附录 / 汇报）
│  ├─ LSTM_3.1.2_模型解析_复现第1步.pdf  公式逐条落地与待定项
│  ├─ LSTM复现全过程记录_README.md       调参网格、p̂ 消融、种子噪声分析、全部坑
│  ├─ 交付说明.md                        数据字典 + 下游 merge 示例代码
│  └─ 提示_XGBoost_Bootstrap阶段.md       下一阶段任务书（02 号模型）
└─ AI协作/
   └─ AI提示词模板.md           四类可复制提示词（迁移 / 排错 / 调参 / 交付）
```

## 30 秒上手

```powershell
cd 代码
python -m pip install -r requirements.txt          # 安装依赖（主要是 CPU 版 torch）
python lstm_template.py --config config_mcm2025C.json --dry-run   # 数据自检（秒级）
python lstm_template.py --config config_mcm2025C.json            # 完整训练 + 出交付表
```

迁移到新数据集：复制 `config_mcm2025C.json`，只改 `data` 段的 4 个必填字段
（`panel_csv` / `entity_col` / `year_col` / `target_cols`）即可；没有事件表就删掉 `host_csv`，
模型会自动降级为单通道。

## 已验证事实（可放心引用）

- 通用模板与专用实现在 MCM 数据上**指标逐位一致**：`val_WMAE_Gold = 3.562180`、
  `val_MAE_Gold = 1.776071`、`val_MSE_Total = 65.441387`，2028 澳大利亚趋势
  Gold 20.314 / Total 56.350（见 `代码/模板自检输出/` 与 `数据/`）。
- 调参网格 15 配置 × 5 种子：配置间差异**不显著**（配对 t 检验 |t| < 2.5）；
  单次运行噪声远大于超参数效应 → 不要用超参数解释性能差异。
- 一次 `patience=10` 的训练可派生 patience=6…10 的全部结果（最大差 2.4e-7）。
- WMAE 权重的有效样本量：$N_{eff}=37.8/164$（28% 样本权重为 0）→ 指标实际建立在约 38 个样本上，
  这是种子噪声大的根源（详见模型卡第七节）。

## 设计方法论（模型卡第七节）

模型卡新增 **第七节「如何设计输入向量 / 损失函数 / WMAE」**，含：

| 主题 | 关键结论 |
|---|---|
| 输入向量 | 三类构件（自回归 / 固有属性 / 阶段状态）+ 六条准则；首要红线是**因果可得性**（同届对齐会让 WMAE 虚高到 0.4–2.7） |
| 损失函数 | 三层结构（主任务 + 辅助任务 + 参数正则）；**标准化尺度算损失、原始尺度报指标**；$\lambda$ 首选"量级对齐"；死项检测（常数写进损失 → 梯度为 0） |
| WMAE | 权重四来源、"错谁更疼"先行；$w=0$ 三方案 + 必报 $N_{eff}$；禁用 MAPE；L1/L2 配合看 |
| 协同 | 四步一致性流程：口径 → 尺度 → 权重 → 显著性 |

落地工具：`python metrics_utils.py --demo`（自动输出审计表、权重对照、$\lambda$ 建议、死项检测、一致性检查表）。

## 相关模型编号

| 编号 | 模型 | 与本模型的关系 |
|---|---|---|
| **01** | **LSTM 时序趋势建模** | 本目录 |
| 02 | XGBoost-Bootstrap 区间预测 | **消费本模型输出的趋势值** |
| 03 | PCA 主成分（3.1.1） | 与趋势值并列的输入特征 |
| 04 | Spearman + SHAP 归因 | 解释小项与奖牌的关系 |
| 05 | DID 因果（伟大教练效应） | 独立的因果支线 |
| 06 | Bootstrap + 符号秩检验（零的突破） | 消费 02 的 Bootstrap 样本矩阵 |

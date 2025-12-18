# 项目说明

本仓库包含两个主要方向的实验：

- 学术论文引用预测：基于 XGBoost 多分类模型与 SHAP 解释，以及 GMM 序列预测，对不同学科的论文引用表现进行建模与可解释分析。
- 纽约地铁客流预测：结合 Prophet 的趋势/季节性建模与 LSTM 残差修正，提供交互式可视化与多站点对比。

## 目录与核心脚本
- [yyy_conda/主程序.py](yyy_conda/主程序.py)：XGBoost 多分类（4 类标签）+ 标准化 + 5 折网格搜索；输出性能指标和 SHAP 可解释图。
- [yyy_conda/gmm.py](yyy_conda/gmm.py)：针对各学科（物理、化学、计算机等）的 5+5 年引用序列预测，基于 GMM，自动 9:1 划分训练/测试并生成评价指标与预测结果。
- [yyy_conda/XGBoost+LR.py](yyy_conda/XGBoost+LR.py)：优化版 GMM 配置与预测流程，可批量处理多学科文件并写出预测表。
- [yyy_TSA/正确数据预处理+Pytorch.py](yyy_TSA/正确数据预处理+Pytorch.py)：地铁闸机数据清洗、按天聚合、Prophet 预测、LSTM 残差建模与融合，以及交互式可视化输出。
- 结果目录：`output/`（XGBoost+SHAP 图表与指标）、`gmm/`（GMM 预测与评估）、`yyy_TSA/output_plots/` 和多个 HTML/CSV 交互文件；其余 `output/`、`confusion_matrices/`、`XGBLR_res/` 等为历史/示例结果保存处。
- 根目录 [main.py](main.py) 与 [test.py](test.py) 仅为编辑器示例，可忽略。

## 环境依赖
建议 Python 3.9+。核心依赖：pandas、numpy、scikit-learn、xgboost、shap、matplotlib、seaborn、plotly、torch、prophet、holidays、openpyxl。可直接安装：

```bash
pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn plotly torch prophet holidays openpyxl
```

## 数据准备
- 引用预测（XGBoost + GMM）
	- Excel 至少包含 28 列以计算 TNoC（脚本会汇总第 14-18 与 24-28 列），并包含特征列 TL、NoA、HI、TNoC、NoR、JIF、ECGR(year_1)、PL、PACNCI 以及目标列“标签”。
	- 在 [yyy_conda/主程序.py](yyy_conda/主程序.py) 底部修改 `file_path` 指向待分析的 Excel。
	- GMM 相关脚本默认从 `/data/<subject>.xlsx` 读取并写入 `/gmm/`，如路径不同请在文件末尾修改。

- 地铁客流预测（Prophet + LSTM）
	- 将 turnstile 数据（2014-2018）放在 `C:/Users/fangxiang/Desktop/yyy_TSA/TS/`，或修改 [yyy_TSA/正确数据预处理+Pytorch.py](yyy_TSA/正确数据预处理+Pytorch.py) 中的 `desktop_path`。
	- 首次运行会生成 `daily_results.csv`（按天聚合）和 `prophet_forecast_results.csv`，后续可直接复用。

## 快速开始
1) XGBoost + SHAP 引用分类
- 调整 [yyy_conda/主程序.py](yyy_conda/主程序.py) 的 `file_path` 后运行：
```bash
python yyy_conda/主程序.py
```
- 输出：`output/model_performance.csv`、混淆矩阵及 SHAP 图（蜂群图、特征重要性、瀑布图、力图、依赖图）。

2) GMM 引用序列预测
- 将各学科 Excel 放在脚本指定路径（默认为 `/data/`），执行：
```bash
python yyy_conda/gmm.py
```
- 输出：`gmm/<subject>.xlsx` 预测结果与 `gmm/evaluation/an_am_metrics.xlsx` 指标。

3) 纽约地铁客流预测（Prophet + LSTM）
- 准备原始数据或复用已有 `daily_results.csv`，运行：
```bash
python yyy_TSA/正确数据预处理+Pytorch.py
```
- 输出：`output_plots/` 下的混合模型图、`hybrid_forecast_results.csv`、以及交互式 HTML（多站点对比与预测）。

## 结果与可视化
- 模型性能：`output/model_performance.csv`（XGBoost）、`gmm/evaluation/an_am_metrics.xlsx`（GMM）。
- 可解释性：`output/shap_*.png` 系列图展示特征影响。
- 预测结果：`gmm/*.xlsx`（学科引用预测）、`output_plots/hybrid_forecast_results.csv`（客流混合预测）。
- 交互页面：`yyy_TSA/interactive_*.html`、`output_plots/*.html` 用于浏览器查看。

## 提示
- 随机种子固定为 42，便于复现；如需不同拆分可修改脚本中的随机数设定。
- 若 GPU 可用，PyTorch/LSTM 会自动使用；无 GPU 也可在 CPU 上运行。
- 部分路径使用绝对地址或以 `/` 开头的占位，请根据本机环境调整后再运行。


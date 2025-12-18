#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据挖掘和机器学习分析脚本
使用XGBoost和SHAP分析化学引用数据
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, \
    recall_score
import xgboost as xgb
import shap
import warnings
import os
import traceback
import sys
import io
import base64
import matplotlib.font_manager as fm

# 忽略警告信息
warnings.filterwarnings('ignore')

# 设置matplotlib全局参数
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS', 'DejaVu Sans',
                                   'sans-serif']  # 中文字体设置
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['xtick.direction'] = 'out'  # 刻度线朝外
plt.rcParams['ytick.direction'] = 'out'  # 刻度线朝外
plt.rcParams['mathtext.default'] = 'regular'  # 数学文本使用常规字体
plt.rcParams['savefig.dpi'] = 150  # 保存图片的DPI
plt.rcParams['figure.dpi'] = 150  # 显示图片的DPI

# 设置随机种子以保证结果可重现
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# 目标变量列名
TARGET_COL = '标签'  # 新的目标列名

# 要使用的特征列（使用英文缩写）
FEATURE_COLS = ['TL', 'NoA', 'HI', 'TNoC', 'NoR', 'JIF', 'ECGR', 'PL', 'PACNCI']

# 创建输出目录
if not os.path.exists('output'):
    os.makedirs('output')


def load_data(file_path):
    """
    加载Excel数据文件并重命名列名，计算TNoC为第14-18列与第24-28列的和

    参数:
        file_path: Excel文件路径

    返回:
        加载并重命名后的DataFrame
    """
    print(f"加载数据: {file_path}")
    try:
        data = pd.read_excel(file_path)
        print(f"数据加载成功: {data.shape[0]}行, {data.shape[1]}列")

        # 验证是否足以包含第28列
        if data.shape[1] < 28:
            raise ValueError(f"数据文件只有{data.shape[1]}列，需至少28列以计算TNoC")

        # 计算TNoC为第14-18列（索引13-17）和第24-28列（索引23-27）的和
        # 先将相关列转换为数值，处理非数值数据
        citation_cols = list(range(13, 18)) + list(range(23, 28))
        for col in citation_cols:
            data.iloc[:, col] = pd.to_numeric(data.iloc[:, col], errors='coerce').fillna(0)
        data['TNoC'] = data.iloc[:, 13:18].sum(axis=1) + data.iloc[:, 23:28].sum(axis=1)

        # 列名映射字典（中文→英文缩写，排除TNoC）
        column_mapping = {
            '标题长度': 'TL',
            '作者数量': 'NoA',
            '作者h指数': 'HI',
            '参考文献数量': 'NoR',
            '五年影响因子': 'JIF',
            'year_1': 'ECGR',
            '篇幅': 'PL',
            '十年CNCI': 'PACNCI'
        }

        # 重命名列（只映射存在的列）
        data = data.rename(columns={k: v for k, v in column_mapping.items() if k in data.columns})

        # 打印新的列名以验证
        print("\n处理后数据集中的列名:")
        print(data.columns.tolist())

        return data
    except Exception as e:
        print(f"数据加载失败: {str(e)}")
        raise


def explore_data(df):
    """
    探索性数据分析

    参数:
        df: 数据DataFrame
    """
    print("\n=== 数据探索 ===")

    # 查看基本信息
    print("\n基本信息:")
    print(df.info())

    # 查看统计摘要
    print("\n统计摘要:")
    print(df.describe())

    # 检查缺失值
    missing_values = df.isnull().sum()
    print("\n缺失值:")
    print(missing_values[missing_values > 0])

    # 检查特征列的存在性
    missing_features = [col for col in FEATURE_COLS if col not in df.columns]
    if missing_features:
        print(f"\n警告: 以下必需的特征列缺失: {missing_features}")

    # 目标变量分布
    if TARGET_COL in df.columns:
        print("\n目标变量分布:")
        print(df[TARGET_COL].value_counts())

        # 绘制目标变量分布
        plt.figure(figsize=(8, 6))
        df[TARGET_COL].value_counts().sort_index().plot(kind='bar')
        plt.title('Target Distribution Plot')
        plt.xlabel('Label')
        plt.ylabel('Number')
        plt.xticks(rotation=0)
        plt.savefig('output/target_distribution.png')
        plt.close()
    else:
        print(f"\n警告: 在数据集中未找到目标列 '{TARGET_COL}'")


def preprocess_data(df):
    """
    数据预处理: 处理缺失值、标准化等

    参数:
        df: 原始数据DataFrame

    返回:
        处理后的DataFrame、特征X、目标y
    """
    print("\n=== 数据预处理 ===")

    # 复制数据以避免修改原始数据
    df_processed = df.copy()

    # 验证所有必需的列是否存在
    missing_features = [col for col in FEATURE_COLS if col not in df_processed.columns]
    if missing_features:
        raise ValueError(f"缺少必需的特征列: {missing_features}")

    if TARGET_COL not in df_processed.columns:
        raise ValueError(f"在数据集中未找到目标列 '{TARGET_COL}'")

    # 显示数据样本
    print("\n前5行样本:")
    print(df_processed[FEATURE_COLS + [TARGET_COL]].head())

    # 检查并处理缺失值
    missing_values = df_processed[FEATURE_COLS + [TARGET_COL]].isnull().sum()
    if missing_values.sum() > 0:
        print(f"\n发现{missing_values.sum()}个缺失值")
        # 用均值填充数值特征，用众数填充分类特征
        for col in FEATURE_COLS:
            if df_processed[col].isnull().sum() > 0:
                if pd.api.types.is_numeric_dtype(df_processed[col]):
                    df_processed[col].fillna(df_processed[col].mean(), inplace=True)
                    print(f"在'{col}'中用均值填充了{df_processed[col].isnull().sum()}个缺失值")
                else:
                    df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
                    print(f"在'{col}'中用众数填充了{df_processed[col].isnull().sum()}个缺失值")

        # 处理目标列中的缺失值（如果有）
        if df_processed[TARGET_COL].isnull().sum() > 0:
            df_processed[TARGET_COL].fillna(df_processed[TARGET_COL].mode()[0], inplace=True)
            print(f"在'{TARGET_COL}'中用众数填充了{df_processed[TARGET_COL].isnull().sum()}个缺失值")

        print("缺失值已处理")
    else:
        print("\n选定列中没有缺失值")

    # 检查目标变量分布
    target_counts = df_processed[TARGET_COL].value_counts().sort_index()
    print(f"\n目标变量分布:")
    print(target_counts)

    # 绘制目标分布
    plt.figure(figsize=(8, 6))
    target_counts.plot(kind='bar')
    plt.title('Target Distribution')
    plt.xlabel('Label')
    plt.ylabel('Number')
    plt.savefig('output/target_distribution.png')
    plt.close()

    # 分离特征和目标变量 - 仅使用指定的特征列
    X = df_processed[FEATURE_COLS]
    y = df_processed[TARGET_COL]

    print(f"\n特征维度: {X.shape}")
    print(f"选定的特征: {X.columns.tolist()}")

    # 将任何字符串列转换为数值（如果可能）
    for col in X.columns:
        if X[col].dtype == 'object':
            try:
                X[col] = pd.to_numeric(X[col])
                print(f"将列'{col}'转换为数值型")
            except ValueError:
                print(f"警告: 列'{col}'包含非数值型数据，无法标准化")
                # 考虑为分类变量进行独热编码

    # 标准化数值特征
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    print("\n数据预处理完成")

    return df_processed, X_scaled, y


def split_data(X, y):
    """
    分割训练集和测试集

    参数:
        X: 特征数据
        y: 目标变量

    返回:
        训练集和测试集
    """
    print("\n=== 数据集分割 ===")

    # 使用分层抽样以保持类别比例
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    print(f"训练集大小: {X_train.shape[0]}个样本")
    print(f"测试集大小: {X_test.shape[0]}个样本")

    # 验证每个集合中的目标分布
    print("训练集目标分布:")
    print(y_train.value_counts(normalize=True))
    print("测试集目标分布:")
    print(y_test.value_counts(normalize=True))

    return X_train, X_test, y_train, y_test


def train_xgboost_model(X_train, y_train, X_test, y_test):
    """
    训练XGBoost分类模型

    参数:
        X_train, y_train: 训练数据
        X_test, y_test: 测试数据

    返回:
        训练好的模型
    """
    print("\n=== XGBoost模型训练 ===")

    # 检查目标中的唯一类别
    unique_classes = np.sort(y_train.unique())
    print(f"目标变量中的唯一类别: {unique_classes}")

    # 类别从1开始而不是0，需要调整
    # 可以从目标值中减去1或相应地设置XGBoost参数

    # 选项1: 从目标值中减去1（从1为基准转换为0为基准）
    y_train_0indexed = y_train - 1
    y_test_0indexed = y_test - 1

    print(f"调整后的唯一类别: {np.sort(y_train_0indexed.unique())}")

    # 使用改进的参数初始化模型
    best_model = xgb.XGBClassifier(
        objective='multi:softprob',  # 多类问题
        num_class=4,  # 4个类别
        random_state=RANDOM_SEED,
        n_estimators=300,  # 增加树的数量
        learning_rate=0.01,  # 降低学习率
        # subsample=0.9,              # 样本抽样
        # colsample_bytree=0.9,       # 特征抽样
        # max_delta_step=1,           # 有助于处理不平衡数据
        # reg_lambda=1.5,             # L2正则化
    )

    # 使用5折交叉验证而不是3折
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    # 使用更集中的网格搜索和有前景的参数
    grid_search = GridSearchCV(
        estimator=best_model,
        param_grid={
            'max_depth': [5, 8, 9, 10],
            'min_child_weight': [3, 5, 7],
            'gamma': [0, 0.1],
            'reg_alpha': [0.5]
        },
        cv=cv,
        scoring='f1_macro',  # 使用宏平均F1分数更适合多分类不平衡情况
        n_jobs=4,
        verbose=1
    )

    print("开始为最优参数进行网格搜索...")
    # 使用调整后的0为基准的标签进行训练
    grid_search.fit(X_train, y_train_0indexed)

    # 输出最优参数
    print(f"最佳参数: {grid_search.best_params_}")
    print(f"最佳交叉验证分数: {grid_search.best_score_:.4f}")

    # 使用最优参数重新训练模型
    best_model = grid_search.best_estimator_

    # best_model.fit(X_train, y_train_0indexed)

    # 在训练集上评估
    y_train_pred = best_model.predict(X_train)
    print("\n训练集表现:")
    print(f"准确率: {accuracy_score(y_train_0indexed, y_train_pred):.4f}")

    # 在测试集上评估
    y_test_pred = best_model.predict(X_test)
    print("\n测试集表现:")
    print(f"准确率: {accuracy_score(y_test_0indexed, y_test_pred):.4f}")

    # 重要提示: 对于后续使用，我们需要将预测值调整回原始尺度(1-4)
    # 这在evaluate_model函数中处理

    return best_model, y_train_0indexed, y_test_0indexed  # 返回调整后的标签用于评估


def evaluate_model(model, X_test, y_test_original, y_test_0indexed):
    """
    全面的模型性能评估

    参数:
        model: 训练好的模型
        X_test: 测试数据特征
        y_test_original: 原始测试标签(1-4)
        y_test_0indexed: 调整后的测试标签(0-3)
    """
    print("\n=== 模型评估 ===")

    # 预测测试集(预测将是0为基准)
    y_pred_0indexed = model.predict(X_test)

    # 将预测转换回原始尺度(1-4)以便更好地解释
    y_pred_original = y_pred_0indexed + 1

    print(f"预测值范围: {np.min(y_pred_original)}到{np.max(y_pred_original)}")

    # # 使用原始尺度计算混淆矩阵
    # cm = confusion_matrix(y_test_original, y_pred_original)

    # # 用原始标签(1-4)绘制混淆矩阵
    # plt.figure(figsize=(8, 6))
    # # 修改heatmap参数，确保显示所有值，包括0
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
    #             annot_kws={"size": 14}, linewidths=0.5, linecolor='gray')
    # plt.title('混淆矩阵')
    # plt.xlabel('预测标签')
    # plt.ylabel('真实标签')
    # # 设置刻度标签为原始尺度(1-4)
    # classes = sorted(np.unique(np.concatenate([y_test_original, y_pred_original])))
    # plt.xticks(np.arange(len(classes)) + 0.5, classes)
    # plt.yticks(np.arange(len(classes)) + 0.5, classes)
    # plt.savefig('output/confusion_matrix.png')
    # plt.close()

    # 计算并输出分类报告
    # 使用0为基准以保持与模型内部表示的一致性
    print("\n分类报告(0为基准标签):")
    report_0indexed = classification_report(y_test_0indexed, y_pred_0indexed)
    print(report_0indexed)

    # 也提供带原始标签的报告以便解释
    print("\n分类报告(原始1为基准标签):")
    report_original = classification_report(y_test_original, y_pred_original)
    print(report_original)

    # 计算主要指标(使用0为基准以保持一致性)
    accuracy = accuracy_score(y_test_0indexed, y_pred_0indexed)
    precision = precision_score(y_test_0indexed, y_pred_0indexed, average='macro')
    recall = recall_score(y_test_0indexed, y_pred_0indexed, average='macro')
    f1 = f1_score(y_test_0indexed, y_pred_0indexed, average='macro')

    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分数: {f1:.4f}")

    # 将性能指标保存到文件
    performance = pd.DataFrame({
        'Metric': ['准确率', '精确率', '召回率', 'F1分数'],
        'Value': [accuracy, precision, recall, f1]
    })
    performance.to_csv('output/model_performance.csv', index=False)

    return performance, y_pred_original


def perform_shap_analysis(model, X_test, feature_names, y_pred_original):
    """
    使用SHAP库进行模型解释分析
    生成瀑布图、力图、条形图、蜂群图和依赖图

    参数:
        model: 训练好的XGBoost模型
        X_test: 测试数据特征
        feature_names: 特征名称列表
        y_pred_original: 原始尺度(1-4)的预测
    """
    print("\n=== SHAP分析 ===")

    # 确保feature_names与特征数量匹配
    if len(feature_names) != X_test.shape[1]:
        print(f"警告: 特征名称长度({len(feature_names)})与X_test列数({X_test.shape[1]})不匹配")
        feature_names = X_test.columns.tolist()
        print(f"改用数据框列名: {feature_names}")

    # 创建SHAP解释器
    explainer = shap.TreeExplainer(model)

    # 计算SHAP值
    print("计算SHAP值...")
    shap_values = explainer.shap_values(X_test)

    # 调试信息
    print(f"SHAP值类型: {type(shap_values)}")
    if isinstance(shap_values, list):
        print(f"SHAP值是长度为{len(shap_values)}的列表")
        for i, sv in enumerate(shap_values):
            print(f"  - shap_values[{i}]形状: {sv.shape}")
    else:
        print(f"SHAP值形状: {shap_values.shape}")

    try:
        # 1. 蜂群图(摘要图)
        print("生成蜂群图...")
        if len(shap_values.shape) == 3:
            # 对于多类形状(样本, 特征, 类别)
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values[:, :, 0], X_test, feature_names=feature_names, show=False)
            plt.title('SHAP Beeswarm Plot')
            plt.tight_layout()
            fix_negative_signs(plt.gcf())  # 修复负号显示
            plt.savefig('output/shap_beeswarm.png', bbox_inches='tight')
            plt.close()
        else:
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
            plt.title('SHAP Beeswarm Plot')
            plt.tight_layout()
            fix_negative_signs(plt.gcf())  # 修复负号显示
            plt.savefig('output/shap_beeswarm.png', bbox_inches='tight')
            plt.close()

        # 2. 条形图(特征重要性)
        print("生成特征重要性条形图...")
        # 基于SHAP值计算特征重要性
        if len(shap_values.shape) == 3:  # (样本, 特征, 类别)
            # 在样本和类别上平均
            mean_abs_per_feature_class = np.abs(shap_values).mean(axis=0)  # 形状: (特征, 类别)
            shap_importance = mean_abs_per_feature_class.mean(axis=1)  # 形状: (特征,)
        elif isinstance(shap_values, list):
            # 多类情况，数组列表
            mean_abs_shap = np.zeros(len(feature_names))
            for class_values in shap_values:
                mean_abs_shap += np.abs(class_values).mean(axis=0)
            mean_abs_shap /= len(shap_values)
            shap_importance = mean_abs_shap
        else:
            # 对于二进制/单一输出
            shap_importance = np.abs(shap_values).mean(axis=0)

        # 绘制重要性
        plt.figure(figsize=(10, 6))
        importance_series = pd.Series(shap_importance, index=feature_names)
        importance_series = importance_series.sort_values(ascending=False)
        importance_series.plot(kind='barh')
        plt.xlabel('Average |SHAP| Value')
        plt.title('Feature Importance')
        plt.tight_layout()
        fix_negative_signs(plt.gcf())  # 修复负号显示
        plt.savefig('output/shap_feature_importance.png')
        plt.close()

        # 创建用于后续使用的DataFrame
        importance_df = pd.DataFrame({
            'Feature': importance_series.index,
            'Importance': importance_series.values
        })

        # 仅对前5个特征进行进一步分析
        top_features = importance_df['Feature'].iloc[:min(5, len(importance_df))].tolist()

        # 3. 瀑布图
        print("生成瀑布图...")
        if len(shap_values.shape) == 3:  # (样本, 特征, 类别)
            # 使用选定样本为每个类别生成瀑布图
            for class_idx in range(min(4, shap_values.shape[2])):
                class_value = class_idx + 1  # 1为基准的类别值
                # 找到预测为此类别的样本
                class_samples = np.where(y_pred_original == class_value)[0]
                if len(class_samples) > 0:
                    sample_idx = class_samples[0]
                    try:
                        # 使用更新的SHAP API
                        plt.figure(figsize=(10, 6))
                        # 获取此样本和类别的SHAP值
                        sample_values = shap_values[sample_idx, :, class_idx]
                        # 获取此类别的期望值
                        if isinstance(explainer.expected_value, list):
                            base_value = explainer.expected_value[class_idx]
                        else:
                            base_value = explainer.expected_value

                        # 为新的SHAP版本使用正确的格式
                        shap.plots.waterfall(
                            shap.Explanation(
                                values=sample_values,
                                base_values=base_value,
                                data=X_test.iloc[sample_idx].values,
                                feature_names=feature_names
                            ),
                            max_display=10, show=False
                        )
                        plt.title(f'SHAP Waterfall Plot - Class {class_value}')
                        plt.tight_layout()
                        fix_negative_signs(plt.gcf())  # 修复负号显示
                        plt.savefig(f'output/shap_waterfall_class{class_value}.png', bbox_inches='tight')
                        plt.close()
                    except Exception as e:
                        print(f"为类别{class_value}生成瀑布图时出错: {str(e)}")

        # 4. 力图
        print("生成力图...")
        try:
            # 仅使用少量样本
            sample_indices = np.random.choice(X_test.shape[0], min(3, X_test.shape[0]), replace=False)

            for i, idx in enumerate(sample_indices):
                plt.figure(figsize=(12, 3))

                # 对于3D SHAP值(样本, 特征, 类别)
                if len(shap_values.shape) == 3:
                    # 使用第一个类别进行可视化
                    class_idx = 0  # 第一个类别
                    sample_values = shap_values[idx, :, class_idx]
                    if isinstance(explainer.expected_value, list):
                        base_value = explainer.expected_value[class_idx]
                    else:
                        base_value = explainer.expected_value

                    # 将样本值四舍五入到3位小数以避免文本重叠
                    sample_values = np.round(sample_values, 3)

                    # 使用更新的SHAP API
                    # 创建带四舍五入值的自定义解释对象
                    explanation = shap.Explanation(
                        values=sample_values,
                        base_values=base_value,
                        data=np.round(X_test.iloc[idx].values, 3),  # 特征值也四舍五入
                        feature_names=feature_names
                    )

                    shap.plots.force(
                        explanation.base_values,
                        explanation.values,
                        explanation.data,
                        feature_names=feature_names,
                        matplotlib=True,
                        show=False,
                        text_rotation=0  # 保持文本水平
                    )
                else:
                    # 对于其他格式(列表或2D数组)
                    if isinstance(shap_values, list):
                        base_value = explainer.expected_value[0]
                        values = np.round(shap_values[0][idx, :], 3)  # 四舍五入到3位小数
                    else:
                        base_value = explainer.expected_value
                        values = np.round(shap_values[idx, :], 3)  # 四舍五入到3位小数

                    # 使用带四舍五入值的更新SHAP API
                    shap.plots.force(
                        base_value,
                        values,
                        np.round(X_test.iloc[idx].values, 3),  # 特征值也四舍五入
                        feature_names=feature_names,
                        matplotlib=True,
                        show=False,
                        text_rotation=0  # 保持文本水平
                    )

                plt.title(f'SHAP Force Plot - Sample_{i + 1}')
                plt.tight_layout()
                fix_negative_signs(plt.gcf())  # 修复负号显示
                plt.savefig(f'output/shap_force_plot_{i + 1}.png', bbox_inches='tight', dpi=150)
                plt.close()
        except Exception as e:
            print(f"生成力图时出错: {str(e)}")

        # 5. 依赖图
        print("生成依赖图...")
        for feature in top_features:
            try:
                feature_idx = feature_names.index(feature)

                # 对于3D SHAP值(样本, 特征, 类别)
                if len(shap_values.shape) == 3:
                    plt.figure(figsize=(8, 6))
                    # 使用第一个类别进行可视化
                    class_values = shap_values[:, :, 0]

                    # 为此特征绘制依赖图
                    shap.plots.scatter(
                        shap.Explanation(
                            values=class_values,
                            data=X_test.values,
                            feature_names=feature_names
                        )[:, feature_idx],
                        color=shap.Explanation(
                            values=class_values,
                            data=X_test.values,
                            feature_names=feature_names
                        )[:, feature_idx],
                        axis_color='#333333',
                        show=False
                    )
                    plt.title(f'SHAP Dependence Plot - {feature}')
                    plt.tight_layout()
                    fix_negative_signs(plt.gcf())  # 修复负号显示
                    plt.savefig(f'output/shap_dependence_{feature}.png')
                    plt.close()
                else:
                    # 对于其他格式
                    plt.figure(figsize=(8, 6))
                    if isinstance(shap_values, list):
                        # 使用第一个类别进行可视化
                        shap.plots.scatter(
                            shap.Explanation(
                                values=shap_values[0],
                                data=X_test.values,
                                feature_names=feature_names
                            )[:, feature_idx],
                            show=False
                        )
                    else:
                        # 为二进制/回归绘制单个图
                        shap.plots.scatter(
                            shap.Explanation(
                                values=shap_values,
                                data=X_test.values,
                                feature_names=feature_names
                            )[:, feature_idx],
                            show=False
                        )

                    plt.title(f'SHAP Dependence Plot - {feature}')
                    plt.tight_layout()
                    fix_negative_signs(plt.gcf())  # 修复负号显示
                    plt.savefig(f'output/shap_dependence_{feature}.png')
                    plt.close()
            except Exception as e:
                print(f"为{feature}生成依赖图时出错: {str(e)}")
    except Exception as e:
        print(f"SHAP分析中出错: {str(e)}")
        import traceback
        traceback.print_exc()


def fix_negative_signs(fig):
    """
    修正matplotlib图形中的负号显示问题，包括图例、轴标签、标题、注释等所有文本元素
    全面处理Unicode负号(\u2212)转换为标准连字符(-)

    参数:
        fig: matplotlib图表对象

    返回:
        处理后的matplotlib图表对象
    """
    # 遍历所有坐标轴(包括子图)
    for ax in fig.get_axes():
        # 修复坐标轴标题和标签
        if ax.get_title():
            ax.set_title(ax.get_title().replace('\u2212', '-'))

        if ax.get_xlabel():
            ax.set_xlabel(ax.get_xlabel().replace('\u2212', '-'))

        if ax.get_ylabel():
            ax.set_ylabel(ax.get_ylabel().replace('\u2212', '-'))

        # 修复所有文本对象
        for text in ax.texts:
            text.set_text(text.get_text().replace('\u2212', '-'))

        # 修复x刻度标签
        xtick_labels = [label.get_text().replace('\u2212', '-') for label in ax.get_xticklabels()]
        ax.set_xticklabels(xtick_labels)

        # 修复y刻度标签
        ytick_labels = [label.get_text().replace('\u2212', '-') for label in ax.get_yticklabels()]
        ax.set_yticklabels(ytick_labels)

        # 修复图例中的负号
        if ax.get_legend() is not None:
            for text in ax.get_legend().get_texts():
                text.set_text(text.get_text().replace('\u2212', '-'))

            # 处理图例标题
            if ax.get_legend().get_title_text():
                title_text = ax.get_legend().get_title_text()
                title_text.set_text(title_text.get_text().replace('\u2212', '-'))

        # 修复注释
        for child in ax.get_children():
            # 处理所有可能包含文本的元素
            if hasattr(child, 'get_text') and callable(getattr(child, 'get_text')):
                try:
                    current_text = child.get_text()
                    if current_text and isinstance(current_text, str):
                        child.set_text(current_text.replace('\u2212', '-'))
                except Exception:
                    pass

            # 处理包含文本的集合对象(如散点图标签等)
            if hasattr(child, 'get_texts') and callable(getattr(child, 'get_texts')):
                try:
                    for text_item in child.get_texts():
                        if hasattr(text_item, 'get_text') and callable(getattr(text_item, 'get_text')):
                            text_item.set_text(text_item.get_text().replace('\u2212', '-'))
                except Exception:
                    pass

    # 处理整个图表的suptitle(总标题)
    if fig._suptitle:
        fig._suptitle.set_text(fig._suptitle.get_text().replace('\u2212', '-'))

    return fig


def main():
    """主函数"""
    print("=== 使用XGBoost和SHAP进行引用数据分析 ===\n")

    # 1. 加载数据
    file_path = "C:/Users/fangxiang/Downloads/GMM_5+5/computer_science.xlsx"
    df = load_data(file_path)

    # 2. 数据预处理
    df_processed, X_scaled, y = preprocess_data(df)

    # 3. 分割训练集和测试集
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)

    # 4. 训练XGBoost模型
    model, y_train_0indexed, y_test_0indexed = train_xgboost_model(X_train, y_train, X_test, y_test)

    # 5. 评估模型
    performance, y_pred_original = evaluate_model(model, X_test, y_test, y_test_0indexed)

    # 6. SHAP分析
    feature_names = X_scaled.columns.tolist()
    perform_shap_analysis(model, X_test, feature_names, y_pred_original)

    print("\n=== 分析完成! ===")
    print(f"所有结果已保存到'output'文件夹")


if __name__ == "__main__":
    main()
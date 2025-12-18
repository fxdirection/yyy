"""python
# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
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
discipline = "computer_science"
# 忽略警告信息
warnings.filterwarnings('ignore')

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
TARGET_COL = "Label"

# 列名映射字典（中文 → 英文缩写）
COLUMN_MAPPING = {
    '标题长度': 'TL',
    '作者数量': 'NoA',
    '作者h指数': 'HI',
    '参考文献数量': 'NoR',
    '五年影响因子': 'JIF',
    'year_1': 'ECGR',
    '篇幅': 'PL',
    '十年CNCI': 'PACNCI',
}

# 要使用的特征列（明确定义，使用英文缩写）
FEATURE_COLS = ['TL', 'NoA', 'HI', 'TNoC', 'NoR', 'JIF', 'ECGR', 'PL', 'PACNCI']

# 创建输出目录
output_path = f"{discipline}_output"
if not os.path.exists(output_path):
    os.makedirs(output_path)


def load_data(file_path):
    """
    加载Excel数据文件
    """
    print(f"加载数据: {file_path}")
    try:
        data = pd.read_excel(file_path)
        print(f"数据加载成功: {data.shape[0]}行, {data.shape[1]}列")

        # 计算TNoC
        data['TNoC'] = data.iloc[:, 13:18].sum(axis=1) + data.iloc[:, 23:28].sum(axis=1)
        print("已计算TNoC（预测总引用量）")

        # 重命名列为英文缩写
        data.rename(columns=COLUMN_MAPPING, inplace=True)
        print("已将列名映射为英文缩写")

        # 将第39列（索引38）命名为'Label'
        if data.shape[1] > 38:
            data.rename(columns={data.columns[30]: 'Label'}, inplace=True)
            print("已将第39列命名为'Label'作为目标变量")
        else:
            raise ValueError(f"数据只有{data.shape[1]}列，无法访问第39列")

        # 验证必需的列是否存在
        missing_cols = [col for col in FEATURE_COLS + [TARGET_COL] if col not in data.columns]
        if missing_cols:
            print(f"警告: 以下必需列缺失: {missing_cols}")

        # 打印列名以验证
        print("\n数据集中的列名:")
        print(data.columns.tolist())

        # 检查数据类型
        print("\n数据类型:")
        print(data.dtypes)

        return data
    except Exception as e:
        print(f"数据加载失败: {str(e)}")
        raise


def explore_data(df):
    """
    探索性数据分析
    """
    print("\n=== 数据探索 ===")

    # 查看基本信息
    print("\n基本信息:")
    print(df.info())

    # 查看统计摘要
    print("\n统计摘要:")
    print(df[FEATURE_COLS].describe())

    # 检查缺失值
    missing_values = df[FEATURE_COLS + [TARGET_COL]].isnull().sum()
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
        plt.title('Citation Quantile Distribution')
        plt.xlabel('Citation Quantile Label')
        plt.ylabel('Count')
        plt.xticks(rotation=0)
        plt.savefig(f'{output_path}/target_distribution.png')
        plt.close()
    else:
        print(f"\n警告: 在数据集中未找到目标列 '{TARGET_COL}'")


def preprocess_data(df):
    """
    数据预处理: 处理缺失值、标准化等
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
    plt.title('Target Variable Distribution')
    plt.xlabel('Citation Quantile Label')
    plt.ylabel('Count')
    plt.savefig(f'{output_path}/target_distribution.png')
    plt.close()

    # 分离特征和目标变量 - 仅使用指定的特征列
    X = df_processed[FEATURE_COLS]
    y = df_processed[TARGET_COL]

    print(f"\n特征维度: {X.shape}")
    print(f"选定的特征: {X.columns.tolist()}")

    # 将任何字符串列转换为数值（如果可能）
    for col in X.columns:
        if X[col].dtypes == 'object':  # Corrected from dtype to dtypes
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
    """
    print("\n=== 数据集分割 ===")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    print(f"训练集大小: {X_train.shape[0]}个样本")
    print(f"测试集大小: {X_test.shape[0]}个样本")

    print("训练集目标分布:")
    print(y_train.value_counts(normalize=True))
    print("测试集目标分布:")
    print(y_test.value_counts(normalize=True))

    return X_train, X_test, y_train, y_test


def train_xgboost_model(X_train, y_train, X_test, y_test):
    """
    训练XGBoost分类模型
    """
    print("\n=== XGBoost模型训练 ===")

    unique_classes = np.sort(y_train.unique())
    print(f"目标变量中的唯一类别: {unique_classes}")

    y_train_0indexed = y_train - 1
    y_test_0indexed = y_test - 1

    print(f"调整后的唯一类别: {np.sort(y_train_0indexed.unique())}")

    best_model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=4,
        random_state=RANDOM_SEED,
        n_estimators=300,
        learning_rate=0.01,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    grid_search = GridSearchCV(
        estimator=best_model,
        param_grid={
            'max_depth': [5, 8, 9, 10],
            'min_child_weight': [3, 5, 7],
            'gamma': [0, 0.1],
            'reg_alpha': [0.5]
        },
        cv=cv,
        scoring='f1_macro',
        n_jobs=4,
        verbose=1
    )

    print("开始为最优参数进行网格搜索...")
    grid_search.fit(X_train, y_train_0indexed)

    print(f"最佳参数: {grid_search.best_params_}")
    print(f"最佳交叉验证分数: {grid_search.best_score_:.4f}")

    best_model = grid_search.best_estimator_

    y_train_pred = best_model.predict(X_train)
    print("\n训练集表现:")
    print(f"准确率: {accuracy_score(y_train_0indexed, y_train_pred):.4f}")

    y_test_pred = best_model.predict(X_test)
    print("\n测试集表现:")
    print(f"准确率: {accuracy_score(y_test_0indexed, y_test_pred):.4f}")

    return best_model, y_train_0indexed, y_test_0indexed


def evaluate_model(model, X_test, y_test_original, y_test_0indexed):
    """
    全面的模型性能评估
    """
    print("\n=== 模型评估 ===")

    y_pred_0indexed = model.predict(X_test)
    y_pred_original = y_pred_0indexed + 1

    print(f"预测值范围: {np.min(y_pred_original)}到{np.max(y_pred_original)}")

    print("\n分类报告(0为基准标签):")
    report_0indexed = classification_report(y_test_0indexed, y_pred_0indexed)
    print(report_0indexed)

    print("\n分类报告(原始1为基准标签):")
    report_original = classification_report(y_test_original, y_pred_original)
    print(report_original)

    accuracy = accuracy_score(y_test_0indexed, y_pred_0indexed)
    precision = precision_score(y_test_0indexed, y_pred_0indexed, average='macro')
    recall = recall_score(y_test_0indexed, y_pred_0indexed, average='macro')
    f1 = f1_score(y_test_0indexed, y_pred_0indexed, average='macro')

    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分数: {f1:.4f}")

    performance = pd.DataFrame({
        'Metric': ['准确率', '精确率', '召回率', 'F1分数'],
        'Value': [accuracy, precision, recall, f1]
    })
    performance.to_csv(f'{output_path}/model_performance.csv', index=False)

    return performance, y_pred_original


def perform_shap_analysis(model, X_test, feature_names, y_pred_original):
    """
    使用SHAP库进行模型解释分析
    """
    print("\n=== SHAP分析 ===")

    df_processed = pd.read_excel(f'C:/Users/fangxiang/Downloads/GMM_5+5/{discipline}.xlsx')
    df_processed.rename(columns=COLUMN_MAPPING, inplace=True)
    df_processed['TNoC'] = df_processed.iloc[:, 13:18].sum(axis=1) + df_processed.iloc[:, 23:28].sum(axis=1)
    if df_processed.shape[1] > 38:
        df_processed.rename(columns={df_processed.columns[38]: 'Label'}, inplace=True)
    X_original = df_processed[feature_names].iloc[-len(X_test):]

    if len(feature_names) != X_test.shape[1]:
        print(f"警告: 特征名称长度({len(feature_names)})与X_test列数({X_test.shape[1]})不匹配")
        feature_names = X_test.columns.tolist()
        print(f"改用数据框列名: {feature_names}")

    explainer = shap.TreeExplainer(model)

    print("计算SHAP值...")
    shap_values = explainer.shap_values(X_test)

    print(f"SHAP值类型: {type(shap_values)}")
    if isinstance(shap_values, list):
        print(f"SHAP值是长度为{len(shap_values)}的列表")
        for i, sv in enumerate(shap_values):
            print(f"  - shap_values[{i}]形状: {sv.shape}")
    else:
        print(f"SHAP值形状: {shap_values.shape}")

    try:
        class_samples = {}
        for class_idx in range(4):
            class_value = class_idx + 1
            class_indices = np.where(y_pred_original == class_value)[0]
            if len(class_indices) > 0:
                class_samples[class_idx] = class_indices[0]
                print(f"类别{class_value}的代表性样本索引: {class_samples[class_idx]}")

        print("生成蜂群图...")
        if len(shap_values.shape) == 3:
            for class_idx in range(min(4, shap_values.shape[2])):
                class_value = class_idx + 1
                plt.figure(figsize=(10, 8))
                shap.summary_plot(shap_values[:, :, class_idx], X_test, feature_names=feature_names, show=False)
                plt.title(f'SHAP Beeswarm Plot - Category {class_value-1}')
                plt.tight_layout()
                fix_negative_signs(plt.gcf())
                plt.savefig(f'{output_path}/shap_beeswarm_class{class_value-1}.png', bbox_inches='tight')
                plt.close()
        else:
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
            plt.title('SHAP Beeswarm Plot')
            plt.tight_layout()
            fix_negative_signs(plt.gcf())
            plt.savefig(f'{output_path}/shap_beeswarm.png', bbox_inches='tight')
            plt.close()

        print("生成特征重要性条形图...")
        if len(shap_values.shape) == 3:
            mean_abs_per_feature_class = np.abs(shap_values).mean(axis=0)
            shap_importance = mean_abs_per_feature_class.mean(axis=1)
        elif isinstance(shap_values, list):
            mean_abs_shap = np.zeros(len(feature_names))
            for class_values in shap_values:
                mean_abs_shap += np.abs(class_values).mean(axis=0)
            mean_abs_shap /= len(shap_values)
            shap_importance = mean_abs_shap
        else:
            shap_importance = np.abs(shap_values).mean(axis=0)

        plt.figure(figsize=(10, 6))
        importance_series = pd.Series(shap_importance, index=feature_names)
        importance_series = importance_series.sort_values(ascending=False)
        importance_series.plot(kind='barh')
        plt.xlabel('Average SHAP Value')
        plt.title('Feature Importance')
        plt.tight_layout()
        fix_negative_signs(plt.gcf())
        plt.savefig(f'{output_path}/shap_feature_importance.png')
        plt.close()

        importance_df = pd.DataFrame({
            'Feature': importance_series.index,
            'Importance': importance_series.values
        })

        top_features = importance_df['Feature'].iloc[:min(3, len(importance_df))].tolist()

        print("生成瀑布图...")
        if len(shap_values.shape) == 3:
            for class_idx in range(min(4, shap_values.shape[2])):
                class_value = class_idx + 1
                if class_idx in class_samples:
                    sample_idx = class_samples[class_idx]
                    try:
                        plt.figure(figsize=(10, 6))
                        sample_values = shap_values[sample_idx, :, class_idx]
                        if isinstance(explainer.expected_value, list):
                            base_value = explainer.expected_value[class_idx]
                        else:
                            base_value = explainer.expected_value

                        shap.plots.waterfall(
                            shap.Explanation(
                                values=sample_values,
                                base_values=base_value,
                                data=X_test.iloc[sample_idx].values,
                                feature_names=feature_names
                            ),
                            max_display=10, show=False
                        )
                        plt.title(f'SHAP Waterfall Plot - Category {class_value - 1}')
                        plt.tight_layout()
                        fix_negative_signs(plt.gcf())
                        plt.savefig(f'{output_path}/shap_waterfall_class{class_value - 1}.png', bbox_inches='tight')
                        plt.close()
                    except Exception as e:
                        print(f"为类别{class_value}生成瀑布图时出错: {str(e)}")

        print("生成力图...")
        try:
            for class_idx in range(min(4, shap_values.shape[2])):
                class_value = class_idx + 1
                if class_idx in class_samples:
                    sample_idx = class_samples[class_idx]
                    plt.figure(figsize=(12, 3))

                    if len(shap_values.shape) == 3:
                        sample_values = shap_values[sample_idx, :, class_idx]
                        if isinstance(explainer.expected_value, list):
                            base_value = explainer.expected_value[class_idx]
                        else:
                            base_value = explainer.expected_value

                        sample_values = np.round(sample_values, 3)

                        explanation = shap.Explanation(
                            values=sample_values,
                            base_values=base_value,
                            data=np.round(X_test.iloc[sample_idx].values, 3),
                            feature_names=feature_names
                        )

                        shap.plots.force(
                            explanation.base_values,
                            explanation.values,
                            explanation.data,
                            feature_names=feature_names,
                            matplotlib=True,
                            show=False,
                            text_rotation=0
                        )

                        plt.title(f'SHAP Force Plot - Category {class_value - 1}')
                        plt.tight_layout()
                        fix_negative_signs(plt.gcf())
                        plt.savefig(f'{output_path}/shap_force_plot_class{class_value - 1}.png', bbox_inches='tight', dpi=150)
                        plt.close()

        except Exception as e:
            print(f"生成力图时出错: {str(e)}")

        print("生成依赖图...")
        for feature in top_features:
            try:
                feature_idx = feature_names.index(feature)

                if len(shap_values.shape) == 3:
                    for class_idx in range(min(4, shap_values.shape[2])):
                        class_value = class_idx + 1
                        plt.figure(figsize=(8, 6))
                        class_values = shap_values[:, :, class_idx]

                        explanation = shap.Explanation(
                            values=class_values[:, feature_idx],
                            data=X_original[feature].values,
                            feature_names=[feature]
                        )

                        plt.scatter(X_original[feature].values, class_values[:, feature_idx],
                                    alpha=0.5, s=30, c=class_values[:, feature_idx], cmap='coolwarm')
                        plt.colorbar(label='SHAP Value')
                        plt.xlabel(feature)
                        plt.ylabel('SHAP Value')
                        plt.title(f'SHAP Dependence Plot - {feature} - Category {class_value - 1}')
                        plt.tight_layout()
                        fix_negative_signs(plt.gcf())
                        plt.savefig(f'{output_path}/shap_dependence_{feature}_class{class_value - 1}.png')
                        plt.close()

            except Exception as e:
                print(f"为{feature}生成依赖图时出错: {str(e)}")
    except Exception as e:
        print(f"SHAP分析中出错: {str(e)}")
        traceback.print_exc()


def fix_negative_signs(fig):
    """
    修正matplotlib图形中的负号显示问题
    """
    for ax in fig.get_axes():
        if ax.get_title():
            ax.set_title(ax.get_title().replace('\u2212', '-'))

        if ax.get_xlabel():
            ax.set_xlabel(ax.get_xlabel().replace('\u2212', '-'))

        if ax.get_ylabel():
            ax.set_ylabel(ax.get_ylabel().replace('\u2212', '-'))

        for text in ax.texts:
            text.set_text(text.get_text().replace('\u2212', '-'))

        xtick_labels = [label.get_text().replace('\u2212', '-') for label in ax.get_xticklabels()]
        ax.set_xticklabels(xtick_labels)

        ytick_labels = [label.get_text().replace('\u2212', '-') for label in ax.get_yticklabels()]
        ax.set_yticklabels(ytick_labels)

        if ax.get_legend() is not None:
            for text in ax.get_legend().get_texts():
                text.set_text(text.get_text().replace('\u2212', '-'))

            if ax.get_legend().get_title_text():
                title_text = ax.get_legend().get_title_text()
                title_text.set_text(title_text.get_text().replace('\u2212', '-'))

        for child in ax.get_children():
            if hasattr(child, 'get_text') and callable(getattr(child, 'get_text')):
                try:
                    current_text = child.get_text()
                    if current_text and isinstance(current_text, str):
                        child.set_text(current_text.replace('\u2212', '-'))
                except Exception:
                    pass

            if hasattr(child, 'get_texts') and callable(getattr(child, 'get_texts')):
                try:
                    for text_item in child.get_texts():
                        if hasattr(text_item, 'get_text') and callable(getattr(text_item, 'get_text')):
                            text_item.set_text(text_item.get_text().replace('\u2212', '-'))
                except Exception:
                    pass

    if fig._suptitle:
        fig._suptitle.set_text(fig._suptitle.get_text().replace('\u2212', '-'))

    return fig


def main():
    """主函数"""
    print("=== 使用XGBoost和SHAP进行引用数据分析 ===\n")

    file_path = f'C:/Users/fangxiang/Downloads/GMM_5+5/{discipline}.xlsx'
    df = load_data(file_path)

    df_processed, X_scaled, y = preprocess_data(df)

    X_train, X_test, y_train, y_test = split_data(X_scaled, y)

    model, y_train_0indexed, y_test_0indexed = train_xgboost_model(X_train, y_train, X_test, y_test)

    performance, y_pred_original = evaluate_model(model, X_test, y_test, y_test_0indexed)

    feature_names = X_scaled.columns.tolist()
    perform_shap_analysis(model, X_test, feature_names, y_pred_original)

    print("\n=== 分析完成! ===")
    print(f"所有结果已保存到'{output_path}'文件夹")


if __name__ == "__main__":
    main()

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (confusion_matrix, cohen_kappa_score,
                             f1_score, mean_absolute_error)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


# 列名映射
column_mapping = {
    '标题长度': 'TL',
    '作者数量': 'NoA',
    '作者h指数': 'HI',
    '预测总引用量': 'TNoC',  # Will be computed as sum of columns 14-18 and 24-28
    '参考文献数量': 'NoR',
    '五年影响因子': 'JIF',
    'year_1': 'ECGR',
    '篇幅': 'PL',
    '十年CNCI': 'PACNCI'
}


def plot_confusion_matrix(cm, classes, filename, title='混淆矩阵', normalize='true'):
    """
    绘制并保存标准化混淆矩阵图（按行标准化）
    参数:
        normalize: 'true'按行标准化
    """
    plt.figure(figsize=(10, 8))

    # Standardize by row (normalize to show proportions)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm = np.nan_to_num(cm)  # Replace NaN with 0
    fmt = '.2f'
    vmin, vmax = 0, 1

    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                vmin=vmin, vmax=vmax,
                )

    plt.title(title)
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


class XGBoostFeatureTransformer:
    """封装XGBoost特征转换过程"""

    def __init__(self):
        self.encoder = OneHotEncoder(handle_unknown='ignore')
        self.is_fitted = False

    def transform(self, model, X):
        leaf_indices = model.apply(X)
        if not self.is_fitted:
            self.encoder.fit(leaf_indices)
            self.is_fitted = True
        return self.encoder.transform(leaf_indices)


def process_excel_files(folder_path, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    cm_dir = os.path.join(output_dir, "confusion_matrices")
    os.makedirs(cm_dir, exist_ok=True)

    excel_files = [f for f in os.listdir(folder_path) if f.endswith(('.xlsx', '.xls'))]

    if not excel_files:
        print("指定文件夹中没有找到Excel文件")
        return

    all_results = []

    for file in tqdm(excel_files, desc="处理文件中"):
        file_path = os.path.join(folder_path, file)
        output_path = os.path.join(output_dir, file)

        try:
            # 数据读取与处理
            df = pd.read_excel(file_path)

            # Check if required columns exist
            if df.shape[1] < 39:
                print(f"Warning: {file} has only {df.shape[1]} columns, expected at least 39. Skipping.")
                continue

            # Calculate TNoC as sum of columns 14-18 (index 13-17) and 24-28 (index 23-27)
            df['TNoC'] = df.iloc[:, 13:18].sum(axis=1) + df.iloc[:, 23:28].sum(axis=1)

            # Map features using column_mapping
            feature_columns = {v: k for k, v in column_mapping.items()}  # Reverse mapping
            features = ['TNoC']  # TNoC is already computed
            for feature in ['HI', 'JIF', 'PL', 'TL', 'NoA', 'NoR', 'ECGR', 'PACNCI']:
                if feature_columns[feature] in df.columns:
                    features.append(feature_columns[feature])
                else:
                    print(f"Warning: Feature {feature_columns[feature]} not found in {file}. Skipping feature.")
                    df[feature_columns[feature]] = np.nan  # Add as NaN if missing

            # Extract target (column 31, index 30) and reduce labels by 1
            df['Target'] = df.iloc[:, 30] - 1
            unique_classes = np.sort(df['Target'].dropna().unique().astype(int))
            if len(unique_classes) < 2:
                print(f"Warning: {file} has fewer than 2 unique classes. Skipping.")
                continue

            # 准备数据
            features_df = df[features]
            target = df['Target']
            X_train, X_test, y_train, y_test = train_test_split(
                features_df, target, test_size=0.1, random_state=42, stratify=target)

            # 预处理
            numeric_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())])
            preprocessor = ColumnTransformer([('num', numeric_transformer, features_df.columns)])

            # 模型训练
            xgb_model = xgb.XGBClassifier(
                learning_rate=0.2, n_estimators=80, max_depth=7,
                gamma=0.001, random_state=42, eval_metric='logloss')
            X_train_preprocessed = preprocessor.fit_transform(X_train)
            xgb_model.fit(X_train_preprocessed, y_train)

            # 特征转换
            feature_transformer = XGBoostFeatureTransformer()
            X_train_transformed = feature_transformer.transform(xgb_model, X_train_preprocessed)

            # LR模型
            lr_model = LogisticRegression(C=0.01, penalty='l2', random_state=42, max_iter=1000)
            lr_model.fit(X_train_transformed, y_train)

            # 预测
            X_all_preprocessed = preprocessor.transform(features_df)
            X_all_transformed = feature_transformer.transform(xgb_model, X_all_preprocessed)
            df['Predicted_Class'] = lr_model.predict(X_all_transformed) + 1  # Add 1 to match original labels

            # 评估（使用测试集）
            X_test_preprocessed = preprocessor.transform(X_test)
            X_test_transformed = feature_transformer.transform(xgb_model, X_test_preprocessed)
            y_pred = lr_model.predict(X_test_transformed)

            # 混淆矩阵（仅标准化版本）
            cm_test = confusion_matrix(y_test, y_pred, labels=unique_classes)
            base_name = os.path.splitext(file)[0]
            cm_filename = os.path.join(cm_dir, f"{base_name}_normalized.png")
            plot_confusion_matrix(cm_test, unique_classes,
                                  cm_filename, title = "XGBoost",
                                  normalize='true')

            # 保存结果
            df.to_excel(output_path, index=False)
            all_results.append({
                '文件名': file,
                '加权Kappa': cohen_kappa_score(y_test, y_pred, weights='quadratic'),
                '加权F1': f1_score(y_test, y_pred, average='weighted'),
                'MAE': mean_absolute_error(y_test, y_pred),
                '混淆矩阵路径': f"confusion_matrices/{base_name}_normalized.png"
            })

        except Exception as e:
            print(f"处理文件 {file} 时出错: {str(e)}")
            continue

    # 保存汇总结果
    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_path = os.path.join(output_dir, "模型评估汇总.xlsx")
        # Use absolute paths for hyperlinks
        summary_df['混淆矩阵路径'] = summary_df['混淆矩阵路径'].apply(
            lambda x: f'=HYPERLINK("{os.path.abspath(os.path.join(output_dir, x))}", "查看混淆矩阵")')
        summary_df.to_excel(summary_path, index=False)
        print(f"模型评估汇总已保存至: {summary_path}")


if __name__ == "__main__":
    process_excel_files("C:/Users/fangxiang/Downloads/GMM_5+5",
                        "C:/Users/fangxiang/Downloads/XGBoost+LR_res")
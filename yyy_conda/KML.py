
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tslearn.clustering import TimeSeriesKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (cohen_kappa_score,
                             mean_absolute_error,
                             f1_score,
                             ConfusionMatrixDisplay)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# ======================
# 数据加载与预处理
# ======================
def load_and_preprocess(file_path):
    """
    加载数据并进行预处理
    """
    df = pd.read_excel(file_path)
    # 计算TNoC (N-P列, 13-17索引)
    df['TNoC'] = df.iloc[:, 13:18].sum(axis=1)

    # 将第39列（索引38）命名为'Label'作为目标变量
    if df.shape[1] > 38:
        df.rename(columns={df.columns[38]: 'Label'}, inplace=True)
    else:
        raise ValueError(f"数据只有{df.shape[1]}列，无法访问第39列")

    # 获取指定特征
    features = {
        'HI': df.iloc[:, 7],      # H列(7)
        'JIF': df.iloc[:, 11],    # L列(11)
        'PL': df.iloc[:, 33],     # AH列(33)
        'TL': df.iloc[:, 5],      # F列(5)
        'NoA': df.iloc[:, 6],     # G列(6)
        'NoR': df.iloc[:, 10],    # K列(10)
        'ECGR': df.iloc[:, 13],   # N列(13)
        'PACNCI': df.iloc[:, 35]  # AJ列(35)
    }

    # 将特征添加到DataFrame
    for name, values in features.items():
        df[name] = values

    # 处理缺失值
    numeric_cols = ['TNoC', 'HI', 'JIF', 'PL', 'TL', 'NoA', 'NoR', 'ECGR', 'PACNCI']
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # 确保目标列是整数类型
    df['Label'] = pd.to_numeric(df['Label'], errors='coerce').fillna(df['Label'].mode()[0]).astype(int)

    return df

# ======================
# 评估指标计算
# ======================
def calculate_evaluation_metrics(df, true_col='Label', pred_col='新标签'):
    """计算评估指标"""
    if pred_col not in df.columns:
        print(f"警告: 列'{pred_col}'不存在，跳过评估")
        return None

    # 确保数据是数值类型
    true = pd.to_numeric(df[true_col], errors='coerce').fillna(0).astype(int)
    pred = pd.to_numeric(df[pred_col], errors='coerce').fillna(0).astype(int)

    # 计算指标
    kappa = cohen_kappa_score(true, pred, weights='quadratic')
    f1 = f1_score(true, pred, average='weighted')
    mae = mean_absolute_error(true, pred)

    return {'Weighted Kappa': kappa, 'Weighted F1': f1, 'MAE': mae}

# ======================
# KML聚类模块
# ======================
def kml_clustering(data, n_clusters=4):
    """时间序列K均值聚类"""
    # 标准化时间序列数据
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data.T).T

    # 构建纵向K均值模型
    kml = TimeSeriesKMeans(n_clusters=n_clusters,
                           metric="dtw",
                           max_iter=50,
                           random_state=42)
    clusters = kml.fit_predict(scaled_data)
    return clusters

# ======================
# 主执行流程
# ======================
def process_folder(folder_path, output_dir="results"):
    """处理文件夹中的所有Excel文件"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    excel_files = [f for f in os.listdir(folder_path) if f.endswith(('.xlsx', '.xls'))]

    if not excel_files:
        print("指定文件夹中没有找到Excel文件")
        return

    all_results = []
    evaluation_metrics = []

    for file in excel_files:
        file_path = os.path.join(folder_path, file)
        print(f"\n正在处理文件: {file}")

        try:
            # 1. 加载和预处理数据
            df = load_and_preprocess(file_path)

            # 2. 计算AN/AM列的评估指标（如果存在新标签）
            if '新标签' in df.columns:
                metrics = calculate_evaluation_metrics(df, true_col='Label', pred_col='新标签')
                if metrics:
                    metrics['File'] = file
                    evaluation_metrics.append(metrics)
                    print("\nAN/AM列评估结果：")
                    print(f"加权Kappa: {metrics['Weighted Kappa']:.3f}")
                    print(f"加权F1: {metrics['Weighted F1']:.3f}")
                    print(f"MAE: {metrics['MAE']:.3f}")

            # 3. 检查是否有目标变量
            if 'Label' not in df.columns:
                print(f"文件 {file} 没有目标变量列'Label'，跳过...")
                continue

            # 4. 检查目标变量类别
            unique_classes = df['Label'].unique()
            if len(unique_classes) < 2:
                print(f"文件 {file} 的目标列只有1个类别: {unique_classes}，跳过...")
                continue

            # 5. 特征工程
            static_features = df[['TNoC', 'HI', 'JIF', 'PL', 'TL', 'NoA', 'NoR', 'ECGR', 'PACNCI']]

            # 执行KML聚类
            time_series_cols = [col for col in df.columns if col.startswith('year_')]
            if time_series_cols:
                time_series_features = df[time_series_cols]  # Corrected typo
                df['KML_Cluster'] = kml_clustering(time_series_features, n_clusters=3)
                X = pd.concat([static_features, pd.get_dummies(df['KML_Cluster'], prefix='Cluster')], axis=1)
            else:
                X = static_features.copy()

            y = df['Label'] - 1  # 转换为0-based索引

            # 6. 数据标准化
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # 7. 数据集划分
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y,
                test_size=0.3,
                random_state=42,
                stratify=y
            )

            # 8. 模型训练
            lr = LogisticRegression(
                multi_class='multinomial',
                solver='lbfgs',
                max_iter=1000,
                class_weight='balanced'
            )

            lr.fit(X_train, y_train)

            # 9. 预测与评估
            y_pred = lr.predict(X_test)

            # 计算评估指标
            kappa = cohen_kappa_score(y_test, y_pred, weights='quadratic')
            mae = mean_absolute_error(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')

            # 存储结果
            result = {
                'File': file,
                'Weighted Kappa': kappa,
                'MAE': mae,
                'Weighted F1': f1,
                'Features': X.columns.tolist(),
                'Class Distribution': dict(pd.Series(y).value_counts())
            }
            all_results.append(result)

            print(f"\n文件 {file} 的评估结果：")
            print(f"加权Kappa系数: {kappa:.3f}")
            print(f"MAE: {mae:.3f}")
            print(f"加权F1分数: {f1:.3f}")

            # 特征重要性
            importance = pd.DataFrame({
                'Feature': X.columns,
                'Coefficient': lr.coef_[0]
            }).sort_values('Coefficient', ascending=False)

            print("\nTop 10重要特征：")
            print(importance.head(10))

            # 10. 可视化 - 混淆矩阵
            try:
                plt.figure(figsize=(8, 6))
                ConfusionMatrixDisplay.from_predictions(
                    y_test, y_pred,
                    cmap='Blues',
                    normalize='true',
                    display_labels=[str(i) for i in sorted(np.unique(y_test))],
                )
                plt.title(f"KML + LR")
                plt.xlabel("Predicted")
                plt.ylabel("True")
                save_path = os.path.join(output_dir, f"{os.path.splitext(file)[0]}_confusion_matrix.png")
                plt.savefig(save_path, bbox_inches='tight', dpi=150)
                plt.close()
                print(f"混淆矩阵已保存到: {save_path}")
            except Exception as e:
                print(f"保存混淆矩阵时出错: {str(e)}")

            # 保存处理后的数据
            processed_path = os.path.join(output_dir, f"processed_{file}")
            df.to_excel(processed_path, index=False)
            print(f"处理后的数据已保存到: {processed_path}")

        except Exception as e:
            print(f"处理文件 {file} 时出错: {str(e)}")
            continue

    # 保存所有结果
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(os.path.join(output_dir, "all_results.csv"), index=False)
        print("\n模型评估结果已保存到:", os.path.join(output_dir, "all_results.csv"))

    if evaluation_metrics:
        eval_df = pd.DataFrame(evaluation_metrics)
        eval_path = os.path.join(output_dir, "an_am_evaluation.xlsx")
        eval_df.to_excel(eval_path, index=False)
        print("AN/AM列评估结果已保存到:", eval_path)

    return all_results, evaluation_metrics

# 使用示例
if __name__ == "__main__":
    folder_path = "C:/Users/fangxiang/Downloads/GMM_5+5"
    output_dir = "kml"
    process_folder(folder_path, output_dir)

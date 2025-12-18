# XGBoost
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (precision_score, recall_score, roc_auc_score,
                            confusion_matrix, cohen_kappa_score, f1_score,
                            mean_absolute_error)
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from tqdm import tqdm

# 设置中文显示和图形参数
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
warnings.filterwarnings('ignore')

def plot_confusion_matrix(cm, classes, filename, title='混淆矩阵'):
    """绘制并保存混淆矩阵图"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def process_excel_files(folder_path, output_folder):
    """
    处理指定文件夹中的所有Excel文件
    使用XGBoost模型，以AE列(30)减1作为目标变量
    保持9:1训练测试分割，但绘制全部数据的混淆矩阵
    """
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 获取文件夹中所有Excel文件
    excel_files = [f for f in os.listdir(folder_path) if f.endswith(('.xlsx', '.xls'))]

    if not excel_files:
        print("指定文件夹中没有找到Excel文件")
        return

    results = []

    for file in tqdm(excel_files, desc="处理文件中"):
        file_path = os.path.join(folder_path, file)
        output_path = os.path.join(output_folder, file)
        print(f"\n正在处理文件: {file}")

        try:
            # 读取Excel文件
            df = pd.read_excel(file_path)

            # 1. 计算TNoC作为N-P列(13-15)的和
            cols_N_P = df.iloc[:, 13:16]  # 列N-P(13-15)
            df['TNoC'] = cols_N_P.sum(axis=1)

            # 2. 获取其他指定特征
            feature_mapping = {
                'HI': 7,    # H列(7)
                'JIF': 11,  # L列(11)
                'PL': 33,   # AH列(33)
                'TL': 5,    # F列(5)
                'NoA': 6,   # G列(6)
                'NoR': 10,  # K列(10)
                'ECGR': 13, # N列(13)
                'PACNCI': 35 # AJ列(35)
            }

            for name, col in feature_mapping.items():
                df[name] = df.iloc[:, col]

            # 3. 设置AE列(30)为目标变量并减1
            if df.shape[1] < 31:
                print(f"文件 {file} 没有AE列(30)，跳过...")
                continue

            df['Target'] = df.iloc[:, 30] - 1  # 从AE列创建目标列并减1

            # 检查目标变量是否有至少2个类别
            unique_classes = np.sort(df['Target'].unique())
            if len(unique_classes) < 2:
                print(f"文件 {file} 的目标列在减1后只有1个类别: {unique_classes}，跳过...")
                continue

            # 准备特征和目标
            features = df[['TNoC', 'HI', 'JIF', 'PL', 'TL', 'NoA', 'NoR', 'ECGR', "PACNCI"]]
            target = df['Target']
            feature_names = features.columns.tolist()

            # 分割数据集(90%训练，10%测试)
            X_train, X_test, y_train, y_test = train_test_split(
                features, target, test_size=0.1, random_state=42, stratify=target
            )

            # 预处理管道
            numeric_features = features.columns.tolist()
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features)
                ])

            # 训练XGBoost模型
            xgb_model = xgb.XGBClassifier(
                learning_rate=0.2,
                n_estimators=80,
                max_depth=7,
                gamma=0.001,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )

            # 创建预处理和XGBoost的管道
            xgb_pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', xgb_model)
            ])

            xgb_pipeline.fit(X_train, y_train)

            # 对整个数据集进行预测（用于绘制全部数据的混淆矩阵）
            preprocessed_data = preprocessor.transform(features)
            df['Predicted_Class'] = xgb_model.predict(preprocessed_data)

            # 如果是二分类，添加预测概率
            if len(unique_classes) == 2:
                df['Predicted_Probability'] = xgb_model.predict_proba(preprocessed_data)[:, 1]

            # =============================================
            # 评估部分（使用测试集）
            # =============================================
            preprocessed_test = preprocessor.transform(X_test)
            y_pred_test = xgb_model.predict(preprocessed_test)
            y_proba_test = xgb_model.predict_proba(preprocessed_test)

            # 计算测试集上的各项指标
            precision = precision_score(y_test, y_pred_test, average='weighted')
            recall = recall_score(y_test, y_pred_test, average='weighted')
            weighted_kappa = cohen_kappa_score(y_test, y_pred_test, weights='quadratic')
            weighted_f1 = f1_score(y_test, y_pred_test, average='weighted')
            mae = mean_absolute_error(y_test, y_pred_test)

            # AUC计算
            if len(unique_classes) == 2:
                auc = roc_auc_score(y_test, y_proba_test[:, 1])
            else:
                auc = roc_auc_score(y_test, y_proba_test, multi_class='ovo')

            # 测试集混淆矩阵
            cm_test = confusion_matrix(y_test, y_pred_test)

            # =============================================
            # 全部数据的混淆矩阵
            # =============================================
            cm_all = confusion_matrix(target, df['Predicted_Class'])

            # 获取特征重要性
            importance = xgb_model.feature_importances_
            feature_importance = dict(zip(feature_names, importance))

            print(f"\n文件 {file} 的结果:")
            print(f"测试集评估指标:")
            print(f"加权Kappa: {weighted_kappa:.3f}")
            print(f"加权F1分数: {weighted_f1:.3f}")
            print(f"MAE: {mae:.3f}")
            print(f"精确度: {precision:.3f}")
            print(f"召回率: {recall:.3f}")
            print(f"AUC: {auc:.3f}")
            print("\n特征重要性:")
            for feature, imp in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
                print(f"{feature}: {imp:.4f}")
            print("\n测试集混淆矩阵:")
            print(cm_test)
            print("\n全部数据混淆矩阵:")
            print(cm_all)

            # 绘制并保存混淆矩阵
            cm_plot_folder = os.path.join(output_folder, "Confusion_Matrix_Plots")
            os.makedirs(cm_plot_folder, exist_ok=True)

            # 保存测试集混淆矩阵
            plot_confusion_matrix(cm_test, unique_classes,
                                 os.path.join(cm_plot_folder, f"{file}_test_confusion_matrix.png"),
                                 title='测试集混淆矩阵')

            # 保存全部数据混淆矩阵
            plot_confusion_matrix(cm_all, unique_classes,
                                 os.path.join(cm_plot_folder, f"{file}_all_data_confusion_matrix.png"),
                                 title='全部数据混淆矩阵')

            # 只保留标准化后的数据
            standardized_features = pd.DataFrame(preprocessed_data,
                                               columns=[f"std_{name}" for name in feature_names])
            df = pd.concat([df, standardized_features], axis=1)

            # 存储结果
            results.append({
                '文件名': file,
                '加权Kappa': weighted_kappa,
                '加权F1分数': weighted_f1,
                'MAE': mae,
                '精确度': precision,
                '召回率': recall,
                'AUC': auc,
                '类别分布': dict(pd.Series(target).value_counts()),
                '测试集混淆矩阵': cm_test,
                '全部数据混淆矩阵': cm_all,
                '特征重要性': feature_importance
            })

            # 保存带有预测结果的新文件
            df.to_excel(output_path, index=False)
            print(f"已保存预测结果到: {output_path}")

        except Exception as e:
            print(f"处理文件 {file} 时出错: {str(e)}")
            continue

    # 保存汇总结果
    if results:
        # 将结果转换为DataFrame
        summary_df = pd.DataFrame(results)

        # 对混淆矩阵进行特殊处理，转换为字符串格式
        summary_df['测试集混淆矩阵'] = summary_df['测试集混淆矩阵'].apply(lambda x: str(x))
        summary_df['全部数据混淆矩阵'] = summary_df['全部数据混淆矩阵'].apply(lambda x: str(x))

        # 保存到Excel
        summary_path = os.path.join(output_folder, "模型评估汇总.xlsx")

        # 使用ExcelWriter调整列宽
        with pd.ExcelWriter(summary_path, engine='openpyxl') as writer:
            summary_df.to_excel(writer, index=False)

            # 获取工作表对象
            worksheet = writer.sheets['Sheet1']

            # 设置列宽
            for column in worksheet.columns:
                max_length = 0
                column = [cell for cell in column]
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = (max_length + 2)
                worksheet.column_dimensions[column[0].column_letter].width = adjusted_width

        print(f"\n已保存模型评估汇总到: {summary_path}")

    return results


# 使用示例
if __name__ == "__main__":
    input_folder = "/data"  # 修改为你的输入文件夹路径
    output_folder = "/result/XGBoost"   # 修改为你的输出文件夹路径

    # 处理所有Excel文件并保存结果
    process_excel_files(input_folder, output_folder)
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Machine Learning Analysis Script
Calculates TNoC as sum of columns 14-18
Generates normalized confusion matrices for all Excel files
Outputs weighted F1, weighted Kappa, and MAE
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score, cohen_kappa_score, mean_absolute_error
import xgboost as xgb
import os

plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 150

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Constants
TARGET_COL = 'Label'
FEATURE_COLS = ['TL', 'NoA', 'HI', 'TNoC', 'NoR', 'JIF', 'ECGR', 'PL', 'PACNCI']


def calculate_tnoc(df):
    """Calculate TNoC as sum of columns 14-18 (0-based index 13-17)"""
    cols_14_18 = df.iloc[:, 13:18] + df.iloc[:, 23:28] # Columns 14-18 (0-based 13-17)
    return cols_14_18.sum(axis=1)


def detect_subject(filename):
    """Detect subject from filename"""
    filename = filename.lower()
    if 'chem' in filename:
        return 'Chemistry'
    elif 'phys' in filename:
        return 'Physics'
    elif 'bio' in filename:
        return 'Biology'
    elif 'econ' in filename:
        return 'Economics'
    else:
        return os.path.splitext(filename)[0].capitalize()


def load_and_preprocess(file_path):
    """Load and preprocess data with TNoC calculation"""
    try:
        df = pd.read_excel(file_path)

        # Calculate TNoC as sum of columns 14-18
        df['TNoC'] = calculate_tnoc(df)

        # Column name mapping
        column_mapping = {
            '标题长度': 'TL',
            '作者数量': 'NoA',
            '作者h指数': 'HI',
            '参考文献数量': 'NoR',
            '五年影响因子': 'JIF',
            'year_1': 'ECGR',
            '篇幅': 'PL',
            '十年CNCI': 'PACNCI',
            '新标签': 'Label'
        }
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

        # Validate required columns
        missing_features = [col for col in FEATURE_COLS if col not in df.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        if TARGET_COL not in df.columns:
            raise ValueError(f"Target column '{TARGET_COL}' not found")

        # Handle missing values
        df[FEATURE_COLS] = df[FEATURE_COLS].fillna(df[FEATURE_COLS].mean())
        df[TARGET_COL] = df[TARGET_COL].fillna(df[TARGET_COL].mode()[0])

        # Separate features and target
        X = df[FEATURE_COLS]
        y = df[TARGET_COL]

        # Standardize features
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

        return X_scaled, y

    except Exception as e:
        print(f"Data processing error: {str(e)}")
        raise


def plot_normalized_confusion_matrix(y_true, y_pred, class_labels, output_path, subject):
    """Plot and save normalized confusion matrix with class labels adjusted by -1"""
    cm = confusion_matrix(y_true, y_pred, normalize='true')

    # Adjust class labels by subtracting 1
    adjusted_labels = [str(int(label) - 1) for label in class_labels]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                cbar=True, annot_kws={"size": 12}, vmin=0, vmax=1)

    plt.title(f'XGBoost + LR', fontsize=16)
    plt.xlabel('Predicted', fontsize=14)
    plt.ylabel('True', fontsize=14)

    # Use adjusted labels for ticks
    plt.xticks(np.arange(len(adjusted_labels)) + 0.5, adjusted_labels)
    plt.yticks(np.arange(len(adjusted_labels)) + 0.5, adjusted_labels)

    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to: {output_path}")


def calculate_metrics(y_true, y_pred, output_path, subject, filename):
    """Calculate and save evaluation metrics to Excel with additional info"""
    metrics = {
        'Subject': subject,
        'Filename': filename,
        'Weighted F1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'Weighted Kappa': cohen_kappa_score(y_true, y_pred, weights='quadratic'),
        'MAE': mean_absolute_error(y_true, y_pred)
    }

    # Create DataFrame from metrics
    metrics_df = pd.DataFrame([metrics])

    # Check if summary file exists
    summary_path = os.path.join(os.path.dirname(output_path), 'summary_metrics.xlsx')
    if os.path.exists(summary_path):
        # Append to existing file
        existing_df = pd.read_excel(summary_path)
        updated_df = pd.concat([existing_df, metrics_df], ignore_index=True)
    else:
        # Create new file
        updated_df = metrics_df

    # Save updated summary
    updated_df.to_excel(summary_path, index=False)
    print(f"Metrics saved to summary file: {summary_path}")

    # Also save individual metrics (optional)
    metrics_df.to_excel(output_path, index=False)
    print(f"Individual metrics saved to: {output_path}")

    return metrics


def process_file(file_path, output_dir):
    """Process a single Excel file"""
    try:
        filename = os.path.basename(file_path)
        subject = detect_subject(filename)
        print(f"\nProcessing {subject} data from: {file_path}")

        subject_dir = os.path.join(output_dir, subject)
        os.makedirs(subject_dir, exist_ok=True)

        X, y = load_and_preprocess(file_path)
        y_adjusted = y - 1 if y.min() > 0 else y

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_adjusted, test_size=0.2, random_state=RANDOM_SEED, stratify=y_adjusted)

        model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=len(np.unique(y_adjusted)),
            random_state=RANDOM_SEED,
            n_estimators=100
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X)
        y_pred_original = y_pred + 1 if y.min() > 0 else y_pred
        y_original = y

        class_labels = sorted(np.unique(np.concatenate([y_original, y_pred_original])))

        conf_matrix_path = os.path.join("XGBLR", f'{subject}normalized_confusion_matrix.png')
        metrics_path = os.path.join(subject_dir, 'metrics.xlsx')  # Changed to .xlsx

        plot_normalized_confusion_matrix(y_original, y_pred_original, class_labels, conf_matrix_path, subject)
        metrics = calculate_metrics(y_adjusted, y_pred, metrics_path, subject, filename)

        return True

    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return False


def process_folder(input_folder, output_base):
    """Process all Excel files in a folder"""
    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"Input folder not found: {input_folder}")

    os.makedirs(output_base, exist_ok=True)

    processed_files = []
    for file in os.listdir(input_folder):
        if file.endswith(('.xlsx', '.xls')) and not file.startswith('~$'):
            file_path = os.path.join(input_folder, file)
            if process_file(file_path, output_base):
                processed_files.append(file)

    if processed_files:
        print(f"\nSuccessfully processed {len(processed_files)} files:")
        for f in processed_files:
            print(f"- {f}")
    else:
        print("\nNo valid Excel files found for processing")


if __name__ == "__main__":
    INPUT_FOLDER = "C:/Users/fangxiang/Downloads/GMM_5+5"  # Update with your path
    OUTPUT_FOLDER = "XGBLR_res"  # Update with your path

    print("=== Starting analysis of all Excel files ===")
    process_folder(INPUT_FOLDER, OUTPUT_FOLDER)
    print("\n=== Analysis complete ===")
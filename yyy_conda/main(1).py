#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data mining and machine learning analysis script
Using XGBoost and SHAP to analyze citation data
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

# Ignore warnings
warnings.filterwarnings('ignore')

# Set matplotlib global parameters
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'sans-serif']  # English font settings
plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display
plt.rcParams['xtick.direction'] = 'out'  # Ticks outward
plt.rcParams['ytick.direction'] = 'out'  # Ticks outward
plt.rcParams['mathtext.default'] = 'regular'  # Math text uses regular font
plt.rcParams['savefig.dpi'] = 150  # Save figure DPI
plt.rcParams['figure.dpi'] = 150  # Display figure DPI

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Target variable column name
TARGET_COL = 'Label'  # New target column name

# Column name mapping dictionary (Chinese → English abbreviations)
COLUMN_MAPPING = {
    '标题长度': 'TL',
    '作者数量': 'NoA',
    '作者h指数': 'HI',
    '预测总引用量': 'TNoC',
    '参考文献数量': 'NoR',
    '五年影响因子': 'JIF',
    'year_1': 'ECGR',
    '篇幅': 'PL',
    '十年CNCI': 'PACNCI',
    '标签': 'Label'
}

# Features to use (explicitly defined)
FEATURE_COLS = ['TL', 'NoA', 'HI', 'TNoC', 'NoR', 'JIF', 'ECGR', 'PL', 'PACNCI']

# Create output directory
if not os.path.exists('output'):
    os.makedirs('output')


def load_data(file_path):
    """
    Load Excel data file and calculate total predicted citations (sum of columns 14-18 and 24-28)

    Parameters:
        file_path: Excel file path

    Returns:
        Loaded and processed DataFrame
    """
    print(f"Loading data: {file_path}")
    try:
        data = pd.read_excel(file_path)
        print(f"Data loaded successfully: {data.shape[0]} rows, {data.shape[1]} columns")

        # Verify if there are enough columns (at least 28)
        if data.shape[1] < 28:
            raise ValueError(
                f"Data file has only {data.shape[1]} columns, needs at least 28 to calculate total citations")

        # Calculate total citations as sum of columns 14-18 (index 13-17) and 24-28 (index 23-27)
        # First convert relevant columns to numeric, handling non-numeric data
        citation_cols = list(range(13, 18)) + list(range(23, 28))
        for col in citation_cols:
            data.iloc[:, col] = pd.to_numeric(data.iloc[:, col], errors='coerce').fillna(0)

        # Calculate sum and create "TNoC" column
        data['TNoC'] = data.iloc[:, 13:18].sum(axis=1) + data.iloc[:, 23:28].sum(axis=1)

        # Rename columns using mapping
        data.rename(columns=COLUMN_MAPPING, inplace=True)

        # Print column names for verification
        print("\nColumn names in dataset:")
        print(data.columns.tolist())

        # Check data types
        print("\nData types:")
        print(data.dtypes)

        return data
    except Exception as e:
        print(f"Data loading failed: {str(e)}")
        raise


def explore_data(df):
    """
    Exploratory data analysis

    Parameters:
        df: DataFrame with data
    """
    print("\n=== Data Exploration ===")

    # View basic info
    print("\nBasic info:")
    print(df.info())

    # View statistical summary
    print("\nStatistical summary:")
    print(df.describe())

    # Check for missing values
    missing_values = df.isnull().sum()
    print("\nMissing values:")
    print(missing_values[missing_values > 0])

    # Check existence of feature columns
    missing_features = [col for col in FEATURE_COLS if col not in df.columns]
    if missing_features:
        print(f"\nWarning: Missing required feature columns: {missing_features}")

    # Target variable distribution
    if TARGET_COL in df.columns:
        print("\nTarget variable distribution:")
        print(df[TARGET_COL].value_counts())

        # Plot target variable distribution
        plt.figure(figsize=(8, 6))
        df[TARGET_COL].value_counts().sort_index().plot(kind='bar')
        plt.title('Distribution of Citations Quartile')
        plt.xlabel('Label')
        plt.ylabel('Number')
        plt.xticks(rotation=0)
        plt.savefig('output/target_distribution.png')
        plt.close()
    else:
        print(f"\nWarning: Target column '{TARGET_COL}' not found in dataset")


def preprocess_data(df):
    """
    Data preprocessing: handle missing values, standardization, etc.

    Parameters:
        df: Raw data DataFrame

    Returns:
        Processed DataFrame, features X, target y
    """
    print("\n=== Data Preprocessing ===")

    # Copy data to avoid modifying original
    df_processed = df.copy()

    # Verify all required columns exist
    missing_features = [col for col in FEATURE_COLS if col not in df_processed.columns]
    if missing_features:
        raise ValueError(f"Missing required feature columns: {missing_features}")

    if TARGET_COL not in df_processed.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset")

    # Show data sample
    print("\nFirst 5 rows sample:")
    print(df_processed[FEATURE_COLS + [TARGET_COL]].head())

    # Check and handle missing values
    missing_values = df_processed[FEATURE_COLS + [TARGET_COL]].isnull().sum()
    if missing_values.sum() > 0:
        print(f"\nFound {missing_values.sum()} missing values")
        # Fill numeric features with mean, categorical with mode
        for col in FEATURE_COLS:
            if df_processed[col].isnull().sum() > 0:
                if pd.api.types.is_numeric_dtype(df_processed[col]):
                    df_processed[col].fillna(df_processed[col].mean(), inplace=True)
                    print(f"Filled {df_processed[col].isnull().sum()} missing values in '{col}' with mean")
                else:
                    df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
                    print(f"Filled {df_processed[col].isnull().sum()} missing values in '{col}' with mode")

        # Handle missing values in target column (if any)
        if df_processed[TARGET_COL].isnull().sum() > 0:
            df_processed[TARGET_COL].fillna(df_processed[TARGET_COL].mode()[0], inplace=True)
            print(f"Filled {df_processed[TARGET_COL].isnull().sum()} missing values in '{TARGET_COL}' with mode")

        print("Missing values handled")
    else:
        print("\nNo missing values in selected columns")

    # Check target variable distribution
    target_counts = df_processed[TARGET_COL].value_counts().sort_index()
    print(f"\nTarget variable distribution:")
    print(target_counts)

    # Plot target distribution
    plt.figure(figsize=(8, 6))
    target_counts.plot(kind='bar')
    plt.title('Distribution of Target Variables')
    plt.xlabel('Label')
    plt.ylabel('Number')
    plt.savefig('output/target_distribution.png')
    plt.close()

    # Separate features and target - only use specified feature columns
    X = df_processed[FEATURE_COLS]
    y = df_processed[TARGET_COL]

    print(f"\nFeature dimensions: {X.shape}")
    print(f"Selected features: {X.columns.tolist()}")

    # Convert any string columns to numeric (if possible)
    for col in X.columns:
        if pd.api.types.is_object_dtype(X[col]):  # Changed from X[col].dtype == 'object'
            try:
                X[col] = pd.to_numeric(X[col])
                print(f"Converted column '{col}' to numeric")
            except ValueError:
                print(f"Warning: Column '{col}' contains non-numeric data, cannot standardize")
                # Consider one-hot encoding for categorical variables

    # Standardize numeric features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    print("\nData preprocessing complete")

    return df_processed, X_scaled, y


def split_data(X, y):
    """
    Split data into training and test sets

    Parameters:
        X: Feature data
        y: Target variable

    Returns:
        Training and test sets
    """
    print("\n=== Data Splitting ===")

    # Use stratified sampling to maintain class proportions
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")

    # Verify target distribution in each set
    print("Training set target distribution:")
    print(y_train.value_counts(normalize=True))
    print("Test set target distribution:")
    print(y_test.value_counts(normalize=True))

    return X_train, X_test, y_train, y_test


def train_xgboost_model(X_train, y_train, X_test, y_test):
    """
    Train XGBoost classification model

    Parameters:
        X_train, y_train: Training data
        X_test, y_test: Test data

    Returns:
        Trained model
    """
    print("\n=== XGBoost Model Training ===")

    # Check unique classes in target
    unique_classes = np.sort(y_train.unique())
    print(f"Unique classes in target variable: {unique_classes}")

    # Convert to 0-based indexing
    y_train_0indexed = y_train - 1
    y_test_0indexed = y_test - 1

    print(f"Adjusted unique classes: {np.sort(y_train_0indexed.unique())}")

    # Convert DataFrames to numpy arrays to avoid dtype issues
    X_train_array = X_train.values
    X_test_array = X_test.values

    # Initialize model with improved parameters
    best_model = xgb.XGBClassifier(
        objective='multi:softprob',  # Multiclass problem
        num_class=4,  # 4 classes
        random_state=RANDOM_SEED,
        n_estimators=300,  # Increased number of trees
        learning_rate=0.01,  # Lower learning rate
        enable_categorical=False  # Explicitly disable categorical features
    )

    # Use 5-fold cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    # Simplified parameter grid for initial testing
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
        verbose=1,
        error_score='raise'  # Raise errors to debug
    )

    print("Starting grid search for optimal parameters...")
    try:
        grid_search.fit(X_train_array, y_train_0indexed)
    except Exception as e:
        print(f"Error during grid search: {str(e)}")
        raise

    # Output best parameters
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

    # Get best model
    best_model = grid_search.best_estimator_

    # Evaluate on training set
    y_train_pred = best_model.predict(X_train_array)
    print("\nTraining set performance:")
    print(f"Accuracy: {accuracy_score(y_train_0indexed, y_train_pred):.4f}")

    # Evaluate on test set
    y_test_pred = best_model.predict(X_test_array)
    print("\nTest set performance:")
    print(f"Accuracy: {accuracy_score(y_test_0indexed, y_test_pred):.4f}")

    return best_model, y_train_0indexed, y_test_0indexed


def evaluate_model(model, X_test, y_test_original, y_test_0indexed):
    """
    Comprehensive model performance evaluation

    Parameters:
        model: Trained model
        X_test: Test data features
        y_test_original: Original test labels (1-4)
        y_test_0indexed: Adjusted test labels (0-3)
    """
    print("\n=== Model Evaluation ===")

    # Convert DataFrame to numpy array for prediction
    X_test_array = X_test.values if hasattr(X_test, 'values') else X_test

    # Predict test set (predictions will be 0-based)
    y_pred_0indexed = model.predict(X_test_array)

    # Convert predictions back to original scale (1-4) for better interpretation
    y_pred_original = y_pred_0indexed + 1

    print(f"Prediction range: {np.min(y_pred_original)} to {np.max(y_pred_original)}")

    # Calculate and output classification report
    # Use 0-based to maintain consistency with model's internal representation
    print("\nClassification report (0-based labels):")
    report_0indexed = classification_report(y_test_0indexed, y_pred_0indexed)
    print(report_0indexed)

    # Also provide report with original labels for interpretation
    print("\nClassification report (original 1-based labels):")
    report_original = classification_report(y_test_original, y_pred_original)
    print(report_original)

    # Calculate main metrics (use 0-based for consistency)
    accuracy = accuracy_score(y_test_0indexed, y_pred_0indexed)
    precision = precision_score(y_test_0indexed, y_pred_0indexed, average='macro')
    recall = recall_score(y_test_0indexed, y_pred_0indexed, average='macro')
    f1 = f1_score(y_test_0indexed, y_pred_0indexed, average='macro')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Save performance metrics to file
    performance = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
        'Value': [accuracy, precision, recall, f1]
    })
    performance.to_csv('output/model_performance.csv', index=False)

    return performance, y_pred_original


def perform_shap_analysis(model, X_test, feature_names, y_pred_original):
    """
    Model interpretation analysis using SHAP library
    Generate waterfall plots, force plots, bar plots, beeswarm plots, and dependence plots
    """
    print("\n=== SHAP Analysis ===")

    # Convert X_test to numpy array if it's a DataFrame
    X_test_array = X_test.values if hasattr(X_test, 'values') else X_test

    # Get original data (unstandardized) for dependence plots
    df_processed = pd.read_excel("C:/Users/fangxiang/Downloads/GMM_processed/chemistry.xlsx")
    df_processed.rename(columns=COLUMN_MAPPING, inplace=True)
    X_original = df_processed[feature_names].iloc[-len(X_test_array):]

    # Ensure feature_names matches number of features
    if len(feature_names) != X_test_array.shape[1]:
        print(f"Warning: Feature names length mismatch")
        feature_names = feature_names[:X_test_array.shape[1]]

    explainer = shap.TreeExplainer(model)

    print("Calculating SHAP values...")
    try:
        shap_values = explainer.shap_values(X_test_array)
    except Exception as e:
        print(f"Error calculating SHAP values: {str(e)}")
        return

    print(f"SHAP values type: {type(shap_values)}")
    print(f"SHAP values shape: {np.array(shap_values).shape}")

    try:
        # Select representative samples for each class
        class_samples = {}
        for class_idx in range(4):
            class_indices = np.where(y_pred_original == class_idx)[0]
            if len(class_indices) > 0:
                class_samples[class_idx] = class_indices[0]
                print(f"Representative sample index for class {class_idx}: {class_samples[class_idx]}")

        # 1. Generate beeswarm plot for each class
        print("Generating beeswarm plots...")
        if isinstance(shap_values, list) or len(np.array(shap_values).shape) == 3:
            # Handle multiclass case
            shap_values_array = np.array(shap_values)
            for class_idx in range(shap_values_array.shape[0]):
                plt.figure(figsize=(10, 8))
                shap.summary_plot(shap_values_array[class_idx], X_test_array,
                                  feature_names=feature_names, show=False)
                plt.title(f'SHAP Beeswarm Plot - Class {class_idx}')
                plt.tight_layout()
                fix_negative_signs(plt.gcf())
                plt.savefig(f'output/shap_beeswarm_class{class_idx}.png', bbox_inches='tight')
                plt.close()
        else:
            # Handle binary case
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_test_array,
                              feature_names=feature_names, show=False)
            plt.title('SHAP Beeswarm Plot')
            plt.tight_layout()
            fix_negative_signs(plt.gcf())
            plt.savefig('output/shap_beeswarm.png', bbox_inches='tight')
            plt.close()

        # 2. Generate feature importance plot
        print("Generating feature importance plot...")
        if isinstance(shap_values, list) or len(np.array(shap_values).shape) == 3:
            # Multiclass case - average absolute SHAP values across classes and samples
            shap_importance = np.abs(np.array(shap_values)).mean(axis=(0, 1))
        else:
            # Binary case - average absolute SHAP values across samples
            shap_importance = np.abs(shap_values).mean(axis=0)

        # Create and plot importance
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': shap_importance
        }).sort_values('Importance', ascending=False)

        plt.figure(figsize=(10, 6))
        importance_df.plot(kind='barh', x='Feature', y='Importance', legend=False)
        plt.xlabel('Mean |SHAP Value|')
        plt.title('Feature Importance')
        plt.tight_layout()
        fix_negative_signs(plt.gcf())
        plt.savefig('output/shap_feature_importance.png')
        plt.close()

        # Get top 3 features
        top_features = importance_df['Feature'].head(3).tolist()

        # 3. Generate waterfall plot for each class
        print("Generating waterfall plots...")
        if isinstance(shap_values, list) or len(np.array(shap_values).shape) == 3:
            shap_values_array = np.array(shap_values)
            for class_idx in range(shap_values_array.shape[0]):
                if class_idx in class_samples:
                    sample_idx = class_samples[class_idx]
                    try:
                        plt.figure(figsize=(10, 6))
                        if isinstance(explainer.expected_value, list):
                            base_value = explainer.expected_value[class_idx]
                        else:
                            base_value = explainer.expected_value

                        shap.plots.waterfall(
                            shap.Explanation(
                                values=shap_values_array[class_idx][sample_idx],
                                base_values=base_value,
                                data=X_test_array[sample_idx],
                                feature_names=feature_names
                            ),
                            max_display=10, show=False
                        )
                        plt.title(f'SHAP Waterfall Plot - Class {class_idx}')
                        plt.tight_layout()
                        fix_negative_signs(plt.gcf())
                        plt.savefig(f'output/shap_waterfall_class{class_idx}.png', bbox_inches='tight')
                        plt.close()
                    except Exception as e:
                        print(f"Error generating waterfall plot for class {class_idx}: {str(e)}")

        # 4. Generate force plot for each class
        print("Generating force plots...")
        if isinstance(shap_values, list) or len(np.array(shap_values).shape) == 3:
            shap_values_array = np.array(shap_values)
            for class_idx in range(shap_values_array.shape[0]):
                if class_idx in class_samples:
                    sample_idx = class_samples[class_idx]
                    try:
                        plt.figure(figsize=(12, 3))
                        if isinstance(explainer.expected_value, list):
                            base_value = explainer.expected_value[class_idx]
                        else:
                            base_value = explainer.expected_value

                        shap.plots.force(
                            base_value,
                            shap_values_array[class_idx][sample_idx],
                            X_test_array[sample_idx],
                            feature_names=feature_names,
                            matplotlib=True,
                            show=False,
                            text_rotation=0
                        )
                        plt.title(f'SHAP Force Plot - Class {class_idx}')
                        plt.tight_layout()
                        fix_negative_signs(plt.gcf())
                        plt.savefig(f'output/shap_force_plot_class{class_idx}.png', bbox_inches='tight', dpi=150)
                        plt.close()
                    except Exception as e:
                        print(f"Error generating force plot for class {class_idx}: {str(e)}")

        # 5. Dependence plots - using original feature values
        print("Generating dependence plots...")
        for feature in top_features:
            try:
                feature_idx = feature_names.index(feature)
                if isinstance(shap_values, list) or len(np.array(shap_values).shape) == 3:
                    shap_values_array = np.array(shap_values)
                    for class_idx in range(shap_values_array.shape[0]):
                        plt.figure(figsize=(8, 6))
                        plt.scatter(X_original[feature].values, shap_values_array[class_idx][:, feature_idx],
                                    alpha=0.5, s=30, c=shap_values_array[class_idx][:, feature_idx], cmap='coolwarm')
                        plt.colorbar(label='SHAP Value')
                        plt.xlabel(feature + ' (Original Value)')
                        plt.ylabel('SHAP Value')
                        plt.title(f'SHAP Dependence Plot - {feature} - Class {class_idx}')
                        plt.tight_layout()
                        fix_negative_signs(plt.gcf())
                        plt.savefig(f'output/shap_dependence_{feature}_class{class_idx}.png')
                        plt.close()
                else:
                    plt.figure(figsize=(8, 6))
                    plt.scatter(X_original[feature].values, shap_values[:, feature_idx],
                                alpha=0.5, s=30, c=shap_values[:, feature_idx], cmap='coolwarm')
                    plt.colorbar(label='SHAP Value')
                    plt.xlabel(feature + ' (Original Value)')
                    plt.ylabel('SHAP Value')
                    plt.title(f'SHAP Dependence Plot - {feature}')
                    plt.tight_layout()
                    fix_negative_signs(plt.gcf())
                    plt.savefig(f'output/shap_dependence_{feature}.png')
                    plt.close()
            except Exception as e:
                print(f"Error generating dependence plot for {feature}: {str(e)}")
                traceback.print_exc()

    except Exception as e:
        print(f"Error in SHAP analysis: {str(e)}")
        traceback.print_exc()


def fix_negative_signs(fig):
    """Fix minus sign display issues in matplotlib figures"""
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

    if fig._suptitle:
        fig._suptitle.set_text(fig._suptitle.get_text().replace('\u2212', '-'))

    return fig

def main():
    """Main function"""
    print("=== Citation Data Analysis using XGBoost and SHAP ===\n")

    # 1. Load data
    file_path = "C:/Users/fangxiang/Downloads/GMM_processed/chemistry.xlsx"
    df = load_data(file_path)

    # 2. Data preprocessing
    df_processed, X_scaled, y = preprocess_data(df)

    # 3. Split into training and test sets
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)

    # 4. Train XGBoost model
    model, y_train_0indexed, y_test_0indexed = train_xgboost_model(X_train, y_train, X_test, y_test)

    # 5. Evaluate model
    performance, y_pred_original = evaluate_model(model, X_test, y_test, y_test_0indexed)

    # 6. SHAP analysis - convert to numpy array first
    feature_names = X_scaled.columns.tolist()
    X_test_array = X_test.values if hasattr(X_test, 'values') else X_test
    perform_shap_analysis(model, X_test_array, feature_names, y_pred_original)

    print("\n=== Analysis Complete! ===")
    print(f"All results saved to 'output' folder")

if __name__ == "__main__":
    main()
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             f1_score, precision_score, recall_score, cohen_kappa_score,
                             mean_absolute_error, ConfusionMatrixDisplay)
import xgboost as xgb
import shap
import warnings
import traceback

# Suppress warnings
warnings.filterwarnings('ignore')

# Global settings
plt.style.use('seaborn')
plt.rcParams.update({
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'axes.unicode_minus': False,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'figure.dpi': 150,
    'savefig.dpi': 300
})

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Constants
TARGET_COL = 'Label'
FEATURE_COLS = ['TL', 'NoA', 'HI', 'TNoC', 'NoR', 'JIF', 'ECGR', 'PL', 'PACNCI']


def create_output_structure(base_path, subject):
    """Create organized output directory structure"""
    paths = {
        'main': os.path.join(base_path, subject),
        'figures': os.path.join(base_path, subject, 'figures'),
        'tables': os.path.join(base_path, subject, 'tables'),
        'models': os.path.join(base_path, subject, 'models')
    }

    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    return paths


def detect_subject(file_path):
    """Detect subject from filename or content"""
    filename = os.path.basename(file_path).lower()

    # Common subject patterns
    subjects = {
        'chem': 'Chemistry',
        'phys': 'Physics',
        'bio': 'Biology',
        'econ': 'Economics',
        'math': 'Mathematics',
        'eng': 'Engineering'
    }

    for key, subject in subjects.items():
        if key in filename:
            return subject

    # Fallback to filename stem
    return os.path.splitext(filename)[0].capitalize()


def load_and_preprocess(file_path):
    """Load and preprocess data with comprehensive checks"""
    df = pd.read_excel(file_path)

    # Calculate TNoC (sum of columns 14-18 and 24-28)
    cols_14_18 = df.iloc[:, 13:18].apply(pd.to_numeric, errors='coerce').fillna(0)
    cols_24_28 = df.iloc[:, 23:28].apply(pd.to_numeric, errors='coerce').fillna(0)
    df['TNoC'] = cols_14_18.sum(axis=1) + cols_24_28.sum(axis=1)

    # Standardize column names
    column_mapping = {
        'Title Length': 'TL',
        'Number of Authors': 'NoA',
        'H-Index': 'HI',
        'References': 'NoR',
        'Journal Impact Factor': 'JIF',
        'Early Citation Growth Rate': 'ECGR',
        'Page Length': 'PL',
        '10-Year CNCI': 'PACNCI',
        'Label': 'Label'
    }

    return df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})


def analyze_data_quality(df, output_paths, subject):
    """Perform comprehensive EDA"""
    # Basic info
    with open(os.path.join(output_paths['tables'], f'{subject}_data_quality.txt'), 'w') as f:
        f.write(f"=== {subject} Data Quality Report ===\n\n")
        f.write("Basic Information:\n")
        df.info(buf=f)
        f.write("\n\nMissing Values:\n")
        f.write(df.isnull().sum().to_string())
        f.write("\n\nDescriptive Statistics:\n")
        f.write(df.describe().to_string())

    # Target distribution plot
    plt.figure(figsize=(10, 6))
    df[TARGET_COL].value_counts().sort_index().plot(kind='bar')
    plt.title(f'{subject} - Target Distribution')
    plt.savefig(os.path.join(output_paths['figures'], f'{subject}_target_distribution.png'))
    plt.close()


def train_and_evaluate(X_train, X_test, y_train, y_test, output_paths, subject):
    """Complete modeling workflow"""
    # Adjust labels to 0-based
    y_train_adj = y_train - 1
    y_test_adj = y_test - 1

    # Model training with hyperparameter tuning
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=len(np.unique(y_train)),
        random_state=RANDOM_SEED,
        n_estimators=300,
        learning_rate=0.01
    )

    param_grid = {
        'max_depth': [5, 7, 9],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train_adj)
    best_model = grid_search.best_estimator_

    # Save model
    best_model.save_model(os.path.join(output_paths['models'], f'{subject}_model.json'))

    # Evaluation
    y_pred_adj = best_model.predict(X_test)
    y_pred = y_pred_adj + 1  # Convert back to original labels

    # Comprehensive metrics
    metrics = {
        'accuracy': accuracy_score(y_test_adj, y_pred_adj),
        'precision_macro': precision_score(y_test_adj, y_pred_adj, average='macro'),
        'recall_macro': recall_score(y_test_adj, y_pred_adj, average='macro'),
        'f1_macro': f1_score(y_test_adj, y_pred_adj, average='macro'),
        'f1_weighted': f1_score(y_test_adj, y_pred_adj, average='weighted'),
        'kappa_weighted': cohen_kappa_score(y_test, y_pred, weights='quadratic'),
        'mae': mean_absolute_error(y_test, y_pred)
    }

    # Save metrics
    pd.DataFrame.from_dict(metrics, orient='index', columns=['Value']).to_csv(
        os.path.join(output_paths['tables'], f'{subject}_metrics.csv'))

    # Confusion matrix
    plt.figure(figsize=(10, 8))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        cmap='Blues',
        normalize='true',
        display_labels=np.unique(np.concatenate([y_test, y_pred]))
    )
    plt.title(f'{subject} - Normalized Confusion Matrix')
    plt.savefig(os.path.join(output_paths['figures'], f'{subject}_confusion_matrix.png'))
    plt.close()

    # Feature importance
    plt.figure(figsize=(12, 6))
    xgb.plot_importance(best_model, max_num_features=15)
    plt.title(f'{subject} - Feature Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(output_paths['figures'], f'{subject}_feature_importance.png'))
    plt.close()

    # SHAP analysis
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_test)

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, feature_names=X_train.columns, show=False)
    plt.title(f'{subject} - SHAP Summary Plot')
    plt.savefig(os.path.join(output_paths['figures'], f'{subject}_shap_summary.png'), bbox_inches='tight')
    plt.close()

    return best_model, metrics


def process_subject_file(file_path, output_base):
    """Complete processing pipeline for a single subject file"""
    subject = detect_subject(file_path)
    print(f"\n=== Processing {subject} ===")

    try:
        # Setup directory structure
        output_paths = create_output_structure(output_base, subject)

        # Load and preprocess
        df = load_and_preprocess(file_path)

        # Data quality analysis
        analyze_data_quality(df, output_paths, subject)

        # Prepare features and target
        X = df[FEATURE_COLS].apply(pd.to_numeric, errors='coerce').fillna(0)
        y = df[TARGET_COL]

        # Standardize features
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)

        # Model training and evaluation
        model, metrics = train_and_evaluate(X_train, X_test, y_train, y_test, output_paths, subject)

        # Save processed data
        df.to_excel(os.path.join(output_paths['tables'], f'{subject}_processed.xlsx'), index=False)

        print(f"Successfully processed {subject}")
        return True

    except Exception as e:
        print(f"Error processing {subject}: {str(e)}")
        traceback.print_exc()
        return False


def process_all_subjects(input_folder, output_base):
    """Process all Excel files in input folder"""
    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"Input folder not found: {input_folder}")

    # Create summary directory
    summary_dir = os.path.join(output_base, 'Summary')
    os.makedirs(summary_dir, exist_ok=True)

    all_metrics = []
    processed_files = []

    for file in os.listdir(input_folder):
        if file.endswith(('.xlsx', '.xls')) and not file.startswith('~$'):
            file_path = os.path.join(input_folder, file)
            success = process_subject_file(file_path, output_base)

            if success:
                subject = detect_subject(file_path)
                metrics_path = os.path.join(output_base, subject, 'tables', f'{subject}_metrics.csv')

                if os.path.exists(metrics_path):
                    metrics = pd.read_csv(metrics_path, index_col=0)
                    metrics['Subject'] = subject
                    all_metrics.append(metrics)

                processed_files.append(file)

    # Save combined metrics
    if all_metrics:
        combined_metrics = pd.concat(all_metrics)
        combined_metrics.to_excel(os.path.join(summary_dir, 'all_subjects_metrics.xlsx'))
        print("\nSaved combined metrics for all subjects")

    print(f"\nProcessing complete. Successfully processed {len(processed_files)} files:")
    for f in processed_files:
        print(f"- {f}")


if __name__ == "__main__":
    # Configuration
    INPUT_FOLDER = "/gmm"  # Update this path
    OUTPUT_BASE = "/result/XgBoost"  # Update this path

    # Run processing
    process_all_subjects(INPUT_FOLDER, OUTPUT_BASE)
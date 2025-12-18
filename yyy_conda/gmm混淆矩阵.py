import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, cohen_kappa_score, f1_score, mean_absolute_error


def plot_standardized_confusion_matrix(true_labels, pred_labels, filename, output_folder):
    """
    Plot a standardized confusion matrix with labels reduced by 1 and save as PNG.
    """
    # Reduce labels by 1 (1,2,3,4 -> 0,1,2,3)
    true_labels = np.array(true_labels) - 1
    pred_labels = np.array(pred_labels) - 1

    # Compute confusion matrix
    cm = confusion_matrix(true_labels, pred_labels, labels=[0, 1, 2, 3])

    # Standardize by row (normalize to show proportions)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)  # Replace NaN with 0

    # Plot using seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=[0, 1, 2, 3], yticklabels=[0, 1, 2, 3])
    plt.title(f'GMM')
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Save plot
    output_path = os.path.join(output_folder, f'cm_{filename[:-5]}.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    return output_path


def compute_metrics(true_labels, pred_labels):
    """
    Compute weighted Cohen's Kappa, weighted F1 score, and MAE.
    """
    # Reduce labels by 1 for consistency
    true_labels = np.array(true_labels) - 1
    pred_labels = np.array(pred_labels) - 1

    # Weighted Cohen's Kappa
    kappa = cohen_kappa_score(true_labels, pred_labels, weights='quadratic')

    # Weighted F1 Score
    f1 = f1_score(true_labels, pred_labels, average='weighted')

    # Mean Absolute Error
    mae = mean_absolute_error(true_labels, pred_labels)

    return kappa, f1, mae


def process_excel_files(input_folder, output_folder, metrics_output_file):
    """
    Process Excel files to create standardized confusion matrices and compute metrics.
    Save metrics to a new Excel file.
    """
    # Create output folder for plots if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # List to store metrics for all files
    metrics_data = []

    # Iterate through all Excel files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.xlsx') or filename.endswith('.xls'):
            input_path = os.path.join(input_folder, filename)

            try:
                print(f"Processing file: {filename}")

                # Read the Excel file
                df = pd.read_excel(input_path)

                # Check if required columns exist
                if df.shape[1] < 39:
                    print(f"Warning: {filename} has only {df.shape[1]} columns, expected at least 39. Skipping.")
                    continue

                # Extract true (col 31, index 30) and predicted (col 39, index 38) labels
                true_labels = df.iloc[:, 30].dropna()
                pred_labels = df.iloc[:, 38].dropna()

                # Ensure labels are integers and align lengths
                common_indices = true_labels.index.intersection(pred_labels.index)
                true_labels = true_labels.loc[common_indices].astype(int)
                pred_labels = pred_labels.loc[common_indices].astype(int)

                if len(true_labels) == 0 or len(pred_labels) == 0:
                    print(f"Warning: No valid labels in {filename}. Skipping.")
                    continue

                # Plot standardized confusion matrix
                cm_path = plot_standardized_confusion_matrix(true_labels, pred_labels, filename, output_folder)
                print(f"Saved confusion matrix: {cm_path}")

                # Compute metrics
                kappa, f1, mae = compute_metrics(true_labels, pred_labels)

                # Store metrics
                metrics_data.append({
                    'File': filename,
                    'Weighted Kappa': kappa,
                    'Weighted F1': f1,
                    'MAE': mae
                })

            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

    # Save metrics to Excel
    if metrics_data:
        metrics_df = pd.DataFrame(metrics_data)
        metrics_output_path = os.path.join(output_folder, metrics_output_file)
        metrics_df.to_excel(metrics_output_path, index=False)
        print(f"Saved metrics to: {metrics_output_path}")
    else:
        print("No metrics computed due to errors or empty data.")

    print("\nAll files processed!")


if __name__ == "__main__":
    # Specify input and output folders
    input_folder = "C:/Users/fangxiang/Downloads/GMM_processed"  # Replace with your input folder path
    output_folder = "confusion_matrices"  # Folder for confusion matrix plots
    metrics_output_file = "metrics_summary.xlsx"  # Output file for metrics

    process_excel_files(input_folder, output_folder, metrics_output_file)
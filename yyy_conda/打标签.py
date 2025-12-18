import os
import pandas as pd
import numpy as np


def process_excel_files(input_folder, output_folder):
    """
    Process Excel files in the input folder:
    - Compute sum of columns 14-18 and 24-28, store in column 40.
    - Label column 40 based on quartiles (top 25%: 1, 26-50%: 2, 51-75%: 3, rest: 4), store in column 39.
    - Save modified files to output folder.
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through all Excel files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.xlsx') or filename.endswith('.xls'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            try:
                print(f"Processing file: {filename}")

                # Read the Excel file
                df = pd.read_excel(input_path)

                # Ensure the DataFrame has enough columns
                if df.shape[1] < 40:
                    # Add missing columns with NaN if necessary
                    for i in range(df.shape[1], 40):
                        df[f'Column_{i + 1}'] = np.nan

                # Ensure columns 14-18 and 24-28 are numeric
                for col in list(range(13, 18)) + list(range(23, 28)):
                    df.iloc[:, col] = pd.to_numeric(df.iloc[:, col], errors='coerce').fillna(0)

                # Calculate sum of columns 14-18 (index 13-17) and 24-28 (index 23-27)
                sum_cols = df.iloc[:, 13:18].sum(axis=1) + df.iloc[:, 23:28].sum(axis=1)

                # Store sum in column 40 (index 39)
                df.iloc[:, 39] = sum_cols

                # Calculate quartiles for column 40
                quartiles = np.percentile(sum_cols, [25, 50, 75])
                q1, q2, q3 = quartiles

                # Assign labels based on quartiles
                labels = pd.Series(4, index=df.index)  # Default label is 4
                labels[sum_cols > q3] = 1  # Top 25%
                labels[(sum_cols > q2) & (sum_cols <= q3)] = 2  # 26-50%
                labels[(sum_cols > q1) & (sum_cols <= q2)] = 3  # 51-75%

                # Store labels in column 39 (index 38)
                df.iloc[:, 38] = labels

                # Save the modified DataFrame to the output folder
                df.to_excel(output_path, index=False)
                print(f"Successfully processed and saved: {output_path}")

            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

    print("\nAll files processed!")


if __name__ == "__main__":
    # Specify input and output folders
    input_folder = "C:/Users/fangxiang/Downloads/GMM_5+5"  # Replace with your input folder path
    output_folder = "C:/Users/fangxiang/Downloads/GMM_processed"  # Replace with your output folder path

    process_excel_files(input_folder, output_folder)
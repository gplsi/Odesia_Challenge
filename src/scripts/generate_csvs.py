import os
import argparse
import pandas as pd

def parse_custom_csv(csv_path, model_folder):
    """Parse a file with lines that may have 2 or 3 columns:
       - 2 columns => subDataset, macroF1
       - 3 columns => subDataset, exactMatch, macroF1
       Returns a DataFrame with columns:
          [Dataset, SubDataset, ExactMatch, MacroF1]
    """
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # skip empty lines

            # Skip a header row if it starts with "Dataset" or "SubDataset"
            lower_line = line.lower()
            if lower_line.startswith("dataset") or lower_line.startswith("subdataset"):
                continue

            # Split by comma
            parts = line.split(',')

            if len(parts) == 2:
                # We have only MacroF1
                subDataset, macroF1 = parts
                exactMatch = ""
            elif len(parts) == 3:
                # We have ExactMatch and MacroF1
                subDataset, exactMatch, macroF1 = parts
            else:
                # If there's any weird line with more or fewer columns, skip or handle it
                print(f"Skipping malformed line in {csv_path}: {line}")
                continue

            rows.append({
                "Dataset": model_folder,
                "SubDataset": subDataset,
                "ExactMatch": exactMatch,
                "MacroF1": macroF1
            })

    df = pd.DataFrame(rows)
    return df

def generate_main_csv(folder_in, folder_out):
    dataframes = []
    
    if not os.path.exists(folder_in):
        print(f"Error: Input folder '{folder_in}' does not exist.")
        return

    model_folders = os.listdir(folder_in)
    
    for model_folder in model_folders:
        folder_path = os.path.join(folder_in, model_folder)
        if not os.path.isdir(folder_path):
            continue  # Skip if it's not a directory

        csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

        if not csv_files:
            print(f'No CSV found in {model_folder}')
            continue

        # In this example, we only handle the first CSV if multiple exist
        csv_path = os.path.join(folder_path, csv_files[0])
        try:
            df_parsed = parse_custom_csv(csv_path, model_folder)
            if not df_parsed.empty:
                dataframes.append(df_parsed)
        except Exception as e:
            print(f"Skipping {csv_path} due to parsing error: {e}")

    if dataframes:
        df_final = pd.concat(dataframes, ignore_index=True)
        os.makedirs(folder_out, exist_ok=True)
        output_path = os.path.join(folder_out, 'main.csv')
        df_final.to_csv(output_path, index=False)
        print(f"✅ Main CSV created at {output_path}")
    else:
        print("⚠️ No valid CSV data found. Nothing to concatenate.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate main CSV from 2-col or 3-col 'mixed' CSVs in subdirectories.")
    
    default_input = "workspace/data/results"
    default_output = "workspace/data/"
    
    parser.add_argument("--folder_in", default=default_input, help=f"Path to input folder (default: {default_input})")
    parser.add_argument("--folder_out", default=default_output, help=f"Path to output folder (default: {default_output})")

    args = parser.parse_args()
    generate_main_csv(args.folder_in, args.folder_out)

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 03:04:12 2025

@author: mujtabasaeed
"""

import json
import pandas as pd

# File paths
csv_path = "./nnUNet_results/summary_3dcascade.xlsx"
json_input_path = "./nnUNet_results/summary_3dcascade.json"
json_output_path = "./nnUNet_results/summary_3dcascade_final.json"

# Load the original JSON to get case paths and reference file paths
with open(json_input_path, "r") as file:
    original_data = json.load(file)

case_paths = [case["prediction_file"] for case in original_data["metric_per_case"]]
reference_paths = [case["reference_file"] for case in original_data["metric_per_case"]]

# Load CSV file from the "edited" sheet
df = pd.read_excel(csv_path, sheet_name="edited")

# Extract mean row (foreground_mean and mean["1"])
mean_row = df[df["Case"] == "Mean"].iloc[0].drop("Case").to_dict()

# Remove mean row from case data
df_cases = df[df["Case"] != "Mean"]

# Convert to required JSON format
metric_per_case = []
for i, row in df_cases.iterrows():
    case_name = row["Case"]
    metrics = row.drop("Case").to_dict()
    
    case_entry = {
        "metrics": {"1": metrics},
        "prediction_file": case_paths[i],  # Use original case path
        "reference_file": reference_paths[i]  # Include reference file path
    }
    metric_per_case.append(case_entry)

# Final JSON structure
final_json = {
    "foreground_mean": mean_row,  # Foreground mean
    "mean": {"1": mean_row},  # Mean for class "1"
    "metric_per_case": metric_per_case
}

# Save to summary.json
with open(json_output_path, "w") as file:
    json.dump(final_json, file, indent=4)

print(f"JSON saved at: {json_output_path}")
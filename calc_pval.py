# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 08:24:40 2025

@author: mujtabasaeed
"""

import pandas as pd
import scipy.stats as stats

# Load Excel files
file1 = "./nnUNet_results/2d.xlsx"  # Change to your actual file names
file2 = "./nnUNet_results/3d_fullres.xlsx"
file3 = "./nnUNet_results/3d_lowres.xlsx"

# Read Dice scores from each file
def load_dice_scores(filename):
    df = pd.read_excel(filename, sheet_name="edited")  # Read all sheets
    #sheet_name = list(df.keys())[0]  # Assuming the first sheet contains the data
    data = df["Dice"].iloc[1:75].values  # Extract rows 2 to 75 (0-based index)
    return data

dice1 = load_dice_scores(file1)
dice2 = load_dice_scores(file2)
dice3 = load_dice_scores(file3)

# Function to perform statistical tests
def compare_models(dice_a, dice_b, model_a, model_b):
    t_stat, p_ttest = stats.ttest_rel(dice_a, dice_b)  # Paired t-test
    w_stat, p_wilcoxon = stats.wilcoxon(dice_a, dice_b)  # Wilcoxon test

    print(f"Comparison: {model_a} vs {model_b}")
    print(f"Paired t-test: t-stat = {t_stat:.4f}, p-value = {p_ttest:.4f}")
    print(f"Wilcoxon test: W-stat = {w_stat:.4f}, p-value = {p_wilcoxon:.4f}\n")

# Compare models pairwise
compare_models(dice1, dice2, "2d", "3d fullres")
compare_models(dice1, dice3, "2d", "3d lowres")
compare_models(dice2, dice3, "3d fullres", "3d lowres")

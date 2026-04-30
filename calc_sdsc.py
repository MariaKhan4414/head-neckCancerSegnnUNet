# -*- coding: utf-8 -*-
"""
Created on Sat Mar 29 06:46:45 2025

@author: mujtabasaeed
"""

import os
import numpy as np
import nibabel as nib
import pandas as pd
from scipy.spatial.distance import directed_hausdorff, cdist
from skimage.morphology import binary_erosion, ball

def load_nifti(nifti_path):
    """Load a NIfTI file and return the binary mask."""
    nifti_img = nib.load(nifti_path)
    mask = nifti_img.get_fdata()
    return (mask > 0).astype(np.uint8)  # Convert to binary mask (1: object, 0: background)

def extract_surface(mask):
    """Extract surface points from a 3D binary mask."""
    struct_elem = ball(1)  # Structuring element for 3D erosion
    eroded_mask = binary_erosion(mask, struct_elem)
    surface = mask - eroded_mask  # Surface points are where original mask is 1 and eroded mask is 0
    return np.array(np.where(surface)).T  # Convert to (N, 3) coordinate list

def compute_surface_dsc(mask_gt, mask_pred):
    """Compute the Surface Dice Similarity Coefficient."""
    surface_gt = extract_surface(mask_gt)
    surface_pred = extract_surface(mask_pred)

    if len(surface_gt) == 0 or len(surface_pred) == 0:
        return 0.0  # If no surface points exist, return 0

    tolerance = 2  # Tolerance in voxels for surface matching

    # Compute distances from GT surface to Predicted surface
    dists_gt = cdist(surface_gt, surface_pred).min(axis=1)
    dists_pred = cdist(surface_pred, surface_gt).min(axis=1)

    # Count how many points are within the tolerance distance
    match_gt = np.sum(dists_gt <= tolerance)
    match_pred = np.sum(dists_pred <= tolerance)

    surface_dsc = (2 * (match_gt + match_pred)) / (len(surface_gt) + len(surface_pred))
    return surface_dsc

def compute_hd(mask_gt, mask_pred):
    """Compute the Hausdorff Distance (HD) and 95th percentile HD (HD95)."""
    surface_gt = extract_surface(mask_gt)
    surface_pred = extract_surface(mask_pred)

    if len(surface_gt) == 0 or len(surface_pred) == 0:
        return float('inf'), float('inf')  # If no surface points exist, return infinite distance

    # Compute directed Hausdorff distances
    hd_gt_to_pred = directed_hausdorff(surface_gt, surface_pred)[0]
    hd_pred_to_gt = directed_hausdorff(surface_pred, surface_gt)[0]

    # Compute HD (maximum of the directed distances)
    hd = max(hd_gt_to_pred, hd_pred_to_gt)

    # Compute HD95 (95th percentile of pairwise distances)
    all_distances = cdist(surface_gt, surface_pred).flatten()
    hd95 = np.percentile(all_distances, 95)

    return hd, hd95

def process_all_cases(gt_dir, pred_dir, output_csv="results_2d.xlsx"):
    """Process all cases and compute metrics, then save results to CSV."""
    results = []

    gt_files = sorted(os.listdir(gt_dir))
    pred_files = sorted(os.listdir(pred_dir))

    for gt_file, pred_file in zip(gt_files, pred_files):
        gt_path = os.path.join(gt_dir, gt_file)
        pred_path = os.path.join(pred_dir, pred_file)

        # Load masks
        mask_gt = load_nifti(gt_path)
        mask_pred = load_nifti(pred_path)

        # Compute metrics
        surface_dsc = compute_surface_dsc(mask_gt, mask_pred)
        hd, hd95 = compute_hd(mask_gt, mask_pred)

        results.append([gt_file, surface_dsc, hd, hd95])
        print(f"Processed {gt_file}: Surface DSC={surface_dsc:.4f}, HD={hd:.4f}, HD95={hd95:.4f}")

    # Convert to DataFrame and save
    df = pd.DataFrame(results, columns=["Case", "Surface_DSC", "HD", "HD95"])
    df.to_excel(output_csv, index=False)

    # Compute mean values
    mean_dsc = df["Surface_DSC"].mean()
    mean_hd = df["HD"].mean()
    mean_hd95 = df["HD95"].mean()

    print("\n===== FINAL REPORT =====")
    print(f"Mean Surface DSC: {mean_dsc:.4f}")
    print(f"Mean Hausdorff Distance (HD): {mean_hd:.4f}")
    print(f"Mean 95th Percentile HD (HD95): {mean_hd95:.4f}")

    return df

# Example usage
gt_dir = "./nnUNet_raw/Dataset002_HN/labelsTr"  # Directory containing ground truth NIfTI files
pred_dir = "./nnUNet_results/3d_cascade/postprocessed"  # Directory containing predicted segmentation NIfTI files

# Process all cases
df_results = process_all_cases(gt_dir, pred_dir)

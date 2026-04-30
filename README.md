# head-neckCancerSegnnUNet
# Head and Neck Tumor Segmentation using nnU-Net

This repository contains code for head and neck tumor segmentation using the nnU-Net framework on PET/CT images.

---

## Overview

This work evaluates different nnU-Net configurations:

- 2D U-Net  
- 3D Low Resolution  
- 3D Full Resolution  
- 3D Cascade  

---

## Dataset

The dataset used is the **Head and Neck Radiomics (HN1)** dataset.

It is publicly available from:

https://www.cancerimagingarchive.net/

---

## Files

- `hn_tumor_seg.ipynb` → main training notebook  
- `dicom_to_nifti.py` → DICOM to NIfTI conversion  
- `scan_dataset.py` → dataset preparation  
- `calc_sdsc.py` → SDSC calculation  
- `calc_pval.py` → statistical analysis  

---

## Requirements

- Python  
- PyTorch  
- nnU-Net  

---

## Code Availability

The code for this study is publicly available in this repository.

---

## Citation

[Add your paper citation here]

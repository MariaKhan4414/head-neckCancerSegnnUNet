import os
import json
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

# Paths (Modify this according to your setup)
SOURCE_DIR = "nnUNet_raw/dataset"  # Original dataset root
TARGET_DIR = "nnUNet_raw/Dataset003_HN"  # Destination for nnUNet format
IMAGE_TR_DIR = os.path.join(TARGET_DIR, "imagesTr")
LABEL_TR_DIR = os.path.join(TARGET_DIR, "labelsTr")

os.makedirs(IMAGE_TR_DIR, exist_ok=True)
os.makedirs(LABEL_TR_DIR, exist_ok=True)

def resample_image(image, reference_image):
    """Resample image to match reference image spacing, size, origin, and direction."""
    new_size = list(reference_image.GetSize())
    new_spacing = list(reference_image.GetSpacing())

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size)
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetOutputOrigin(reference_image.GetOrigin())  # Match origin
    resample.SetOutputDirection(reference_image.GetDirection())  # Match direction
    
    return resample.Execute(image)

def convert_dicom_to_nifti(dicom_dir, output_filename):
    reader = sitk.ImageSeriesReader()
    dicom_files = reader.GetGDCMSeriesFileNames(dicom_dir)
    reader.SetFileNames(dicom_files)
    image = reader.Execute()
    
    sitk.WriteImage(image, output_filename)
    print(f"Converted {dicom_dir} to {output_filename}")
    return image  # Return image for further processing

def transform_nifti_labels(nifti_path):
    image = sitk.ReadImage(nifti_path)
    image_array = sitk.GetArrayFromImage(image)
    image_array[image_array == 255] = 1
    label_image = sitk.GetImageFromArray(image_array)
    label_image.CopyInformation(image)
    sitk.WriteImage(label_image, nifti_path)
    print(f"Transformed labels in {nifti_path} (255 → 1)")

def process_case(case_folder):
    patient_id = os.path.basename(case_folder)
    case_folders = [f for f in os.listdir(case_folder) if os.path.isdir(os.path.join(case_folder, f))]

    ct_path, pet_path, label_path = None, None, None
    pet_image = None  # Store PET image to match CT resampling

    for scan_folder in case_folders:
        scan_path = os.path.join(case_folder, scan_folder)
        sub_folders = [os.path.join(scan_path, f) for f in os.listdir(scan_path) if os.path.isdir(os.path.join(scan_path, f))]

        for sub_folder in sub_folders:
            if "Segmentation" in sub_folder:
                label_path = sub_folder
            else:
                try:
                    reader = sitk.ImageSeriesReader()
                    dicom_files = reader.GetGDCMSeriesFileNames(sub_folder)
                    reader.SetFileNames(dicom_files)
                    image = reader.Execute()
                    size = image.GetSize()

                    if size[0] == 512:
                        ct_path = sub_folder
                    elif size[0] == 256:
                        pet_path = sub_folder
                        pet_image = image  # Store PET image
                except:
                    continue

    # Convert PET
    if pet_path:
        pet_nifti_path = os.path.join(IMAGE_TR_DIR, f"{patient_id}_0001.nii.gz")
        pet_image = convert_dicom_to_nifti(pet_path, pet_nifti_path)

    # Convert CT & Resample to match PET
    if ct_path and pet_image:
        ct_nifti_path = os.path.join(IMAGE_TR_DIR, f"{patient_id}_0000.nii.gz")
        ct_image = convert_dicom_to_nifti(ct_path, ct_nifti_path)
        ct_resampled = resample_image(ct_image, reference_image=pet_image)
        sitk.WriteImage(ct_resampled, ct_nifti_path)
        print(f"Resampled CT image to match PET properties (Spacing, Origin, Direction)")

    # Convert Labels
    if label_path:
        label_nifti_path = os.path.join(LABEL_TR_DIR, f"{patient_id}.nii.gz")
        convert_dicom_to_nifti(label_path, label_nifti_path)
        transform_nifti_labels(label_nifti_path)

def generate_dataset_json():
    dataset_json = {
        "channel_names": {"0": "CT", "1": "PET"},
        "labels": {"background": "0", "tumor": "1"},
        "numTraining": len(os.listdir(LABEL_TR_DIR)),
        "file_ending": ".nii.gz"
    }
    with open(os.path.join(TARGET_DIR, "dataset.json"), "w") as f:
        json.dump(dataset_json, f, indent=4)

if __name__ == "__main__":
    case_folders = [os.path.join(SOURCE_DIR, f) for f in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, f))]

    for case in tqdm(case_folders, desc="Processing Cases"):
        process_case(case)

    generate_dataset_json()
    print("Data conversion completed successfully!")

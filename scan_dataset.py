import os
import pandas as pd
import SimpleITK as sitk

# Define paths
IMAGE_DIR = "nnUNet_raw/Dataset002_HN/imagesTr"  # Path to CT/PET images
LABEL_DIR = "nnUNet_raw/Dataset002_HN/labelsTr"  # Path to segmentation masks
OUTPUT_FILE = "CT_metadata.csv"  # Output file name

# Get list of CT images (only files ending with '_0000.nii.gz')
ct_cases = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith("_0000.nii.gz")])[:10]

metadata_list = []

for case in ct_cases:
    ct_path = os.path.join(IMAGE_DIR, case)
    patient_id = case.replace("_0000.nii.gz", "")  # Remove modality suffix to get patient ID
    label_path = os.path.join(LABEL_DIR, f"{patient_id}.nii.gz")  # Get corresponding label file

    try:
        # Read CT Image
        ct_image = sitk.ReadImage(ct_path)
        label_image = sitk.ReadImage(label_path) if os.path.exists(label_path) else None
        
        # Extract metadata
        ct_metadata = {
            "PatientID": patient_id,
            "CT_Size": ct_image.GetSize(),
            "CT_Spacing": ct_image.GetSpacing(),
            "CT_Origin": ct_image.GetOrigin(),
            "CT_Direction": ct_image.GetDirection(),
            "Label_Size": label_image.GetSize() if label_image else "N/A",
            "Label_Spacing": label_image.GetSpacing() if label_image else "N/A",
            "Label_Origin": label_image.GetOrigin() if label_image else "N/A",
            "Label_Direction": label_image.GetDirection() if label_image else "N/A",
        }
        metadata_list.append(ct_metadata)

    except Exception as e:
        print(f"Error processing {case}: {e}")

# Save metadata to a CSV file
df = pd.DataFrame(metadata_list)
df.to_csv(OUTPUT_FILE, index=False)
print(f"Metadata saved to {OUTPUT_FILE}")

import os
import torch
import cv2
import numpy as np
from PIL import Image
from skimage import filters, morphology
from utils import normalize_percentile_to_255
from datahandlers import LF_M4RawDataset, DataTransform_M4RAW
import albumentations as A
from albumentations.pytorch import ToTensorV2
import time

count=0

def img_remove_artifact(img, min_size, area_threshold):
    threshold_binary = img > filters.threshold_multiotsu(img, classes=2)
    threshold_binary = morphology.binary_erosion(threshold_binary)
    keep_mask = morphology.remove_small_objects(threshold_binary, min_size=min_size)
    keep_mask = morphology.remove_small_holes(keep_mask, area_threshold=area_threshold)
    
    img_filter = np.multiply(img, keep_mask)
    
    return img_filter, keep_mask

def canny_edge_detector(image, low_threshold=30, high_threshold=50, kernel_size=3):
    # Apply Gaussian blur to reduce noise and improve edge detection
    image = ((image + 1) * 127.5).clip(0, 255).astype(np.uint8)  # Rescale to [0, 255]

    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    # Apply the Canny edge detector
    magnitude = cv2.Canny(blurred_image, low_threshold, high_threshold)
    
    magnitude = normalize_percentile_to_255(magnitude)

    return magnitude

def process_batch(batch, output_dir, batch_idx, low_threshold, high_threshold, kernel_size):
    global count
    # Create subdirectories for the batch
    batch_img_folder = os.path.join(output_dir, "img")
    batch_contour_folder = os.path.join(output_dir, "contours")
    
    os.makedirs(batch_img_folder, exist_ok=True)
    os.makedirs(batch_contour_folder, exist_ok=True)

    # Iterate through each image in the batch
    for idx, img in enumerate(batch):  # Assuming batch is a tuple of images and labels
        img = img[0]  # Remove the extra dimension that may come from dataset's output
        
        # Convert to NumPy and rescale before Canny
        img_np = img.numpy()
        
        # Apply Canny edge detection
        img_contour = canny_edge_detector(img_np, low_threshold, high_threshold, kernel_size)

        # Create a unique filename using both batch index and image index
        filename = f"image_{count}.png"
        count+=1

        # Save the original image (scaled back to [0, 255] and converted to uint8)
        img_rescaled = ((img.numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
        Image.fromarray(img_rescaled).save(os.path.join(batch_img_folder, filename))
        
        # Save the contour image
        img_contour = img_contour.astype(np.uint8)  # Ensure it's uint8 for saving
        Image.fromarray(img_contour).save(os.path.join(batch_contour_folder, filename))

    print(f"Processed batch {batch_idx} saved to {output_dir}")

def process_dataloader(dataloader, output_dir, low_threshold=80, high_threshold=150, kernel_size=3):
    for batch_idx, batch in enumerate(dataloader):
        print(f"Processing batch {batch_idx + 1}...")
        process_batch(batch, output_dir, batch_idx, low_threshold, high_threshold, kernel_size)


###### Transforms #######
# transform_hf = A.Compose([
#     A.Resize(256, 256),
#     ToTensorV2()
# ])
# hf_dataset_train = HF_MRI_Dataset(root_dir="../sauron/dataset/brain_fastMRI_DICOM/fastMRI_brain_DICOM", 
#                                   transform=transform_hf,
#                                   split="train")
# hf_train = torch.utils.data.DataLoader(hf_dataset_train, batch_size=1, shuffle=False, num_workers=1)
# output_dir = "dataset/hf"      
# process_dataloader(hf_train, output_dir)

lf_transform = DataTransform_M4RAW(img_size=256, combine_coil=True)
lf_dataset_train = LF_M4RawDataset(root_dir='dataset/low_field/multicoil_test', transform=lf_transform)
output_dir="dataset/lf/test"
lf_train = torch.utils.data.DataLoader(lf_dataset_train, batch_size=1, shuffle=False, num_workers=1)
process_dataloader(lf_train, output_dir)
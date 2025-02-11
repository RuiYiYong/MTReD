""" 
This script computes LPIPS scores given folder paths to original and reprojected images. 
The expected folder structure is shown below. 

PATH_TO_MTRED
|__01
|__02
|__...

REPROJECTED_IMAGES_PATH
|__MASt3R
    |__01
    |__02
    |__...
|__Colmap
    |__01
    |__02
    |__...
    
Where 01, 02, etc represents the videos in MTReD 
"""

# Imports
from os import walk

import pandas as pd
from PIL import Image
import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision import transforms

# Directories
PATH_TO_MTRED = "PATH_TO_MTRED"
REPROJECTED_IMAGES = "REPROJECTED_IMAGES_PATH"
MODELS = ["MASt3R", "Colmap"]

# Set up LPIPS
lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze')

# Loop model type
for model_name in MODELS:
    titles = []
    num_images_original = []
    num_images_final = []
    lpips_avgs = []

    # Loop each video in MTReD
    for i in range (1, 20):
        i_str = str(i)
        if len(i_str) < 2:
            i_str = f"0{i_str}"
            
        # Get the directories which contain the images
        original_image_directory = f"{PATH_TO_MTRED}/{i_str}"
        reprojected_image_directory = f"{REPROJECTED_IMAGES}/{model_name}/{i_str}"
        
        # Get the files in each folder
        original_files = next(walk(original_image_directory), (None, None, []))[2] 
        reprojected_files = next(walk(reprojected_image_directory), (None, None, []))[2] 
        
        # Start collecting metrics
        og_count = 0
        reprojected_count = 0
        similarity_avg = 0
        
        # Loop files
        for original_file in original_files:
            og_count += 1
            
            # Skip if no reprojection (i.e. no camera pose was found for this image)
            if original_file not in reprojected_files:
                continue
            
            # Open both images
            convert_tensor = transforms.ToTensor()
            original_image = convert_tensor(Image.open(f'{original_image_directory}/{original_file}'))
            projected_image = convert_tensor(Image.open(f'{reprojected_image_directory}/{original_file}'))

            original_image = original_image[None, :, :, :]
            projected_image = projected_image[None, :, :, :]
            
            # Copmute LPIPS score
            sim = lpips(original_image, projected_image).item()
            
            # Running average for LPIPS 
            similarity_avg = (similarity_avg*reprojected_count + sim)/(reprojected_count+1)
            reprojected_count += 1
        
        # Record metrics for this video in MTReD
        titles.append(f"{model_name}{i_str}")
        num_images_original.append(og_count)
        num_images_final.append(reprojected_count)
        lpips_avgs.append(similarity_avg)
    
    # Crate Dataframe
    df = pd.DataFrame([titles, num_images_original, num_images_final, lpips_avgs]).T
    df.columns = ["Datapoint", "# OG Images", "# Reprojected Images", "LPIPS Averages"]
    
    # Export results
    df.to_csv(f"{model_name}_OUTPUT_NAME.csv")
            
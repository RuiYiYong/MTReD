""" 
This script computes DiFPS scores given folder paths to original and reprojected images. 
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
from transformers import AutoImageProcessor, AutoModel
from torchvision import transforms
import torch.nn as nn

# Set up
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)

# Directories
PATH_TO_MTRED = "PATH_TO_MTRED"
REPROJECTED_IMAGES = "REPROJECTED_IMAGES_PATH"
MODELS = ["MASt3R", "Colmap"]

# Loop model type
for model_name in MODELS:
    titles = []
    num_images_original = []
    num_images_final = []
    difps_avg = []

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
            original_image = Image.open(f'{original_image_directory}/{original_file}')
            projected_image = Image.open(f'{reprojected_image_directory}/{original_file}')
            
            # Copmute DiFPS score
            with torch.no_grad():
                inputs1 = processor(images=original_image, return_tensors="pt").to(device)
                outputs1 = model(**inputs1)
                image_features1 = outputs1.last_hidden_state
                image_features1 = image_features1.mean(dim=1)

            with torch.no_grad():
                inputs2 = processor(images=projected_image, return_tensors="pt").to(device)
                outputs2 = model(**inputs2)
                image_features2 = outputs2.last_hidden_state
                image_features2 = image_features2.mean(dim=1)

            cos = nn.CosineSimilarity(dim=0)
            sim = cos(image_features1[0],image_features2[0]).item()
            sim = (sim+1)/2
            
            # Running average for LPIPS 
            similarity_avg = (similarity_avg*reprojected_count + sim)/(reprojected_count+1)
            reprojected_count += 1
        
        # Record metrics for this video in MTReD
        titles.append(f"{model_name}{i_str}")
        num_images_original.append(og_count)
        num_images_final.append(reprojected_count)
        difps_avg.append(similarity_avg)
    
    # Crate Dataframe
    df = pd.DataFrame([titles, num_images_original, num_images_final, difps_avg]).T
    df.columns = ["Datapoint", "# OG Images", "# Reprojected Images", "DiFPS Averages"]
    
    # Export results
    df.to_csv(f"{model_name}_OUTPUT_NAME.csv")
            
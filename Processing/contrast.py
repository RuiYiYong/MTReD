""" 
This script processes the MTReD dataset using a contrast based processing.
The expected input directory structure is as follows.

PATH_TO_MTRED
|__01
|__02
|__...
Where 01, 02, etc represents the videos in MTReD 
"""

import os
import cv2
from glob import glob

# PARAMETERS
CONFIDENCE_THRESHOLD = 0.5

def adjust_contrast_brightness(img, contrast, brightness):
    # Adjust contrast while accounting for brightness
    brightness += int(round(255*(1-contrast)/2))
    return cv2.addWeighted(img, contrast, img, 0, brightness)

def main(input_folder, output_folder):
    for dataset_index in range (1, 20):
        i_str = str(dataset_index)
        if len(i_str) < 2:
            i_str = f"0{i_str}"
        source_folder = f"{input_folder}/{i_str}"
        output_folder = f"{output_folder}/{i_str}_Contrast"
        
        # Crate if doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            
        # Loop images
        for image_path in glob(f"{source_folder}/*"):
            # load image
            image = cv2.imread(image_path)

            # Increase contrast
            output_image = adjust_contrast_brightness(image, 1.8, 0)
            
            # Save
            cv2.imwrite(f"{output_folder}/{image_path.split('/')[-1]}", output_image)

if __name__ == "__main__":
    # Load Folder
    PATH_TO_MTRED = f"PATH_TO_MTRED_DATASET"
    OUTPUT_PATH = f"OUTPUT_PATH"
    main(PATH_TO_MTRED, OUTPUT_PATH)
            
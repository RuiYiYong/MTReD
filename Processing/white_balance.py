""" 
This script processes the MTReD dataset using a white balacning algorithm with background 
extraction using a grounding dino and SAM pipeline.
"""

# Imports
import os
import sys
from glob import glob
from typing import List

import numpy as np
import cv2
import torch
import supervision as sv
from groundingdino.util.inference import Model
from segment_anything import SamPredictor
from segment_anything import sam_model_registry, SamPredictor

# MODEL SETUP
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(f'{CURRENT_DIR}/groundingdino/'))
HOME = os.getcwd()

SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = os.path.join(HOME, "weights", "sam_vit_h_4b8939.pth")
sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(device=DEVICE)
sam_predictor = SamPredictor(sam)

GROUNDING_DINO_CONFIG_PATH = os.path.join(os.getcwd(), "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(HOME, "weights", "groundingdino_swint_ogc.pth")
grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

# THRESHOLDS AND PARAMTERS
CLASSES = ['dune', 'ship', 'water', 'sky', 'island', 'land']
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25
CONFIDENCE_THRESHOLD = 0.45

# Function for SAM
def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)

# Function to enhance 
def enhance_class_name(class_names: List[str]) -> List[str]:
    return [
        f"all {class_name}s"
        for class_name
        in class_names
    ]
   
# Function to get mask means for a target class         
def get_means(image, target_class):
    # detect objects
    detections = grounding_dino_model.predict_with_classes(
        image=image,
        classes=enhance_class_name(class_names=CLASSES),
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD
    )

    # convert detections to masks
    detections.mask = segment(
        sam_predictor=sam_predictor,
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        xyxy=detections.xyxy
    )

    # annotate image with detections
    box_annotator = sv.BoxAnnotator()
    mask_annotator = sv.MaskAnnotator()
    labels = [
        f"{CLASSES[class_id]} {confidence:0.2f}" 
        for _, _, confidence, class_id, _ 
        in detections]
    annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
    annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

    # Get class names
    titles = [
        CLASSES[class_id]
        for class_id
        in detections.class_id
    ]

    # Get the mean
    image_means = []
    channel_index = 0
    valid_image = False
    # Loop detections
    for i, mask in enumerate(detections.mask):
        # If this image has a mask for target class 
        if titles[i] == target_class and detections.confidence[i] > CONFIDENCE_THRESHOLD:
            # Image becomes valid
            valid_image = True
            for channel in range(3):
                image_channel = image[:,:,channel]
                
                # Flatten both 
                image_channel = image_channel.flatten()
                mask_flatten = mask.flatten()
                
                # Keep valids and delete rest 
                vals = np.delete(image_channel, mask_flatten)

                # Account for mulitple masks per image
                if channel_index > 2:
                    current_index = channel_index%3
                    image_means[current_index] = image_means[current_index]*int(channel_index/3)+np.mean(vals)*(int(channel_index/3)+1)
                else:
                    image_means.append(np.mean(vals))
                channel_index += 1
    return [image_means, valid_image]

#  Main pipeline
def main(input_path, output_path):
    for class_index, target_class in enumerate(['sky', 'water']):
        # Loop each scene
        for dataset_index in range (1, 20):
            means = []
            i_str = str(dataset_index)
            if len(i_str) < 2:
                i_str = f"0{i_str}"
            input_folder = f"{input_path}/{i_str}"
            output_folder = f"{output_path}/{i_str}_{target_class}_white_balance"
            
            # Crate output directory if doesn't exist
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
                
            # Loop images to collect means
            for image_path in glob(f"{input_folder}/*"):
                # load image
                image = cv2.imread(image_path)
                # Get the mean value per colour channel of target background
                [image_means, valid_image] = get_means(image, target_class)
                # Append if valid
                if (valid_image):    
                    means.append(image_means)
            scene_means = np.mean(np.array(means), axis=0)
            
            # Loop again to process images
            for image_path in glob(f"{input_folder}/*"):
                # load image
                image = cv2.imread(image_path)
                
                # Get the mean value per colour channel of target background
                [image_means, valid_image] = get_means(image, target_class)
                
                # Calculate correction facotr and normalise if valid
                correction_factors = [scene_means / mean for mean in image_means]
                
                if valid_image:
                    # Create dummy image
                    output_image = np.zeros(image.shape)
                    for channel in range(3):
                        image_channel = image[:,:,channel]
                        transformed_channel = cv2.multiply(image_channel.astype(float), correction_factors[channel])
                                
                        # Assign to image
                        output_image[:, :, channel] = transformed_channel.reshape(image.shape[0], -1)
                    
                    # Save
                    cv2.imwrite(f"{output_folder}/{image_path.split('/')[-1]}", output_image)

if __name__ == "__main__":
    # FOLDERS
    PATH_TO_MTRED = f"PATH_TO_MTRED_DATASET"
    OUTPUT_PATH = f"PATH_TO_OUTPUT_DIRECTORY"
    main(PATH_TO_MTRED, OUTPUT_PATH)
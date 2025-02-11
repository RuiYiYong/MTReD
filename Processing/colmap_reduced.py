""" 
This script processes the MTReD dataset using Colmap outputs, reducing the dataset to only
frames which participate in SfM reconstrution. The expected directory structures are as follows.
Note that Colmap reconstructions should be exported as text files to obtain the said folder structure.

COLMAP_RESULTS_FOLDER
|__01
    |__images.txt
    |__cameras.txt
    |__...
|__02
    |__images.txt
    |__cameras.txt
    |__...
|__...

PATH_TO_MTRED
|__01
|__02
|__...
Where 01, 02, etc represents the videos in MTReD 
"""

# Imports
import os
import shutil

# DIRECTORIES
COLMAP_RESULTS_FOLDER = "PATH TO Colmap RESULTS"
PATH_TO_MTRED = "PATH TO MTRED DATASET DIRECTORY"
OUTPUT_FOLDER = "PATH TO OUTPUT DIRECTORY"

# Loop scenes
for i in range(1, 20):
    # String conversions
    index_str = str(i)
    if len(index_str) < 2: 
        index_str = f"0{index_str}"
    
    # Get path to colmap output file
    file_path = f"{COLMAP_RESULTS_FOLDER}/{index_str}/images.txt"
        
    # Collect the names for each frame used in colmap reconstruction 
    frames = set()
    fp = open(file_path, 'r')
    line = True
    while line:
        line = fp.readline()
        if (len(line)):
            if (line[0] != '#'):
                frame = line.split(' ')[-1].split("\n")[0]
                if (frame.split(".")[-1] == "png"):
                    frames.add(frame)
    
    # Create output directory
    output_path = f"{OUTPUT_FOLDER}/{index_str}"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Copy frames 
    for frame in frames:
        shutil.copy(f"{PATH_TO_MTRED}/{index_str}/{frame}", f"{output_path}/{frame}")

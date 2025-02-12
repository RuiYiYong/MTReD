""" 
This script takes a 3D point cloud and estimated camera poses from Colmap outputs and creates
projected images for each estimated camera pose. Points are expanded by a factor of 3. The 
COLMAP_RESULTS_FOLDER is expected to contain results in the following format.

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
"""

import math
import os

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R


# Expand point size (increase the mapping from 3D point to pixel ratio)
def expand_point(image, point, colour, distance, distances, radius=0):
    x = int(point[0])
    y = int(point[1])
    max_h = image.shape[0]
    max_w = image.shape[1]
   
    # Define the square region of interest around the center within the radius
    y_min = max(0, y - radius)
    y_max = min(max_h, y + radius + 1)
    x_min = max(0, x - radius)
    x_max = min(max_w, x + radius + 1)
    
    # Create a grid of coordinates within the bounding box
    y_grid, x_grid = np.ogrid[y_min:y_max, x_min:x_max]
    
    # Calculate squared distances from the center for each point in the grid
    distance_square = (y_grid - y) ** 2 + (x_grid - x) ** 2
    
    # Create a circular mask within the specified radius
    circular_mask = distance_square <= radius ** 2

    # Only update pixels where the new distance is smaller than the current distance
    region_distances = distances[y_min:y_max, x_min:x_max]
    mask_to_update = (distance < region_distances) & circular_mask

    # Update distances and image color where mask is True
    region_distances[mask_to_update] = distance
    image[y_min:y_max, x_min:x_max][mask_to_update] = colour

    # Save the updated distances back to the original array
    distances[y_min:y_max, x_min:x_max] = region_distances

    return image, distances

# Need this to determine if a point is in front or behind the camera
def get_points_cam_coords(points, extrinsic_matrix):
    """Convert 3D points from world coordinates to camera coordinates using the extrinsic matrix."""
    # Ensure points are in homogeneous coordinates
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
    
    # Convert points using the extrinsic matrix
    transformed_points = extrinsic_matrix @ points_homogeneous.T
    
    # Return the transformed points in Cartesian coordinates
    return transformed_points

def main(input_folder, output_folder):
    # Loop each video
    for i in range (1, 20):
        i_str = str(i)
        if len(i_str) < 2:
            i_str = f"0{i_str}"
            
        # Get output files from Colmap
        camera_file = f"{input_folder}/{i_str}/cameras.txt"
        images_file = f"{input_folder}/{i_str}/images.txt"
        points_file = f"{input_folder}/{i_str}/points3D.txt"
        save_folder = f"{output_folder}/{i_str}"

        # Create output folder
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
            
        # Load information
        camera_lines = open(camera_file).read().splitlines()[3:]
        image_lines = open(images_file).read().splitlines()[4:]
        # Get the points as an array (contains, error, track information, etc - convert everything first) 
        point_lines = open(points_file).read().splitlines()[3:]
        point_lines = np.char.split(point_lines)
        max_length = max(len(row) for row in point_lines)
        point_lines = np.array([row + [None] * (max_length - len(row)) for row in point_lines])

        # Extract point information 
        points = point_lines[:, 1:4].astype(float)  
        color = point_lines[:, 4:7].astype(float) 
        color = (color-np.min(color))/(np.max(color)-np.min(color)) # Normalise between 0 and 1
        
        # Initialise 
        camera_lines_dict = {}

        # Store the camera info as a dict using its id as the key
        for camera_line in camera_lines:
            camera_line_parts = camera_line.split(' ')
            camera_lines_dict[int(camera_line_parts[0])] = camera_line
        
        # Each frame has 2 lines of info in images.txt
        frames = int(len(image_lines)/2)
        
        # Loop all frames (estimated camera poses found)
        for frame_index in range(frames):
            # Extrinsic Matrix
            line1 = image_lines[frame_index*2].split(' ')
            rot = R.from_quat([line1[1], line1[2], line1[3], line1[4]]).as_matrix()
            trans = [line1[5], line1[6], line1[7]]
            intrinsic_params = camera_lines_dict[int(line1[8])].split(' ')
            world2cam = np.zeros((4,4))
            world2cam[:3, :3] = rot
            world2cam[:3, 3] = trans
            world2cam[3, 3] = 1
            cam2world = np.linalg.inv(world2cam)
            
            # Intrinsic Matrix
            intrinsic = np.array([
                [float(intrinsic_params[4]), 0, float(intrinsic_params[5])],
                [0, float(intrinsic_params[4]), float(intrinsic_params[6])],
                [0, 0, 1]
            ])

            # Distortion coefficients
            dist_coeffs = np.array([float(intrinsic_params[7]), 0, 0, 0, 0])

            # Project points into 2d
            points_2d, _ = cv2.projectPoints(points, world2cam[:3, :3], world2cam[:3, 3], intrinsic, dist_coeffs)

            # Flatten
            points_2d = points_2d.squeeze()

            # Create image and distance projections (each starts of as infinite)
            w = int(intrinsic_params[2])
            h = int(intrinsic_params[3])
            distances = np.full((h, w), np.inf) 
            image = np.zeros((h, w, 3))
            
            # Transform all the points from 3D to the frame of the current estimated camera
            points_cam_coords = get_points_cam_coords(points, world2cam)

            # Loop projected points
            for i, point in enumerate(points_2d):
                # Check if the point lies within the w, h and that it is in front of the camera
                if 0 <= point[0] < w and 0 <= point[1] < h and points_cam_coords[2, i] <= 0:
                    # Calculate the Euclidean distance
                    distance = math.sqrt((cam2world[0, 3]-points[i][0])**2 + (cam2world[1, 3] - points[i][1])**2 + (cam2world[2, 3] - points[i][2])**2)
                    # Update the image with expanded points based on the minimum distance rule
                    image, distances = expand_point(image, point, color[i], distance, distances, radius=3)
                    
            # Flip the image 
            image = image[:, ::-1, :] 
            
            # Save image
            Image.fromarray((image * 255).astype(np.uint8)).save(f"{save_folder}/{line1[-1]}")


if __name__ == "__main__":
    COLMAP_RESULTS_FOLDER = "COLMAP_RESULTS_FOLDER"
    OUTPUT_FOLDER = "FOLDER_TO_OUTPUT_RESULTS"
    main(COLMAP_RESULTS_FOLDER, OUTPUT_FOLDER)
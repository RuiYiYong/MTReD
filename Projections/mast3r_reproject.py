""" 
This script takes a 3D point cloud and estimated camera poses from MASt3R outputs and creates
projected images for each estimated camera pose. Points are expanded by a factor of 3. The 
MAST3R_RESULTS_FOLDER is expected to contain results in the following format.

MAST3R_RESULTS_FOLDER
|__01
    |__spare_pc.ply
    |__transforms.json
    |__...
|__02
    |__spare_pc.ply
    |__transforms.json
    |__...
|__...
"""

import json
import math
import os

import cv2
import numpy as np
import open3d as o3d
from PIL import Image


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
    # Loop videos
    for i in (range (1, 20)):
        i_str = str(i)
        if len(i_str) < 2:
            i_str = f"0{i_str}"
            
        # Folder to save reprojected images to
        save_folder = f"{output_folder}/{i_str}"
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
            
        # Get MASt3R output files 
        ply_file = f"{input_folder}/{i_str}/sparse_pc.ply"
        transforms_file = f"{input_folder}/{i_str}/transforms.json"

        # Load in transformation matrices and get cam params
        with open(transforms_file) as f:
            cam_params = json.load(f)

        # Load info from the PLY file
        point_cloud = o3d.io.read_point_cloud(ply_file)
        points = np.asarray(point_cloud.points)  
        color = np.asarray(point_cloud.colors)  
        # Normalise between 0 and 1
        color = (color-np.min(color))/(np.max(color)-np.min(color)) 
        
        # Loop frames (estimated camera poses)
        frames = len(cam_params["frames"])
        for frame_index in range(frames):
            # Get camera intrinsics and extrinisc
            intrinsic = np.array([
                [cam_params["fl_x"], 0, cam_params["cx"]],
                [0, cam_params["fl_y"], cam_params["cy"]],
                [0, 0, 1]
            ])
            cam2world = np.array(cam_params["frames"][frame_index]["transform_matrix"])
            world2cam = np.linalg.inv(cam2world)
            
            # Distortion coeeficients 
            dist_coeffs = np.array([cam_params["k1"], cam_params["k2"], cam_params["p1"], cam_params["p2"], 0])

            # Project points into 2d
            points_2d, _ = cv2.projectPoints(points, world2cam[:3, :3], world2cam[:3, 3], intrinsic, dist_coeffs)

            # Flatten
            points_2d = points_2d.squeeze()

            # Create image and distance projections (each starts of as infinite)
            distances = np.full((cam_params["h"], cam_params["w"]), np.inf)
            image = np.zeros((cam_params["h"], cam_params["w"], 3))
            
            # Transform all the points from 3D to the frame of the current estimated camera
            points_cam_coords = get_points_cam_coords(points, world2cam)

            # Loop projected points
            for i, point in enumerate(points_2d):
                # Check if the point lies within the w, h and that it is in front of the camera
                if 0 <= point[0] < cam_params['w'] and 0 <= point[1] < cam_params['h'] and points_cam_coords[2, i] <= 0:
                    # Calculate the Euclidean distance
                    distance = math.sqrt((cam2world[0, 3]-points[i][0])**2 + (cam2world[1, 3] - points[i][1])**2 + (cam2world[2, 3] - points[i][2])**2)
                    # Update the image with expanded points based on the minimum distance rule
                    image, distances = expand_point(image, point, color[i], distance, distances, radius=3)
            
            # Flip image
            image = image[:, ::-1, :] 
            
            # Save image
            Image.fromarray((image * 255).astype(np.uint8)).save(f"{save_folder}/{cam_params['frames'][frame_index]['file_path'].split('/')[-1]}")

if __name__ == "__main__":
    MAST3R_RESULTS_FOLDER = "MAST3R_RESULTS_FOLDER"
    OUTPUT_FOLDER = "REPROJECTED_IMAGES_OUTPUT_FOLDER"
    main(MAST3R_RESULTS_FOLDER, OUTPUT_FOLDER)
    
#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# sparse gradio demo functions
# --------------------------------------------------------
import sys
sys.path.insert(0, "PATH_TO_MAST3R/MAST3R DIRECTORY")

import math
import gradio
import os
import numpy as np
import functools
import trimesh
import copy
from scipy.spatial.transform import Rotation
import tempfile
import shutil
import json
from collections import OrderedDict
from array import array
import cv2
import torch
from os import listdir
from os.path import isfile, join

from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from mast3r.cloud_opt.tsdf_optimizer import TSDFPostProcess
from mast3r.model import AsymmetricMASt3R

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images
from dust3r.utils.device import to_numpy
from dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes
from dust3r.demo import get_args_parser as dust3r_get_args_parser

import matplotlib.pyplot as pl
from camera_conversion import Cameras

# Global variables and parameter setup
tmpdirname = "/tmp/TMP_TEST"
model = AsymmetricMASt3R.from_pretrained("naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric").to('cuda')
image_size = 512

class SparseGAState():
    def __init__(self, sparse_ga, should_delete=False, cache_dir=None, outfile_name=None):
        self.sparse_ga = sparse_ga
        self.cache_dir = cache_dir
        self.outfile_name = outfile_name
        self.should_delete = should_delete

    def __del__(self):
        if not self.should_delete:
            return
        if self.cache_dir is not None and os.path.isdir(self.cache_dir):
            shutil.rmtree(self.cache_dir)
        self.cache_dir = None
        if self.outfile_name is not None and os.path.isfile(self.outfile_name):
            os.remove(self.outfile_name)
        self.outfile_name = None


def get_args_parser():
    parser = dust3r_get_args_parser()
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--gradio_delete_cache', default=None, type=int,
                        help='age/frequency at which gradio removes the file. If >0, matching cache is purged')

    actions = parser._actions
    for action in actions:
        if action.dest == 'model_name':
            action.choices = ["MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"]
    # change defaults
    parser.prog = 'mast3r demo'
    return parser


def _convert_scene_output_to_glb(outfile, imgs, pts3d, mask, focals, cams2world, cam_size=0.05,
                                 cam_color=None, as_pointcloud=False,
                                 transparent_cams=False, silent=False):
    assert len(pts3d) == len(mask) <= len(imgs) <= len(cams2world) == len(focals)
    pts3d = to_numpy(pts3d)
    imgs = to_numpy(imgs)
    focals = to_numpy(focals)
    cams2world = to_numpy(cams2world)

    scene = trimesh.Scene()

    # full pointcloud
    if as_pointcloud:
        pts = np.concatenate([p[m.ravel()] for p, m in zip(pts3d, mask)]).reshape(-1, 3)
        col = np.concatenate([p[m] for p, m in zip(imgs, mask)]).reshape(-1, 3)
        valid_msk = np.isfinite(pts.sum(axis=1))
        pct = trimesh.PointCloud(pts[valid_msk], colors=col[valid_msk])
        pct.export(os.path.join("./", 'sparse_pc.ply'))
        scene.add_geometry(pct)
    else:
        meshes = []
        for i in range(len(imgs)):
            pts3d_i = pts3d[i].reshape(imgs[i].shape)
            msk_i = mask[i] & np.isfinite(pts3d_i.sum(axis=-1))
            meshes.append(pts3d_to_trimesh(imgs[i], pts3d_i, msk_i))
        mesh = trimesh.Trimesh(**cat_meshes(meshes))
        scene.add_geometry(mesh)

    # add each camera
    for i, pose_c2w in enumerate(cams2world):
        if isinstance(cam_color, list):
            camera_edge_color = cam_color[i]
        else:
            camera_edge_color = cam_color or CAM_COLORS[i % len(CAM_COLORS)]
        add_scene_cam(scene, pose_c2w, camera_edge_color,
                      None if transparent_cams else imgs[i], focals[i],
                      imsize=imgs[i].shape[1::-1], screen_width=cam_size)

    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
    scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))
    if not silent:
        print('(exporting 3D scene to', outfile, ')')
    scene.export(file_obj=outfile)
    return outfile
          
def get_3D_model_from_scene(silent, scene_state, min_conf_thr=2, as_pointcloud=False, mask_sky=False,
                            clean_depth=False, transparent_cams=False, cam_size=0.05, TSDF_thresh=0):
    """
    extract 3D_model (glb file) from a reconstructed scene
    """
    if scene_state is None:
        return None
    outfile = scene_state.outfile_name
    if outfile is None:
        return None

    # get optimized values from scene
    scene = scene_state.sparse_ga
    rgbimg = scene.imgs
    focals = scene.get_focals().cpu()
    cams2world = scene.get_im_poses().cpu()
    
    # Get camera poses
    make_nerf_transform_matrix_json(scene)
    
    # 3D pointcloud from depthmap, poses and intrinsics
    if TSDF_thresh > 0:
        tsdf = TSDFPostProcess(scene, TSDF_thresh=TSDF_thresh)
        pts3d, _, confs = to_numpy(tsdf.get_dense_pts3d(clean_depth=clean_depth))
    else:
        pts3d, _, confs = to_numpy(scene.get_dense_pts3d(clean_depth=clean_depth))
    msk = to_numpy([c > min_conf_thr for c in confs])
    return _convert_scene_output_to_glb(outfile, rgbimg, pts3d, msk, focals, cams2world, as_pointcloud=as_pointcloud,
                                        transparent_cams=transparent_cams, cam_size=cam_size, silent=silent)


def get_reconstructed_scene(outdir, gradio_delete_cache, model, device, silent, image_size, current_scene_state,
                            filelist, optim_level, lr1, niter1, lr2, niter2, min_conf_thr, matching_conf_thr,
                            as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size, scenegraph_type, winsize,
                            win_cyclic, refid, TSDF_thresh, shared_intrinsics, **kw):
    """
    from a list of images, run mast3r inference, sparse global aligner.
    then run get_3D_model_from_scene
    """
    imgs = load_images(filelist, size=image_size, verbose=not silent)
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1
        filelist = [filelist[0], filelist[0] + '_2']

    scene_graph_params = [scenegraph_type]
    if scenegraph_type in ["swin", "logwin"]:
        scene_graph_params.append(str(winsize))
    elif scenegraph_type == "oneref":
        scene_graph_params.append(str(refid))
    if scenegraph_type in ["swin", "logwin"] and not win_cyclic:
        scene_graph_params.append('noncyclic')
    scene_graph = '-'.join(scene_graph_params)
    pairs = make_pairs(imgs, scene_graph=scene_graph, prefilter=None, symmetrize=True)
    if optim_level == 'coarse':
        niter2 = 0
    # Sparse GA (forward mast3r -> matching -> 3D optim -> 2D refinement -> triangulation)
    if current_scene_state is not None and \
        not current_scene_state.should_delete and \
            current_scene_state.cache_dir is not None:
        cache_dir = current_scene_state.cache_dir
    elif gradio_delete_cache:
        cache_dir = tempfile.mkdtemp(suffix='_cache', dir=outdir)
    else:
        cache_dir = os.path.join(outdir, 'cache')
    scene = sparse_global_alignment(filelist, pairs, cache_dir,
                                    model, lr1=lr1, niter1=niter1, lr2=lr2, niter2=niter2, device=device,
                                    opt_depth='depth' in optim_level, shared_intrinsics=shared_intrinsics,
                                    matching_conf_thr=matching_conf_thr, **kw)
    if current_scene_state is not None and \
        not current_scene_state.should_delete and \
            current_scene_state.outfile_name is not None:
        outfile_name = current_scene_state.outfile_name
    else:
        outfile_name = tempfile.mktemp(suffix='_scene.glb', dir=outdir)

    scene_state = SparseGAState(scene, gradio_delete_cache, cache_dir, outfile_name)
    outfile = get_3D_model_from_scene(silent, scene_state, min_conf_thr, as_pointcloud, mask_sky,
                                      clean_depth, transparent_cams, cam_size, TSDF_thresh)
    return scene_state, outfile


def set_scenegraph_options(inputfiles, win_cyclic, refid, scenegraph_type):
    num_files = len(inputfiles) if inputfiles is not None else 1
    show_win_controls = scenegraph_type in ["swin", "logwin"]
    show_winsize = scenegraph_type in ["swin", "logwin"]
    show_cyclic = scenegraph_type in ["swin", "logwin"]
    max_winsize, min_winsize = 1, 1
    if scenegraph_type == "swin":
        if win_cyclic:
            max_winsize = max(1, math.ceil((num_files - 1) / 2))
        else:
            max_winsize = num_files - 1
    elif scenegraph_type == "logwin":
        if win_cyclic:
            half_size = math.ceil((num_files - 1) / 2)
            max_winsize = max(1, math.ceil(math.log(half_size, 2)))
        else:
            max_winsize = max(1, math.ceil(math.log(num_files, 2)))
    winsize = gradio.Slider(label="Scene Graph: Window Size", value=max_winsize,
                            minimum=min_winsize, maximum=max_winsize, step=1, visible=show_winsize)
    win_cyclic = gradio.Checkbox(value=win_cyclic, label="Cyclic sequence", visible=show_cyclic)
    win_col = gradio.Column(visible=show_win_controls)
    refid = gradio.Slider(label="Scene Graph: Id", value=0, minimum=0,
                          maximum=num_files - 1, step=1, visible=scenegraph_type == 'oneref')
    return win_col, winsize, win_cyclic, refid

# Function to get MASt3R outputs in nerf studio format
def make_nerf_transform_matrix_json(scene):
    # Get original image height and width
    img = cv2.imread(input_files_og[0])
    h = img.shape[0]
    w = img.shape[1] 
    
    # Calculate scale
    scale = w/scene.imgs[0].shape[1]

    intrinsics = scene.get_intrinsics().cpu()*scale # Get intrinsic matrix
    cams2world = scene.get_im_poses().cpu().numpy() # Get external matrix

    # Convert to Nerf Studio Format
    extrinsics = [] 
    for i, ext in enumerate(to_numpy(cams2world)):
        ext[:3, 3:] *= 1
        extrinsics.append((ext @ OPENGL).tolist())
    
    fx =  intrinsics[i][0][0].item(),  
    fy =  intrinsics[i][1][1].item(),
    cx = intrinsics[0][0][2].item()
    cy = intrinsics[0][1][2].item()
    
    # Initialise class and export
    camera_poses = Cameras(fx, fy, cx, cy, extrinsics, w, h)
    camera_poses.export(scene.img_paths)

# Main demo
def main_demo(tmpdirname, model, device='cuda', image_size=512, silent=False, delete_cache=False, inputfiles = []):
    # Helper functions
    def recon_fun(scene, inputfiles, optim_level, lr1, niter1, lr2, niter2, min_conf_thr, matching_conf_thr, 
                  as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size, 
                  scenegraph_type, winsize, win_cyclic, refid, TSDF_thresh, shared_intrinsics):
        return get_reconstructed_scene(tmpdirname, delete_cache, model, device, silent, image_size, 
                                       scene, inputfiles, optim_level, lr1, niter1, lr2, niter2, 
                                       min_conf_thr, matching_conf_thr, as_pointcloud, mask_sky, 
                                       clean_depth, transparent_cams, cam_size, scenegraph_type, 
                                       winsize, win_cyclic, refid, TSDF_thresh, shared_intrinsics)

    def model_from_scene_fun(scene, min_conf_thr, as_pointcloud, mask_sky, clean_depth, 
                             transparent_cams, cam_size, TSDF_thresh):
        return get_3D_model_from_scene(silent, scene, min_conf_thr, as_pointcloud, mask_sky, 
                                       clean_depth, transparent_cams, cam_size, TSDF_thresh)

    # Set initial scene state
    scene = None
    print(f"Running demo with tmpdirname: {tmpdirname} on device: {device}")

    # Define input parameters
    optim_level = 'refine'
    lr1, niter1 = 0.07, 500
    lr2, niter2 = 0.014, 200
    min_conf_thr, matching_conf_thr = 1.5, 5.0
    as_pointcloud, mask_sky, clean_depth, transparent_cams = True, False, True, False
    cam_size, TSDF_thresh = 0.2, 0.0
    scenegraph_type, winsize, win_cyclic, refid = 'complete', 1, False, 0
    shared_intrinsics = True

    # Run the reconstruction function
    scene, outmodel = recon_fun(scene, inputfiles, optim_level, lr1, niter1, lr2, niter2, 
                                min_conf_thr, matching_conf_thr, as_pointcloud, mask_sky, 
                                clean_depth, transparent_cams, cam_size, scenegraph_type, 
                                winsize, win_cyclic, refid, TSDF_thresh, shared_intrinsics)

    # Optionally update the 3D model based on threshold changes
    outmodel = model_from_scene_fun(scene, min_conf_thr, as_pointcloud, mask_sky, 
                                    clean_depth, transparent_cams, cam_size, TSDF_thresh)

    # Save or process the output model as needed
    print("3D model generated:", outmodel)

# Main program function
def main(mtred_folder, output_folder, mast3r_folder):
    # Loop scenes
    for i in (range (1, 20)):
        # Get folder names
        i = str(i)
        if len(i) < 2:
            i = f"0{i}"
            
        # Get image names and store 
        input_folder = f"{mtred_folder}/{i}"
        inputfiles = [f"{input_folder}/{f}" for f in listdir(input_folder) if isfile(join(input_folder, f))]
        global input_files_og
        input_files_og = [f"{input_folder}/{f}" for f in listdir(input_folder) if isfile(join(input_folder, f))]
        # Run MASt3R
        main_demo(tmpdirname, model, device='cuda', image_size=image_size, silent=False, delete_cache=False, inputfiles = inputfiles)

        # Delete cache after running
        shutil.rmtree(tmpdirname, ignore_errors=False, onerror=None)
        
        # Create results folder
        output_dir = f"{output_folder}/{i}_output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Move output files
        shutil.move(f"{mast3r_folder}/transforms.json", f"{output_dir}/transforms.json")
        shutil.move(f"{mast3r_folder}/sparse_pc.ply", f"{output_dir}/sparse_pc.ply")
        shutil.move(f"{mast3r_folder}/Output.txt", f"{output_dir}/Output.txt")
        
        # Copy input images
        shutil.copytree(input_folder, f"{output_dir}/images")


if __name__ == "__main__":
    MTRED_PATH = "PATH_TO_MTRED"
    OUTPUT_PATH = "PATH_TO_OUTPUT_DIRECTORY"
    MAST3R_PATH = "PATH_TO_MAST3R_PARENT_DRIECTORY"
    main(MTRED_PATH, OUTPUT_PATH, MAST3R_PATH)




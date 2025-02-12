# MTReD: 3D Reconstruction Dataset for Fly-over Videos of Maritime Domain
## MTreD
<p>MTReD consists of 19 fly-over Maritime scenes with images frame extracted from open-source YouTube videos. The dataset covers a variety of Maritime scenes including ships, islands, and caostlines. The MTReD is found in the dataset folder of this repositiory. </p>

![Screenshot from 2025-02-12 11-40-14](https://github.com/user-attachments/assets/d0b6d999-1e83-4377-a72a-c902dd594006)


## MASt3R
<p> The MASt3R demo script has been slightly modified to run automatically on the MTReD. This new script is located at "/mast3r_modified/mast3r/demo_program.py. For each scene, the script should output a PLY point cloud file, camera pose information, and  a text file containing the reprojection error. Note that due to these modifications to outputs, the original demo script run on gradio might run into some errors. For setup of the MASt3R environment, refer to the original <a href="https://github.com/naver/mast3r">MASt3R repository</a>.</p>

## Colmap
<p> Colmap is the model used to impelment SfM and was used through its GUI interface. Detailed setup and use instructions can be found on <a href="https://colmap.github.io/">Colmap's project page</a>. An automatic reconstruction was run for each scene and results were outputted using file --> export model as text. 
</p>

# Sample Outputs
<p>Note that some sample outputs have been provided (1 for Colmap and 1 for MASt3R). These can be used with the projection script, however an error will be returned once the script looks for results from the 2nd scene. However, if your setup is correct, you should be able to obtain reprojected frames for the first scene for both MASt3R and Colmap.</p>

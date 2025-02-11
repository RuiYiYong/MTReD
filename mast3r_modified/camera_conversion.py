from collections import OrderedDict 
import json

class Cameras:
    def __init__(self, fx, fy, cx, cy, extrinsics, w, h):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.extrinsics = extrinsics
        self.w = w
        self.h = h
    
    def export(self, image_directory):
        out_file = OrderedDict()
        out_file['w'] = self.w
        out_file['h'] = self.h
        out_file['cx'] = self.cx
        out_file['cy'] = self.cy
        out_file['fl_x'] = self.fx[0]

        out_file['fl_y'] = self.fy[0]
        out_file["ply_file_path"] = "sparse_pc.ply"
        
        # Deformation (Always 0 on mast3r) 
        out_file['k1'] = 0.0
        out_file['k2'] = 0.0
        out_file['p1'] = 0.0
        out_file['p2'] = 0.0
        
        # Get file names
        for i in range (0, len(image_directory)):
            image_directory[i] = image_directory[i].split("/")[-1]
        for i in range (0, len(image_directory)):
            image_directory[i] = f"images/{image_directory[i]}"
            
        # Combine file names with extrinsics matrices 
        frames = []
        for i, file_name in enumerate(image_directory):
            frames.append({
                    'file_path': file_name, 
                    'transform_matrix': self.extrinsics[i], 
                    })
        out_file['frames'] = frames
        
        # Export
        with open('./transforms.json', 'w') as file: 
            json.dump(out_file, file, ensure_ascii=False, indent='\t')

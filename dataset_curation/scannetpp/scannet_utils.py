import os
import json
import torch
import numpy as np
import cv2
from pytorch3d.io import load_ply
from pytorch3d.structures import Meshes


def load_scannet_ply(scene_file, device="cpu"):
    """Load a .ply mesh from ScanNet++ and return a PyTorch3D Meshes object."""
    verts, faces = load_ply(scene_file)
    return Meshes(verts=[verts.to(device)], faces=[faces.to(device)])


def load_camera_params(camera_file_path, img_name):
    """
    Load camera parameters for a given frame from a ScanNet++ nerfstudio JSON file.

    The JSON has shared intrinsics (fl_x, fl_y, cx, cy, h, w) and a "frames" list.
    Each frame has a "file_path", a 4x4 "transform_matrix" (cam-to-world, Blender/OpenGL
    convention), and an "is_bad" flag.

    Returns (R, T, K, image_hw, Trans_CV) as pytorch3d-convention tensors, or
    (None, None, None, None, None) if the frame is missing or marked bad.

    Note on coordinate transforms:
        The loaded transform_matrix is cam-to-world in Blender/OpenGL convention.
        We convert to CV world-to-cam, then apply a column-swap correction to align
        with COLMAP (verified against ScanNet++ official rasterization code), then
        convert to pytorch3d convention.
    """
    with open(camera_file_path, 'r') as f:
        cam_data = json.load(f)

    fl_x = cam_data.get("fl_x")
    fl_y = cam_data.get("fl_y")
    cx = cam_data.get("cx")
    cy = cam_data.get("cy")
    h = cam_data.get("h")
    w = cam_data.get("w")

    K = torch.tensor([
        [fl_x, 0.0, cx],
        [0.0, fl_y, cy],
        [0.0, 0.0, 1.0]
    ], dtype=torch.float32).unsqueeze(0)

    frame_found = None
    for frame in cam_data.get("frames", []):
        if frame.get("file_path").split(".")[0] == img_name:
            if frame.get("is_bad", False):
                print("bad frame {} - {}".format(camera_file_path, img_name))
                return None, None, None, None, None
            frame_found = frame
            break

    if frame_found is None:
        print("frame not found {} - {}".format(camera_file_path, img_name))
        return None, None, None, None, None

    transform = frame_found.get("transform_matrix")
    if transform is None:
        print("transform not found {} - {}".format(camera_file_path, img_name))
        return None, None, None, None, None

    M = torch.tensor(transform, dtype=torch.float32)
    if M.shape != (4, 4):
        raise ValueError("Expected transform_matrix of shape (4,4)")

    R_bcam_to_cvcam = torch.tensor([[1, 0, 0, 0],
                                     [0, -1, 0, 0],
                                     [0, 0, -1, 0],
                                     [0, 0, 0, 1]], dtype=torch.float32, device=M.device)

    R_cv_to_pyt3d = torch.tensor([[-1, 0, 0, 0],
                                   [0, -1, 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]], dtype=torch.float32, device=M.device)

    # Blender cam-to-world -> CV world-to-cam
    Trans_CV = R_bcam_to_cvcam @ torch.linalg.inv(M)
    # Column-swap correction to align with COLMAP world-to-cam convention
    R_col = torch.zeros((3, 3), dtype=torch.float32)
    R_col[:, 1] = Trans_CV[:3, 0]
    R_col[:, 0] = Trans_CV[:3, 1]
    R_col[:, 2] = -Trans_CV[:3, 2]
    Trans_CV[:3, :3] = R_col

    # CV to pytorch3d
    RR = R_cv_to_pyt3d @ Trans_CV

    # pytorch3d uses right-multiplication, so rotation must be transposed
    R = RR[:3, :3].T.unsqueeze(0)   # (1, 3, 3)
    T = RR[:3, 3].unsqueeze(0)      # (1, 3)

    return R, T, K, (h, w), Trans_CV


def get_image_resolution(scene_folder):
    """Return (height, width) from the first image in dslr/downscaled_undistorted_images."""
    image_dir = os.path.join(scene_folder, "dslr/downscaled_undistorted_images")
    image_name = os.listdir(image_dir)[0]
    image_path = os.path.join(image_dir, image_name)

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Error: Image {image_path} does not exist!")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error: Failed to load image {image_path}")

    h, w = image.shape[:2]
    return (h, w)

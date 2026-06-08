import os
import torch
import numpy as np
import cv2
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes, join_meshes_as_scene


def load_multiple_objs(scene_folder, device="cpu", merge_meshes=True):
    """
    Load all .obj files listed in meta.txt as a single merged PyTorch3D Meshes object.

    Expected layout:
        meta file:  {scene_folder}/meta.txt
        mesh files: {scene_folder}/../meshes/{obj_name}.obj
    """
    meta_file = os.path.join(scene_folder, "meta.txt")
    meshes_folder = os.path.join(os.path.dirname(scene_folder), "meshes")

    if not os.path.exists(meta_file):
        print(f"Error: {meta_file} does not exist!")
        return None
    if not os.path.exists(meshes_folder):
        print(f"Error: {meshes_folder} does not exist!")
        return None

    obj_paths = []
    with open(meta_file, "r") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 1:
                continue
            obj_name = parts[1]
            obj_path = os.path.join(meshes_folder, f"{obj_name}.obj")
            if os.path.exists(obj_path):
                obj_paths.append(obj_path)
            else:
                print(f"Warning: {obj_path} not found.")

    if not obj_paths:
        print("No valid .obj files found.")
        return None

    try:
        mesh = load_objs_as_meshes(obj_paths, device=torch.device(device))
        if merge_meshes:
            mesh = join_meshes_as_scene(mesh)
        return mesh
    except Exception as e:
        print(f"Error loading meshes: {e}")
        return None


def load_objmesh_as_list(scene_folder, device="cpu"):
    """
    Load all .obj files listed in meta.txt as a list of individual PyTorch3D Meshes.

    Expected layout:
        meta file:  {scene_folder}/meta.txt
        mesh files: {scene_folder}/../meshes/{obj_name}.obj
    """
    meta_file = os.path.join(scene_folder, "meta.txt")
    meshes_folder = os.path.join(os.path.dirname(scene_folder), "meshes")

    if not os.path.exists(meta_file):
        print(f"Error: {meta_file} does not exist!")
        return None
    if not os.path.exists(meshes_folder):
        print(f"Error: {meshes_folder} does not exist!")
        return None

    obj_paths = []
    with open(meta_file, "r") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 1:
                continue
            obj_name = parts[1]
            obj_path = os.path.join(meshes_folder, f"{obj_name}.obj")
            if os.path.exists(obj_path):
                obj_paths.append(obj_path)
            else:
                print(f"Warning: {obj_path} not found.")

    if not obj_paths:
        print("No valid .obj files found.")
        return None

    return [load_objs_as_meshes([p], device=torch.device(device)) for p in obj_paths]


def load_intrinsics(scene_folder, output_tensor=True):
    """Load the camera intrinsic matrix from intrinsics.txt."""
    intrinsics_file = os.path.join(scene_folder, "intrinsics.txt")

    if not os.path.exists(intrinsics_file):
        raise FileNotFoundError(f"Error: {intrinsics_file} does not exist!")

    intrinsics = []
    with open(intrinsics_file, "r") as file:
        for line in file:
            row = list(map(float, line.strip().split()))
            intrinsics.append(row)

    if output_tensor:
        return torch.tensor(intrinsics, dtype=torch.float32).unsqueeze(0)
    else:
        return np.array(intrinsics).astype(np.float32)


def select_poses(scene_folder, interval=1):
    """
    Uniformly select pose files from the camera_pose folder at a given interval.
    Returns (pose_ids, R_list, T_list) where R is (1,3,3) and T is (1,3).
    """
    pose_folder = os.path.join(scene_folder, "camera_pose")

    if not os.path.exists(pose_folder):
        raise FileNotFoundError(f"Error: {pose_folder} does not exist!")

    pose_files = sorted([f for f in os.listdir(pose_folder) if f.endswith(".txt")])
    if not pose_files:
        raise ValueError("No pose files found in the directory!")

    selected_files = pose_files[::interval]
    R_list, T_list, pose_ids = [], [], []

    for pose_file in selected_files:
        pose_path = os.path.join(pose_folder, pose_file)
        with open(pose_path, "r") as file:
            lines = file.readlines()

        if len(lines) < 3:
            print(f"Skipping {pose_file} due to insufficient lines.")
            continue

        pose_matrix = np.array([list(map(float, line.strip().split())) for line in lines])

        if pose_matrix.shape == (3, 4):
            R, T = pose_matrix[:, :3], pose_matrix[:, 3]
        elif pose_matrix.shape == (4, 4):
            R, T = pose_matrix[:3, :3], pose_matrix[:3, 3]
        else:
            print(f"Skipping {pose_file} due to invalid shape {pose_matrix.shape}.")
            continue

        pose_ids.append(int(pose_file.split(".")[0]))
        R_list.append(torch.tensor(R, dtype=torch.float32).unsqueeze(0))
        T_list.append(torch.tensor(T, dtype=torch.float32).unsqueeze(0))

    return pose_ids, R_list, T_list


def read_a_pose(pose_file, cam_convention="pytorch3d"):
    """
    Read pose (R, T) from a .txt file (cam-to-world under CV convention) and
    convert to pytorch3d world-to-cam convention.
    """
    with open(pose_file, "r") as file:
        lines = file.readlines()

    if len(lines) < 3:
        print(f"Skipping {pose_file} due to insufficient lines.")
        return

    pose_c2w_cv = np.array([list(map(float, line.strip().split())) for line in lines])

    # Computer Vision camera convention to PyTorch3D camera convention
    pose_cv2p3d = np.array([[-1, 0, 0, 0],
                             [0, -1, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]])

    if cam_convention == 'pytorch3d':
        T_w2c = pose_cv2p3d @ np.linalg.inv(pose_c2w_cv)
    elif cam_convention == "cv":
        T_w2c = np.linalg.inv(pose_c2w_cv)
    else:
        raise NotImplementedError()

    if T_w2c.shape == (3, 4):
        R, T = T_w2c[:, :3], T_w2c[:, 3]
    elif T_w2c.shape == (4, 4):
        R, T = T_w2c[:3, :3], T_w2c[:3, 3]
    else:
        print(f"Skipping {pose_file} due to invalid shape {T_w2c.shape}.")
        return

    if cam_convention == 'pytorch3d':
        # pytorch3d uses right-multiplication, so rotation must be transposed
        return torch.tensor(R.T, dtype=torch.float32).unsqueeze(0), \
               torch.tensor(T, dtype=torch.float32).unsqueeze(0)
    else:
        return R, T


def get_image_resolution(scene_folder):
    """Return (height, width) from the first image in the rgb folder."""
    image_path = os.path.join(scene_folder, "rgb", "000000.png")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Error: Image {image_path} does not exist!")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error: Failed to load image {image_path}")

    h, w = image.shape[:2]
    return (h, w)

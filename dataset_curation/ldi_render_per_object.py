'''
Render LDIs from one object file and the corresponding camera poses.
'''

import argparse
import os
import sys
import numpy as np
import open3d as o3d
import torch
import glob
import trimesh
from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRasterizer,
    PerspectiveCameras,
    TexturesVertex,
)
from pytorch3d.structures import Meshes
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.ops import sample_points_from_meshes
import matplotlib.pyplot as plt
from PIL import Image

sys.path.append(os.path.dirname(__file__))
import scrream
import scannetpp


DEVICE = 'cpu'
IMG_SIZE = 512
CAM_LENS = 35
CAM_SENSOR_WIDTH = 32


parser = argparse.ArgumentParser()
parser.add_argument("--object_path", type=str, required=True,
                    help="Path to the input object/scene (.glb/.obj/.ply or scene folder, depending on dataset_type)")
parser.add_argument("--camera_path", type=str, required=True,
                    help="Path to the camera poses (a folder of per-view files, a single pose file, or an image id, depending on dataset_type)")
parser.add_argument("--num_layers", type=int, required=True,
                    help="Number of LDI layers to render (faces_per_pixel)")
parser.add_argument("--online_sanity_check", type=int, required=True,
                    help="If non-zero, abort early when the first view's LDI is inconsistent with the rendered image")
parser.add_argument("--view_number", type=int,
                    help="Number of views to render (per-view datasets: objaverse, front3d, gso)")
parser.add_argument("--dataset_type", type=str,
                    choices=["objaverse", "front3d", "gso", "scrream", "scannetpp"],
                    default="objaverse",
                    help="Dataset to render, selecting the loading/transformation/saving pipeline")
parser.add_argument("--point_priority_thres", type=int, default=None,
                    help="Skip meshes with more visible vertices than this threshold (scannetpp only)")
args = parser.parse_args()


def trace_transforms_for_geometry(scene: trimesh.Scene, geometry_name: str) -> torch.Tensor:
    '''
    Trace the transform graph to compute the cumulative transform for the given geometry.
    Returns identity if geometry_name is not found.
    '''
    graph = scene.graph.transforms

    node_for_geometry = None
    for (parent, child), edge_data in graph.edge_data.items():
        if edge_data.get('geometry') == geometry_name:
            node_for_geometry = child
            break

    if node_for_geometry is None:
        return torch.eye(4, dtype=torch.float32)

    transform_stack = []
    node = node_for_geometry
    while node != "world" and node in graph.parents:
        parent = graph.parents[node]
        local_mat = graph.edge_data.get((parent, node), {}).get("matrix", np.eye(4))
        transform_stack.append(local_mat)
        node = parent

    final_transform = np.eye(4)
    for mat in reversed(transform_stack):
        final_transform = final_transform @ mat

    return torch.tensor(final_transform, dtype=torch.float32)


def load_glb_as_mesh_updated(glb_path, device="cuda"):
    '''Load a GLB model, transform all meshes to world space, and merge into a single mesh.'''
    scene_or_mesh = trimesh.load(glb_path)

    if isinstance(scene_or_mesh, trimesh.Scene):
        meshes = []
        for geometry_name, mesh in scene_or_mesh.geometry.items():
            if not isinstance(mesh, trimesh.Trimesh):
                continue

            if geometry_name in scene_or_mesh.graph.nodes_geometry:
                transform_np = scene_or_mesh.graph[geometry_name][0]
                transform = torch.tensor(transform_np, dtype=torch.float32)
            else:
                transform = trace_transforms_for_geometry(scene_or_mesh, geometry_name)

            verts = torch.tensor(mesh.vertices, dtype=torch.float32)
            verts_h = torch.cat([verts, torch.ones(len(verts), 1)], dim=-1)
            verts_world = (transform @ verts_h.T).T[:, :3]
            faces = torch.tensor(mesh.faces, dtype=torch.int64)
            meshes.append((verts_world, faces))

        if len(meshes) == 0:
            raise ValueError("No meshes found in GLB file.")

        all_verts = torch.cat([m[0] for m in meshes], dim=0)
        all_faces = torch.cat([
            m[1] + sum(m2[0].shape[0] for m2 in meshes[:i])
            for i, m in enumerate(meshes)
        ], dim=0)
    else:
        combined_mesh = scene_or_mesh
        combined_mesh.merge_vertices()
        all_verts = torch.tensor(combined_mesh.vertices, dtype=torch.float32)
        all_faces = torch.tensor(combined_mesh.faces, dtype=torch.int64)

    all_verts = all_verts.to(device)
    all_faces = all_faces.to(device)

    return Meshes(verts=[all_verts], faces=[all_faces])


def load_camera_params_glb(camera_path: str, cam_lens, cam_sensor_width, img_size, device):
    '''
    Convert cam_to_world transformation under Blender coordinate to
    cam_to_world transformation from glTF world coordinate to pytorch3d camera coordinate.
    '''
    T_b_w2cam = np.load(camera_path)  # 3x4 matrix from Blender
    T_b_w2cam = np.concatenate((T_b_w2cam, np.array([[0, 0, 0, 1]])), axis=0)  # 4x4

    # glFT->Blender transformation by swapping y-z axes
    R_gltf2blender = np.array([[1, 0, 0, 0],
                                [0, 0, -1, 0],
                                [0, 1, 0, 0],
                                [0, 0, 0, 1]])

    # Blender to PyTorch3D coordinates
    R_bcam2py3d = np.array([[-1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, -1, 0],
                             [0, 0, 0, 1]])

    # glFT world pts -> blender world pts -> blender cam pts -> pytorch3d cam pts
    T_py_cam2w = R_bcam2py3d @ T_b_w2cam @ R_gltf2blender

    # pytorch3d uses right-multiplication for rendering, so rotation must be transposed
    R = torch.tensor(T_py_cam2w[:3, :3].T, dtype=torch.float32, device=device)[None]
    T = torch.tensor(T_py_cam2w[:3, -1], dtype=torch.float32, device=device)[None]
    K = compute_intrinsics(cam_lens, cam_sensor_width, img_size, device)

    return R, T, K


def load_camera_params_obj(camera_path: str, cam_lens, cam_sensor_width, img_size, device):
    '''
    Convert cam_to_world transformation under Blender coordinate to
    cam_to_world transformation from OBJ world coordinate to pytorch3d camera coordinate.
    '''
    res = np.load(camera_path, allow_pickle=True)
    if isinstance(res, np.ndarray):
        T_b_w2cam = res
        assert cam_lens is not None
    elif isinstance(res.item(), dict):
        res = res.item()
        T_b_w2cam, cam_lens = res["T_b_w2cam"], res["cam_len"]
    else:
        raise NotImplementedError()

    T_b_w2cam = np.concatenate((T_b_w2cam, np.array([[0, 0, 0, 1]])), axis=0)  # 4x4

    # Blender to PyTorch3D coordinates
    R_bcam2py3d = np.array([[-1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, -1, 0],
                             [0, 0, 0, 1]])

    # Blender to OBJ coordinate
    R_b2obj = np.array([[1, 0, 0, 0],
                        [0, 0, 1, 0],
                        [0, -1, 0, 0],
                        [0, 0, 0, 1]])

    # OBJ-coord -> Blender-coord -> Blender cam -> pytorch3d cam
    T_py_cam2w = R_bcam2py3d @ T_b_w2cam @ np.linalg.inv(R_b2obj)

    # pytorch3d uses right-multiplication, so rotation must be transposed
    R = torch.tensor(T_py_cam2w[:3, :3].T, dtype=torch.float32, device=device)[None]
    T = torch.tensor(T_py_cam2w[:3, -1], dtype=torch.float32, device=device)[None]
    K = compute_intrinsics(cam_lens, cam_sensor_width, img_size, device)

    return R, T, K


def compute_intrinsics(cam_lens, cam_sensor_width, img_size, device):
    focal_length = (cam_lens * img_size) / cam_sensor_width

    K = torch.tensor([
        [focal_length, 0, img_size / 2],
        [0, focal_length, img_size / 2],
        [0, 0, 1]
    ], dtype=torch.float32, device=device)[None]

    return K


def normalize_mesh(mesh: Meshes, R=None, T=None):
    verts = mesh.verts_packed()

    bbox_min, _ = verts.min(dim=0)
    bbox_max, _ = verts.max(dim=0)
    scale = 1.0 / torch.max(bbox_max - bbox_min)
    verts = verts * scale

    bbox_min, _ = verts.min(dim=0)
    bbox_max, _ = verts.max(dim=0)
    center = (bbox_max + bbox_min) / 2.0
    verts = verts - center

    return Meshes(verts=[verts], faces=mesh.faces_list(), textures=mesh.textures)


def render_ldi(mesh, R, T, K, num_layers, img_size, device):
    if mesh.verts_packed().numel() == 0 or mesh.faces_packed().numel() == 0:
        raise ValueError("Mesh has no vertices or faces.")

    cameras = PerspectiveCameras(
        # in_ndc=False is required when using intrinsics in pixel units
        in_ndc=False,
        R=R,
        T=T,
        focal_length=K[:, 0, 0],
        principal_point=K[:, :2, 2],
        image_size=((img_size, img_size) if isinstance(img_size, int) else img_size,),
        device=device
    )

    raster_settings = RasterizationSettings(
        image_size=img_size,
        blur_radius=0.0,
        faces_per_pixel=num_layers,
        perspective_correct=True
    )

    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    fragments = rasterizer(mesh)
    zbuf = fragments.zbuf  # [B, H, W, faces_per_pixel]

    return {"depth": zbuf.cpu().numpy()}


def filter_mesh_from_proj(mesh, cameras, K, img_size, device):
    '''Filter out vertices and faces that fall completely outside the camera's FOV.'''
    verts = mesh.verts_packed().to(device)
    faces = mesh.faces_packed().to(device)

    verts_batch = verts.unsqueeze(0)
    verts_cam = cameras.get_world_to_view_transform().transform_points(verts_batch)[0]

    fx = K[0, 0, 0]
    fy = K[0, 1, 1]
    cx = K[0, 0, 2]
    cy = K[0, 1, 2]

    X = verts_cam[:, 0]
    Y = verts_cam[:, 1]
    Z = verts_cam[:, 2]
    u = fx * X / Z + cx
    v = fy * Y / Z + cy

    H = W = img_size if isinstance(img_size, int) else img_size[0]
    W = img_size if isinstance(img_size, int) else img_size[1]

    valid = (Z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    face_valid = valid[faces].any(dim=1)
    new_faces = faces[face_valid]

    unique_indices, inverse_indices = torch.unique(new_faces, return_inverse=True)
    new_verts = verts[unique_indices]
    new_faces = inverse_indices.view(-1, 3)

    return Meshes(verts=[new_verts], faces=[new_faces])


def render_ldi_effcient(mesh, R, T, K, num_layers, img_size, device, point_priority_thres):
    '''
    Render LDI with FOV pre-filtering for efficiency.
    Returns None if the filtered mesh exceeds point_priority_thres vertices.
    '''
    if mesh.verts_packed().numel() == 0 or mesh.faces_packed().numel() == 0:
        raise ValueError("Mesh has no vertices or faces.")

    cameras = PerspectiveCameras(
        in_ndc=False,
        R=R.to(device),
        T=T.to(device),
        focal_length=K[:, 0, 0].to(device),
        principal_point=K[:, :2, 2].to(device),
        image_size=((img_size, img_size) if isinstance(img_size, int) else img_size,),
        device=device
    )

    mesh_filtered = filter_mesh_from_proj(mesh, cameras, K, img_size, device)

    verts_count = mesh_filtered.verts_packed().shape[0]
    if point_priority_thres is not None and verts_count > point_priority_thres:
        return None

    raster_settings = RasterizationSettings(
        image_size=img_size,
        blur_radius=0.0,
        faces_per_pixel=num_layers,
        perspective_correct=True
    )

    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    fragments = rasterizer(mesh_filtered)
    zbuf = fragments.zbuf  # [B, H, W, faces_per_pixel]

    empty_val = 1e10
    zbuf_clean = torch.where(zbuf >= empty_val, torch.full_like(zbuf, -1.0), zbuf)

    return {"depth": zbuf_clean.cpu().numpy()}


def save_ldi_matrix(save_path, obj_name, ldi):
    '''Save depth layers as fp16 compressed .npz file.'''
    ldi = ldi.squeeze().astype(np.float16)
    np.savez_compressed(os.path.join(save_path, '{}_ldi.npz'.format(obj_name)), ldi=ldi)


def sample_and_save_pc(mesh, npts_list, save_path):
    for num_pts in npts_list:
        sampled_points = sample_points_from_meshes(mesh, num_samples=num_pts)
        if sampled_points.shape[0] == 1:
            sampled_points = sampled_points.squeeze(0)

        points_np = sampled_points.cpu().numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_np)
        o3d.io.write_point_cloud(os.path.join(save_path, "res_{}.ply".format(num_pts)), pcd)


def save_ldi_objaverse(object_path, camera_path, num_layers, view_number):
    mesh = load_glb_as_mesh_updated(object_path, DEVICE)
    mesh = normalize_mesh(mesh)

    for view_i in range(view_number):
        camera_file_path = os.path.join(camera_path, "{:03d}.npy".format(view_i))
        R, T, K = load_camera_params_glb(camera_file_path, CAM_LENS, CAM_SENSOR_WIDTH, IMG_SIZE, DEVICE)
        ldi_data = render_ldi(mesh, R, T, K, num_layers, IMG_SIZE, DEVICE)

        if view_i == 0:
            depth_layer = ldi_data["depth"][..., 0].squeeze()
            depth_layer[depth_layer == float('inf')] = depth_layer[depth_layer != float('inf')].max()
            plt.imsave(os.path.join(camera_path, "{:03d}_ldi.png".format(view_i)), depth_layer, cmap='gray')

            if args.online_sanity_check:
                img = np.array(Image.open(os.path.join(camera_path, "000.png")))
                img_mask = (np.sum(img, axis=-1) > 100)
                ldi_mask = (depth_layer != -1)
                if np.sum(np.abs(ldi_mask.astype(float) - img_mask.astype(float))) > 5000:
                    return

        os.makedirs(camera_path, exist_ok=True)
        save_ldi_matrix(camera_path, "{:03d}".format(view_i), ldi_data["depth"])


def save_ldi_front3d(object_path, camera_path, num_layers, view_number):
    mesh = load_objs_as_meshes([object_path], device=DEVICE)

    verts = mesh.verts_packed()
    random_colors = torch.rand_like(verts)
    mesh.textures = TexturesVertex(verts_features=random_colors[None])

    for view_i in range(view_number):
        camera_file_spath = os.path.join(camera_path, "{:03d}.npy".format(view_i))
        R, T, K = load_camera_params_obj(camera_file_spath, None, CAM_SENSOR_WIDTH, IMG_SIZE, DEVICE)
        ldi_data = render_ldi(mesh, R, T, K, num_layers, IMG_SIZE, DEVICE)

        if view_i == 0:
            depth_layer = ldi_data["depth"][..., 0].squeeze()
            depth_layer[depth_layer == float('inf')] = depth_layer[depth_layer != float('inf')].max()
            plt.imsave(os.path.join(camera_path, "{:03d}_ldi.png".format(view_i)), depth_layer, cmap='gray')

        os.makedirs(camera_path, exist_ok=True)
        save_ldi_matrix(camera_path, "{:03d}".format(view_i), ldi_data["depth"])


def save_ldi_point_cloud_gso(object_path, camera_path, num_layers, view_number):
    mesh = load_objs_as_meshes([object_path], device=DEVICE)

    verts = mesh.verts_packed()
    random_colors = torch.rand_like(verts)
    mesh.textures = TexturesVertex(verts_features=random_colors[None])

    for view_i in range(view_number):
        camera_file_spath = os.path.join(camera_path, "{:03d}.npy".format(view_i))
        R, T, K = load_camera_params_obj(camera_file_spath, CAM_LENS, CAM_SENSOR_WIDTH, IMG_SIZE, DEVICE)
        ldi_data = render_ldi(mesh, R, T, K, num_layers, IMG_SIZE, DEVICE)

        if view_i == 0:
            depth_layer = ldi_data["depth"][..., 0].squeeze()
            depth_layer[depth_layer == float('inf')] = depth_layer[depth_layer != float('inf')].max()
            plt.imsave(os.path.join(camera_path, "{:03d}_ldi.png".format(view_i)), depth_layer, cmap='gray')

        os.makedirs(camera_path, exist_ok=True)
        save_ldi_matrix(camera_path, "{:03d}".format(view_i), ldi_data["depth"])

    NUM_POINT_CLOUD = [10000, 20000, 30000, 50000]
    for num_pts in NUM_POINT_CLOUD:
        sampled_points = sample_points_from_meshes(mesh, num_samples=num_pts)
        if sampled_points.shape[0] == 1:
            sampled_points = sampled_points.squeeze(0)

        torch.save(sampled_points, os.path.join(camera_path, "res_{}.pth".format(num_pts)))

        points_np = sampled_points.cpu().numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_np)
        o3d.io.write_point_cloud(os.path.join(camera_path, "res_{}.ply".format(num_pts)), pcd)


def save_ldi_point_cloud_scrream(object_path, camera_file_path, num_layers):
    mesh = scrream.load_multiple_objs(object_path)

    K = scrream.load_intrinsics(object_path)
    R, T = scrream.read_a_pose(camera_file_path)
    image_hw = scrream.get_image_resolution(object_path)

    file_id = int(os.path.basename(camera_file_path).split(".")[0])

    ldi_data = render_ldi(mesh, R, T, K, num_layers, image_hw, DEVICE)

    save_folder = os.path.join(object_path, "ldi")
    os.makedirs(save_folder, exist_ok=True)

    depth_layer = ldi_data["depth"][..., 0].squeeze()
    depth_layer[depth_layer == float('inf')] = depth_layer[depth_layer != float('inf')].max()
    plt.imsave(os.path.join(save_folder, "{:06d}_ldi.png".format(file_id)), depth_layer, cmap='gray')
    save_ldi_matrix(save_folder, "{:06d}".format(file_id), ldi_data["depth"])

    if len(glob.glob(os.path.join(save_folder, "*.ply"))) == 0:
        sample_and_save_pc(mesh, [100000, 250000, 500000], save_folder)



def save_ldi_scannetpp(object_path, img_name, num_layers, point_priority_thres):
    mesh_path = os.path.join(object_path, "scans/mesh_aligned_0.05.ply")
    mesh = scannetpp.load_scannet_ply(mesh_path)

    camera_file_path = os.path.join(object_path, "dslr/nerfstudio/transforms_2_undistorted.json")
    R, T, K, image_hw, Trans_CV = scannetpp.load_camera_params(camera_file_path, img_name)
    if R is None or T is None or K is None:
        print("low-quality:{}-{}, skip!".format(object_path, img_name))
        return

    ldi_data = render_ldi_effcient(mesh, R, T, K, num_layers, image_hw, DEVICE, point_priority_thres)
    if ldi_data is None:
        print("{}_{} has more than {} points, skipping".format(object_path, img_name, point_priority_thres))
        return

    save_folder = os.path.join(object_path, "dslr/ldi")
    os.makedirs(save_folder, exist_ok=True)

    depth_layer = ldi_data["depth"][..., 0].squeeze()
    depth_layer[depth_layer == float('inf')] = depth_layer[depth_layer != float('inf')].max()
    plt.imsave(os.path.join(save_folder, "{}_ldi.jpg".format(img_name)), depth_layer, cmap='gray')

    save_ldi_matrix(save_folder, "{}".format(img_name), ldi_data["depth"])
    np.savez(os.path.join(save_folder, "{}.npz".format(img_name)),
             Tr_w2c_cv=Trans_CV.squeeze().cpu().numpy(),
             K=K.squeeze().cpu().numpy())

    print("saved {}".format(os.path.join(save_folder, "{}".format(img_name))))

    if len(glob.glob(os.path.join(save_folder, "*.ply"))) == 0:
        sample_and_save_pc(mesh, [100000, 250000, 500000], save_folder)


if __name__ == "__main__":
    if args.dataset_type == "objaverse":
        save_ldi_objaverse(args.object_path, args.camera_path, args.num_layers, args.view_number)
    elif args.dataset_type == "front3d":
        save_ldi_front3d(args.object_path, args.camera_path, args.num_layers, args.view_number)
    elif args.dataset_type == "gso":
        save_ldi_point_cloud_gso(args.object_path, args.camera_path, args.num_layers, args.view_number)
    elif args.dataset_type == "scrream":
        save_ldi_point_cloud_scrream(args.object_path, args.camera_path, args.num_layers)
    elif args.dataset_type == "scannetpp":
        save_ldi_scannetpp(args.object_path, args.camera_path, args.num_layers, args.point_priority_thres)   
    else:
        raise NotImplementedError()
point_priority_thres
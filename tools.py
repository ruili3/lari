import argparse
import os
import torch
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from PIL import Image
from src.utils.vis import (
    prob_to_mask,
    colorize,
    denormalize,
)
import numpy as np
from src.lari.model import LaRIModel, DinoSegModel
from rembg import remove
from plyfile import PlyData, PlyElement
import torchvision.transforms as transforms



LAYER_COLOR = [
    [255, 190, 11],  # FFFF0B
    [251, 86, 7],  # FB5607
    [241, 91, 181],  # F15BB5
    [131, 56, 236],  # 8338EC
    [58, 134, 255],  # 3A86FF
]


OPENGL = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])




def save_point_cloud(pcd, rgb, filename, binary=True):
    """Save an RGB point cloud as a PLY file.
    :paras
        @pcd: Nx3 matrix, the XYZ coordinates
        @rgb: Nx3 matrix, the rgb colors for each 3D point
    """

    if rgb is None:
        gray_concat = np.tile(np.array([128], dtype=np.uint8),
                              (pcd.shape[0], 3))
        points_3d = np.hstack((pcd, gray_concat))
    else:
        assert pcd.shape[0] == rgb.shape[0]
        points_3d = np.hstack((pcd, rgb))
    python_types = (float, float, float, int, int, int)
    npy_types = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'),
                 ('green', 'u1'), ('blue', 'u1')]
    if binary is True:
        # Format into Numpy structured array
        vertices = []
        for row_idx in range(points_3d.shape[0]):
            cur_point = points_3d[row_idx]
            vertices.append(
                tuple(
                    dtype(point)
                    for dtype, point in zip(python_types, cur_point)))
        vertices_array = np.array(vertices, dtype=npy_types)
        el = PlyElement.describe(vertices_array, 'vertex')

        # write
        PlyData([el]).write(filename)
    else:
        x = np.squeeze(points_3d[:, 0])
        y = np.squeeze(points_3d[:, 1])
        z = np.squeeze(points_3d[:, 2])
        r = np.squeeze(points_3d[:, 3])
        g = np.squeeze(points_3d[:, 4])
        b = np.squeeze(points_3d[:, 5])

        ply_head = 'ply\n' \
                    'format ascii 1.0\n' \
                    'element vertex %d\n' \
                    'property float x\n' \
                    'property float y\n' \
                    'property float z\n' \
                    'property uchar red\n' \
                    'property uchar green\n' \
                    'property uchar blue\n' \
                    'end_header' % r.shape[0]
        # ---- Save ply data to disk
        np.savetxt(filename, np.column_stack[x, y, z, r, g, b], fmt='%f %f %f %d %d %d', header=ply_head, comments='')


def load_model(model_info, ckpt_path, device):
    model = eval(model_info)
    model.to(device)
    model.eval()

    # Load pretrained weights
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    return model


def process_image_custom(pil_image, resolution=512):
    """
    Read an image, resize the long side to `resolution` and pad the short side with gray,
    so that the final image is (resolution x resolution).

    Returns:
       padded_img (PIL.Image): The processed image.
       crop_coords (tuple): (top, left, bottom, right) coordinates of the valid region.
       original_size (tuple): (width, height) of the original image.
    """
    pil_image = pil_image.convert("RGB")
    original_width, original_height = pil_image.size

    # If already at fixed resolution, no processing is needed.
    if original_width == resolution and original_height == resolution:
        crop_coords = (0, 0, resolution, resolution)
        return pil_image, crop_coords, (original_width, original_height), pil_image

    # Compute scaling factor based on the long side.
    if original_width >= original_height:
        # Width is the long side.
        scale = resolution / float(original_width)
        new_width = resolution
        new_height = int(round(original_height * scale))
        resized_img = pil_image.resize((new_width, new_height), Image.BILINEAR)
        # Compute vertical padding.
        pad_top = (resolution - new_height) // 2
        pad_bottom = resolution - new_height - pad_top
        pad_left, pad_right = 0, 0
    else:
        # Height is the long side.
        scale = resolution / float(original_height)
        new_height = resolution
        new_width = int(round(original_width * scale))
        resized_img = pil_image.resize((new_width, new_height), Image.BILINEAR)
        # Compute horizontal padding.
        pad_left = (resolution - new_width) // 2
        pad_right = resolution - new_width - pad_left
        pad_top, pad_bottom = 0, 0

    # Create new image filled with black
    padded_img = Image.new("RGB", (resolution, resolution), (0, 0, 0))
    padded_img.paste(resized_img, (pad_left, pad_top))

    # The valid region (crop) is where the resized image was pasted.
    crop_coords = (pad_top, pad_left, pad_top + new_height, pad_left + new_width)
    return padded_img, crop_coords, (original_width, original_height), pil_image


def process_image(pil_image, resolution=512):
    """
    Process the image: apply custom resize/pad then convert to normalized tensor.

    Returns:
       img_tensor (torch.Tensor): Tensor of shape (1, 3, resolution, resolution).
       crop_coords (tuple): (top, left, bottom, right) coordinates of the valid region.
       original_size (tuple): (width, height) of the original image.
    """
    padded_img, crop_coords, original_size, ori_img = process_image_custom(
        pil_image, resolution
    )
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    img_tensor = transform(padded_img).unsqueeze(0)
    ori_img_tensor = transform(ori_img).unsqueeze(0)
    return img_tensor, ori_img_tensor, crop_coords, original_size


def post_process_output(input_tensor, crop_coords, original_size):
    """
    Crop the input tensor using the crop_coords and then resize to the original image size.

    Args:
       input_tensor (torch.Tensor): Input with shape (H, W, L, C) where C is 1 or 3.
       crop_coords (tuple): (top, left, bottom, right) coordinates for cropping.
       original_size (tuple): (width, height) of the original image.

    Returns:
       processed_output (torch.Tensor): Output with shape (original_height, original_width, L, C).
    """
    top, left, bottom, right = crop_coords
    # Crop the input spatially: resulting shape (crop_h, crop_w, L, C)
    cropped = input_tensor[top:bottom, left:right, ...]
    crop_h, crop_w, L, C = cropped.shape

    # New shape becomes (1, L * C, crop_h, crop_w)
    reshaped = cropped.permute(2, 3, 0, 1).reshape(1, L * C, crop_h, crop_w)

    # Unpack the original size (width, height) and use bilinear interpolation.
    new_width, new_height = original_size
    mode = "nearest" if L == 1 else "bilinear"

    resized = F.interpolate(
        reshaped, size=(new_height, new_width), mode=mode, align_corners=False
    )
    resized = resized.reshape(L, C, new_height, new_width)

    # Permute to the output shape: (new_height, new_width, L, C)
    processed_output = resized.permute(2, 3, 0, 1)

    return processed_output


def get_masked_depth(lari_map, valid_mask, layer_id):

    layer_id = max(0, layer_id)

    lari_depth = lari_map[:, :, layer_id, 2].cpu().numpy()  # H W
    valid_mask = valid_mask[:, :, layer_id, 0].cpu().numpy()  # H W
    valid_values = lari_depth[valid_mask]

    # Handle empty valid values
    if valid_values.size == 0:
        vis_depth_range = [0, 1]
    else:
        vis_depth_range = [valid_values.min(), valid_values.max()]

    depth_image = Image.fromarray(
        colorize(
            lari_depth,
            vis_depth_range[0],
            vis_depth_range[1],
            invalid_mask=~valid_mask,
            cmap="Spectral",
        )
    ).convert("RGB")
    return depth_image


def save_to_glb(pts3d, color3d, path):
    scene = trimesh.Scene()
    pct = trimesh.PointCloud(pts3d, colors=color3d)
    scene.add_geometry(pct)
    rot_y = np.eye(4)
    rot_y[:3, :3] = Rotation.from_euler("y", np.deg2rad(180)).as_matrix()
    scene.apply_transform(np.linalg.inv(OPENGL @ rot_y))
    outfile = os.path.join(path, "res.glb")
    scene.export(file_obj=outfile)
    return outfile


def get_point_cloud(pred, img, mask, first_layer_color="image", target_folder=None):
    """
    pred h w l 3 - the point cloud
    img: 3 h w - the colored image
    mask: h w l  - indicating the valid layers
    n_samples: int - n of pts to sample and save
    """

    ori_shape = pred.shape
    pred = pred.cpu().numpy()
    pred = pred.reshape(-1, 3)  # M 3

    color_palette = LAYER_COLOR[: min(len(LAYER_COLOR), ori_shape[-2])]
    assert first_layer_color in ["image", "pseudo"]

    # assign color to point clouds: [M,3] -> [M, 6]
    img = torch.clip(denormalize(img).squeeze(0), 0.0, 1.0)
    img = img.permute(1, 2, 0).unsqueeze(2).cpu().numpy()  # H W 1 3
    img = (img * 255.0).astype(np.uint8)

    layered_color = np.array([[color_palette]]).astype(np.uint8)  # 1 1 n_layer 3
    layered_color = np.broadcast_to(
        layered_color, (img.shape[0], img.shape[1], ori_shape[2], 3)
    )  # H W n_layer 3

    if first_layer_color == "image":
        layered_color[:, :, :1, :] = img
    layered_color = layered_color.reshape(-1, 3)

    valid_mask_arr = mask.squeeze().reshape(-1).cpu().numpy()  # [H,W,layers] -> [M]
    pred = pred[valid_mask_arr.astype(bool)]
    layered_color = layered_color[valid_mask_arr.astype(bool)]  # V,3

    save_folder = target_folder if target_folder is not None else os.path.dirname(__file__)

    ply_path = os.path.join(save_folder, "res.ply")
    save_point_cloud(pred, layered_color, filename=ply_path)

    glb_path = save_to_glb(pred, layered_color, save_folder)

    return glb_path, ply_path


def removebg_crop(pil_input):
    
    pil_input = remove(pil_input.convert("RGB"))

    pil_np = np.array(pil_input)
    alpha = pil_np[:, :, 3]
    is_crop = (
        False
        if np.sum(alpha > 0.8 * 255) > 0.1 * (alpha.shape[0] * alpha.shape[1])
        else True
    )

    # adjust object size to fit the image resolution
    if is_crop:
        width, height = pil_input.size
        # adjust object size
        output_np = np.array(pil_input)
        alpha = output_np[:, :, 3]
        bbox = np.argwhere(alpha > 0.8 * 255)
        bbox = (
            np.min(bbox[:, 1]),
            np.min(bbox[:, 0]),
            np.max(bbox[:, 1]),
            np.max(bbox[:, 0]),
        )
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        size = int(size * 1.5)
        bbox = (
            max(center[0] - size // 2, 0),
            max(center[1] - size // 2, 0),
            min(center[0] + size // 2, width),
            min(center[1] + size // 2, height),
        )
        pil_input = pil_input.crop(bbox)  # type: ignore

    return pil_input
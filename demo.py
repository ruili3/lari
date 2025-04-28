import argparse
import os
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from src.utils.vis import prob_to_mask
from huggingface_hub import hf_hub_download
from tools import load_model, process_image, post_process_output, get_masked_depth, get_point_cloud, removebg_crop

parser = argparse.ArgumentParser("Arguments for deploying a LaRI Demo")
parser.add_argument(
    "--image_path",
    type=str,
    default="assets/cole_hardware.png",
    help="input image name",
)

parser.add_argument(
    "--output_path",
    type=str,
    default="./results",
    help="path to save the image",
)

parser.add_argument(
    "--model_info_pm",
    type=str,
    default="LaRIModel(use_pretrained = 'moge_full', num_output_layer = 5, head_type = 'point')",
    help="Network parameters to load the model",
)

parser.add_argument(
    "--model_info_mask",
    type=str,
    default="DinoSegModel(use_pretrained = 'dinov2', dim_proj = 256, pretrained_path = '', num_output_layer = 4, output_type = 'ray_stop')",
    help="Network parameters to load the model",
)

parser.add_argument(
    "--ckpt_path_pm",
    type=str,
    default="lari_obj_16k_pointmap.pth",
    help="Path to pre-trained weights",
)

parser.add_argument(
    "--ckpt_path_mask",
    type=str,
    default="lari_obj_16k_seg.pth",
    help="Path to pre-trained weights",
)

parser.add_argument(
    "--resolution", type=int, default=512, help="Default model resolution"
)

parser.add_argument(
    "--is_remove_background", action="store_true", help="Automatically remove the background."
)

args = parser.parse_args()






device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

# === Load the model

model_path_pm = hf_hub_download(repo_id="ruili3/LaRI", filename=args.ckpt_path_pm, repo_type="model")
model_path_mask = hf_hub_download(repo_id="ruili3/LaRI", filename=args.ckpt_path_mask, repo_type="model")
# Load the model with pretrained weights.
model_pm = load_model(args.model_info_pm, model_path_pm, device)
model_mask = (
    load_model(args.model_info_mask, model_path_mask, device)
    if args.model_info_mask is not None
    else None
)

# === Image pre-processing
pil_input = Image.open(args.image_path)
if args.is_remove_background:
    pil_input = removebg_crop(pil_input) # remove background
input_tensor, ori_img_tensor, crop_coords, original_size = process_image(
    pil_input, resolution=512) # crop & resize to fit the model input size
input_tensor = input_tensor.to(device)


# === Run inference
with torch.no_grad():
    # lari map
    pred_dict = model_pm(input_tensor)
    lari_map = -pred_dict["pts3d"].squeeze(
        0
    )
    # mask
    if model_mask:
        pred_dict = model_mask(input_tensor)
        assert "seg_prob" in pred_dict
        valid_mask = prob_to_mask(pred_dict["seg_prob"].squeeze(0))  # H W L 1
    else:
        h, w, l, _ = lari_map.shape
        valid_mask = torch.new_ones((h, w, l, 1), device=lari_map.device)

# === crop & resize back to the original resolution
if original_size[0] != args.resolution or original_size[1] != args.resolution:
    lari_map = post_process_output(lari_map, crop_coords, original_size)  # H W L 3
    valid_mask = post_process_output(
        valid_mask.float(), crop_coords, original_size
    ).bool()  # H W L 1

max_n_layer = min(valid_mask.shape[-2], lari_map.shape[-2])
valid_mask = valid_mask[:, :, :max_n_layer, :]
lari_map = lari_map[:, :, :max_n_layer, :]


# === save output
os.makedirs(args.output_path, exist_ok=True)

for layer_id in range(max_n_layer):
    depth_pil = get_masked_depth(
        lari_map=lari_map, valid_mask=valid_mask, layer_id=layer_id
    )
    depth_pil.save(os.path.join(args.output_path, f"layered_depth_{layer_id}.jpg"))


# point cloud
glb_path, ply_path = get_point_cloud(
    lari_map, ori_img_tensor, valid_mask, first_layer_color="pseudo",
    target_folder=args.output_path
)

print("All results saved to `{}`.".format(args.output_path))
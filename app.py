import argparse
import gradio
import torch
import torch.backends.cudnn as cudnn
from src.utils.vis import prob_to_mask
from src.lari.model import LaRIModel, DinoSegModel
from tools import load_model, process_image, post_process_output, get_masked_depth, save_to_glb, get_point_cloud, removebg_crop
from huggingface_hub import hf_hub_download

parser = argparse.ArgumentParser("Arguments for deploying a LaRI Demo")

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
args = parser.parse_args()



def model_forward(pil_input, layered_id, rembg_checkbox):
    """
    Perform LaRI estimation by:
    1. image processing
    2. network forward
    3. save masked layered depth image
    4. save point cloud
    """
    if pil_input is None:
        return (None, None, None, None, None, None)

    if rembg_checkbox:
        pil_input = removebg_crop(pil_input)

    # Process the input image.
    input_tensor, ori_img_tensor, crop_coords, original_size = process_image(
        pil_input, resolution=512
    )
    input_tensor = input_tensor.to(device)

    # Run inference.
    with torch.no_grad():
        # lari map
        pred_dict = model_pm(input_tensor)
        lari_map = -pred_dict["pts3d"].squeeze(
            0
        )  # Expected output shape: (H_reso, W_reso, L, 3)
        # mask
        if model_mask:
            pred_dict = model_mask(input_tensor)
            assert "seg_prob" in pred_dict
            valid_mask = prob_to_mask(pred_dict["seg_prob"].squeeze(0))  # H W L 1
        else:
            h, w, l, _ = lari_map.shape
            valid_mask = torch.new_ones((h, w, l, 1), device=lari_map.device)

    # crop & resize the output to the original resolution.
    if original_size[0] != args.resolution or original_size[1] != args.resolution:
        lari_map = post_process_output(lari_map, crop_coords, original_size)  # H W L 3
        valid_mask = post_process_output(
            valid_mask.float(), crop_coords, original_size
        ).bool()  # H W L 1

    max_n_layer = min(valid_mask.shape[-2], lari_map.shape[-2])
    valid_mask = valid_mask[:, :, :max_n_layer, :]
    lari_map = lari_map[:, :, :max_n_layer, :]

    curr_layer_id = min(max_n_layer - 1, layered_id - 1)

    # masked depth list
    depth_image = get_masked_depth(
        lari_map=lari_map, valid_mask=valid_mask, layer_id=curr_layer_id
    )
    # point cloud
    glb_path, ply_path = get_point_cloud(
        lari_map, ori_img_tensor, valid_mask, first_layer_color="pseudo"
    )

    return (
        depth_image,
        glb_path,
        lari_map,
        valid_mask,
        0,
        max_n_layer - 1,
        glb_path,
        ply_path,
        pil_input,
    )


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True


# Download the file
model_path_pm = hf_hub_download(repo_id="ruili3/LaRI", filename=args.ckpt_path_pm, repo_type="model")
model_path_mask = hf_hub_download(repo_id="ruili3/LaRI", filename=args.ckpt_path_mask, repo_type="model")


# Load the model with pretrained weights.
model_pm = load_model(args.model_info_pm, model_path_pm, device)
model_mask = (
    load_model(args.model_info_mask, model_path_mask, device)
    if args.model_info_mask is not None
    else None
)


def change_layer(slider_layer_id, lari_map, valid_mask, min_layer_id, max_layer_id):

    if lari_map is None:
        return

    slider_layer_id = slider_layer_id - 1
    curr_layer_id = min(slider_layer_id, max_layer_id)
    curr_layer_id = max(curr_layer_id, min_layer_id)

    # masked depth list
    depth_image = get_masked_depth(
        lari_map=lari_map, valid_mask=valid_mask, layer_id=curr_layer_id
    )

    return depth_image


def clear_everything():
    return (
        gradio.update(value=None),
        gradio.update(value=None),
        gradio.update(value=None),
        gradio.update(value=None),
        gradio.update(value=None),
        gradio.update(value=None),
        gradio.update(value=None),
    )


with gradio.Blocks(
    css=""".gradio-container {margin: 0 !important; min-width: 100%};""",
    title="LaRI Demo",
) as demo:

    gradio.HTML(
        """
        <h1 style="text-align: center; font-size: 28px; font-weight: bold; margin-bottom: 1em;">
            LaRI: Layered Ray Intersections for Single-view 3D Geometric Reasoning
        </h1>
        """
    )
    gradio.HTML(
        """
        <p style="font-size: 16px; line-height: 1.6;">
        This is the official demo of Layered Ray Intersections 
        (<a href="https://ruili3.github.io/lari/index.html" target="_blank" style="color: #42aaf5;">LaRI</a>). 
        This demo currently supports object-level reconstruction only.
        </p>

        <h3 style="color: #42aaf5;">Get Started</h3>
        <p style="font-size: 16px; line-height: 1.6;">
        As a quick start, click one image from `Examples` and press 'Process'. Try it out with your own images by following these steps:
        <ul>
            <li>Load an image</li>
            <li>(Optional) Check the 'Remove Background' box</li>
            <li>Click the 'Process' button</li>
            <li>Explore the layered depth maps (z-channel of the LaRI point map) by adjusting the 'Layer ID' slider</li>
        </ul>
        </p>

        <p style="font-size: 16px; line-height: 1.6;">
        In the '3D Point Cloud' view, different colors represent different intersection layers:  
        <span style="color: #FFBD1C;">Layer 1</span>,  
        <span style="color: #FB5607;">Layer 2</span>,  
        <span style="color: #F15BB5;">Layer 3</span>,  
        <span style="color: #8338EC;">Layer 4</span>.
        </p>

        <h3 style="color: #42aaf5;">Contact</h3>
        <p style="font-size: 16px; line-height: 1.6;">
        If you have any questions, feel free to open an issue on our 
        <a href="https://github.com/ruili3/lari" target="_blank" style="color: #42aaf5;">GitHub repository</a> ⭐
        </p>
        """
    )



    # , <b style="color: #3A86FF;">layer 5</b>.
    lari_map = gradio.State(None)
    valid_mask = gradio.State(None)
    min_layer_id = gradio.State(None)
    max_layer_id = gradio.State(None)

    with gradio.Column():
        with gradio.Row(equal_height=True):
            with gradio.Column(scale=1):
                image_input = gradio.Image(
                    label="Upload an Image", type="pil", height=350
                )
                with gradio.Row():
                    rembg_checkbox = gradio.Checkbox(label="Remove background")
                    clear_button = gradio.Button("Clear")
                    submit_btn = gradio.Button("Process")
            with gradio.Column(scale=1):
                depth_output = gradio.Image(
                    label="LaRI Map at Z-axis (depth)",
                    type="pil",
                    interactive=False,
                    height=300,
                )
                slider_layer_id = gradio.Slider(
                    minimum=1,
                    maximum=4,
                    step=1,
                    value=1,
                    label="Layer ID",
                    interactive=True,
                )

        with gradio.Row(scale=1):
            outmodel = gradio.Model3D(
                label="3D Point Cloud (Color denotes different layers)",
                interactive=False,
                zoom_speed=0.5,
                pan_speed=0.5,
                height=450,
            )

    with gradio.Row():
        ply_file_output = gradio.File(label="ply output", elem_classes="small-file")
        glb_file_output = gradio.File(label="glb output", elem_classes="small-file")

    submit_btn.click(
        fn=model_forward,
        inputs=[image_input, slider_layer_id, rembg_checkbox],
        outputs=[
            depth_output,
            outmodel,
            lari_map,
            valid_mask,
            min_layer_id,
            max_layer_id,
            glb_file_output,
            ply_file_output,
            image_input,
        ],
    )

    clear_button.click(
        fn=clear_everything,
        outputs=[
            lari_map,
            valid_mask,
            min_layer_id,
            max_layer_id,
            image_input,
            depth_output,
            outmodel,
        ],
    )

    slider_layer_id.change(
        fn=change_layer,
        inputs=[slider_layer_id, lari_map, valid_mask, min_layer_id, max_layer_id],
        outputs=depth_output,
    )

    gradio.Examples(examples=["./assets/cole_hardware.png",
                              "./assets/3m_tape.png",
                              "./assets/horse.png",
                              "./assets/rhino.png",
                              "./assets/alphabet.png",
                              "./assets/martin_wedge.png",
                              "./assets/d_rose.png",
                              "./assets/ace.png",
                              "./assets/bifidus.png",
                              "./assets/fem.png", 
                              ], 
                              inputs=image_input)


demo.launch(share=False)

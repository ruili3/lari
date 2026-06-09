<div align="center">
<h2><span class="lari_name">LaRI</span>: Layered Ray Intersections for Single-view 3D Geometric Reasoning</h2>

[**Rui Li**](https://ruili3.github.io/)<sup>1</sup> · [**Biao Zhang**](https://1zb.github.io/)<sup>1</sup> · [**Zhenyu Li**](https://zhyever.github.io/)<sup>1</sup> · [**Federico Tombari**](https://federicotombari.github.io/)<sup>2,3</sup> · [**Peter Wonka**](https://peterwonka.net/)<sup>2,3</sup>  

<sup>1</sup>KAUST · <sup>2</sup>Google · <sup>3</sup>Technical University of Munich

**ICML 2026**

<a href="https://arxiv.org/abs/2504.18424"><img src='https://img.shields.io/badge/arXiv-KYN-red' alt='Paper PDF'></a>
<a href='https://ruili3.github.io/lari/index.html'><img src='https://img.shields.io/badge/Project_Page-LaRI-green' alt='Project Page'></a>
<a href='https://huggingface.co/spaces/ruili3/LaRI'><img src='https://img.shields.io/badge/Hugging_Face-LaRI-yellow' alt='Hugging Face'></a>
</div>

> **LaRI** is a **single-feed-forward** method that models **unseen 3D geometry** using layered point maps. It enables complete, efficient, and view-aligned geometric reasoning from a single image.



<p align="center">
  <img src="assets/teaser.jpg" alt="teaser" width="95%">
</p>


## 📋 TODO List
- [x] Inference code & Gradio demo
- [x] Evaluation data & code
- [x] Training data & code
- [x] Release the GT generation code


## 🛠️ Environment Setup
1. Create the conda environment and install required libraries:
```bash
conda create -n lari python=3.10 -y
conda activate lari
pip install -r requirements.txt
```
2. Install Pytorch3D following these [instructions](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md).


## 🚀 Quick Start
We currently provide the object-level model at our HuggingFace [Model Hub](https://huggingface.co/ruili3/LaRI/tree/main). Try the examples or use your own images with the methods below:
### Gradio Demo

Launch the Gradio interface locally:

```bash
python app.py
```

Or try it online via [HuggingFace Demo](https://huggingface.co/spaces/ruili3/LaRI).

### Command Line

Run object-level modeling with:

```bash
python demo.py --image_path assets/cole_hardware.png
```

> The input image path is specified via `--image_path`. Set `--is_remove_background` to remove the background. Layered depth maps and the 3D model will be saved in the `./results` directory by default.



## 📊 Evaluation
### Pre-trained weights and Evaluation Data
| Scene Type | Pre-trained Weights | Evaluation Data |
|----------|----------|----------|
| Object-level    | [checkpoint](https://huggingface.co/ruili3/LaRI/resolve/main/lari_obj_16k_pointmap.pth?download=true)    | Google Scanned Objects ([data](https://huggingface.co/datasets/ruili3/LaRI_dataset/resolve/main/eval/eval_gso.zip?download=true))   |
| Scene-level    | [checkpoint](https://huggingface.co/ruili3/LaRI/resolve/main/lari_scene_pointmap.pth?download=true)    | SCCREAM ([data](https://huggingface.co/datasets/ruili3/LaRI_dataset/resolve/main/eval/eval_scrream.zip?download=true))    |

Download the pre-trained weights and unzip the evaluation data.

### Object-level Evaluation
```sh
./scripts/eval_object.sh
```

### Scene-level Evaluation
```sh
./scripts/eval_scene.sh
```

NOTE: For both object and scene evaluation, set `data_path` and `test_list_path` to the customized absolute paths, set `--pretrained` to your model checkpoint path, and set `--output_dir` to specify where to store the evaluation results.



## 💻 Training
### 💾 Dataset setup
#### 1. Objaverse (object-level)
Download the processed Objaverse [dataset](https://huggingface.co/datasets/ruili3/LaRI_dataset/tree/main/train/objaverse), extract all files (`objaverse_chunk_<ID>.tar.gz`) into the target folder, for example:
```sh
mkdir ./datasets/objaverse_16k
tar -zxvf  ./objaverse_chunk_<ID>.tar.gz -C ./datasets/objaverse_16k
```

#### 2. 3D-FRONT (scene-level)
Download the processed 3D-FRONT [dataset](https://huggingface.co/datasets/ruili3/LaRI_dataset/tree/main/train/3dfront), extract all files to the target folder. For example:
```sh
mkdir ./datasets/3dfront
tar -zxvf  ./front3d_chunk_<ID>.tar.gz -C ./datasets/3dfront
```



#### 3. ScanNet++ (scene-level)
- Download the ScanNet++ [dataset](https://kaldir.vc.in.tum.de/scannetpp/), as well as the ScanNet++ [toolbox](https://github.com/scannetpp/scannetpp).
- Copy the `.yml` configuration files to the ScanNet++ toolbox folder, for example:
```sh
cd /path/to/lari
cp -r ./scripts/scannetpp_proc/*.yml /path/to/scannetpp/scannetpp/dslr/configs
``` 
- Run the following command in the ScanNet++ toolbox folder to downscale and undistort the data.
```sh
cd /path/to/scannetpp
# downscale the images
python -m dslr.downscale dslr/configs/downscale_lari.yml
# undistort the images
python -m dslr.undistort dslr/configs/undistort_lari.yml
```
- Download the ScanNet++ annotation from [here](https://huggingface.co/datasets/ruili3/LaRI_dataset/tree/main/train/scannetpp) and extract it to the `data` subfolder of your ScanNet++ path, for example
```sh
tar -zxvf  ./scannetpp_48k_annotation.tar.gz -C ./datasets/scannetpp_v2/data
```


### 🔥 Train the model
Download MoGe pre-trained [weights](https://huggingface.co/Ruicheng/moge-vitl/resolve/main/model.pt?download=true). For training with object-level data (Objaverse), run
```sh
./scripts/train_object.sh
```
For training with scene-level data (3D-FRONT and ScanNet++), run
```sh
./scripts/train_scene.sh
```
For both training settings, set `data_path`, `train_list_path` and `test_list_path` of each dataset to your customized absolute paths, set `pretrained_path` to the downloaded MoGe weights path, set `--output_dir` and `--wandb_dir` to specify where to store the evaluation results.


## ✨ Acknowledgement
This prject is largely based on [DUSt3R](https://github.com/naver/dust3r), with some model weights and functions from [MoGe](https://github.com/microsoft/moge), [Zero-1-to-3](https://github.com/cvlab-columbia/zero123), and [Marigold](https://github.com/prs-eth/Marigold). Many thanks to these awesome projects for their contributions.

## 📰 Citation
Please cite our paper if you find it helpful:
```
@inproceedings{li2026lari,
  title={LaRI: Layered Ray Intersections for Single-view 3D Geometric Reasoning},
  author={Li, Rui and Zhang, Biao and Li, Zhenyu and Tombari, Federico and Wonka, Peter},
  booktitle={Proceedings of the 43rd International Conference on Machine Learning},
  series={Proceedings of Machine Learning Research},
  publisher={PMLR},
  year={2026}
}
```
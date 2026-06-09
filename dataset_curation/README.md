# LDI Dataset Curation

Scripts for rendering **Layered Depth Images (LDIs)** (and, for some datasets,
point clouds) from 3D objects and scenes. Each dataset has a distributed driver
(`ldi_distributed_*.py`) that spreads work across CPU groups / workers and calls
the shared per-object renderer [`ldi_render_per_object.py`](ldi_render_per_object.py).

Supported datasets: **GSO**, **Objaverse**, **3D-FRONT**, **SCRREAM**, **ScanNet++**.

---

## 1. Installation

```bash
# core rendering deps
pip install open3d trimesh matplotlib pillow numpy psutil torch

# pytorch3d
# https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md
pip install "git+https://github.com/facebookresearch/pytorch3d.git"

# blenderproc — only needed for 3D-FRONT
pip install blenderproc
```

**Blender** is required for the
datasets that render RGB images first (**GSO** and **Objaverse**). Download
Blender 4.2.x and point `--blender_path` at the extracted directory (the one
containing the `blender` executable):

```
/path/to/blender-4.2.0-linux-x64/
└── blender          # executable invoked as {blender_path}/blender
```

| Dataset    | Blender | blenderproc | pytorch3d | Notes                          |
|------------|:-------:|:-----------:|:---------:|--------------------------------|
| GSO        | ✅      | —           | ✅        | renders RGB then LDI + PC      |
| Objaverse  | ✅      | —           | ✅        | renders RGB then LDI           |
| 3D-FRONT   | —       | ✅          | ✅        | blenderproc manages Blender    |
| SCRREAM    | —       | —           | ✅        | LDI/PC only, uses given poses  |
| ScanNet++  | —       | —           | ✅        | LDI only, uses given poses     |

---

## 2. How to run

Below, replace `/path/to/datasets` with your dataset root, `/path/to/output`
with where results should be written, and `/path/to/blender-4.2.0-linux-x64`
with your Blender install. The `--object_path_file` (data list) paths are given
**relative to this directory** so the commands can be run from
`code/lari/dataset_curation`.

Common knobs:
- `--render_first_n_scenes N` — render only the first `N` entries (use `-1` for all; a small value is handy for a smoke test).
- `--render_timeout SECONDS` — kill any single render that exceeds this wall-clock budget.
- `--num_cpu_for_each_process N` — CPUs per worker; workers are formed by chunking the allocated CPU affinity into groups of this size.
- `--online_sanity_check 1` — abort a view early if the first LDI is inconsistent with the rendered image.

### GSO

```bash
python ldi_distributed_gso.py \
    --input_models_path /path/to/datasets/gso/files \
    --object_path_file gso/gso_object_list.json \
    --save_res_path /path/to/output/gso \
    --num_images_per_ele 2 \
    --ele_angles 0 30 60 \
    --azimuths_offset_angle 5 \
    --camera_dist_low 1.6 \
    --camera_dist_high 2.2 \
    --render_first_n_scenes -1 \
    --render_timeout 2000 \
    --blender_path /path/to/blender-4.2.0-linux-x64
```

### Objaverse

```bash
python ldi_distributed_objaverse.py \
    --input_models_path /path/to/datasets/objaverse/objaverse_object_lgm \
    --object_path_file objaverse/objaverse_lgm16K_list.json \
    --save_res_path /path/to/output/objaverse \
    --num_renders 2 \
    --ldi_layers 15 \
    --render_first_n_scenes -1 \
    --online_sanity_check 1 \
    --blender_path /path/to/blender-4.2.0-linux-x64
```

### 3D-FRONT

First download the CC0 textures used for wall/floor/ceiling materials:

```bash
blenderproc download cc_textures /path/to/datasets/3d_front/cc_texture
```

Then render. Note 3D-FRONT is driven by `blenderproc run` (no `--blender_path`)
and takes the 3D-FRONT asset folders explicitly; `--cam_len_low/--cam_len_high`
are required:

```bash
python ldi_distributed_front3d.py \
    --object_path_file front3d/front3d_roomfloor_list.json \
    --save_res_path /path/to/output/3d_front_render \
    --front /path/to/datasets/3d_front/3D-FRONT \
    --future_folder /path/to/datasets/3d_front/3D-FUTURE-model \
    --front_3D_texture_path /path/to/datasets/3d_front/3D-FRONT-texture \
    --cc_material_path /path/to/datasets/3d_front/cc_texture \
    --cam_len_low 25 --cam_len_high 35 \
    --num_renders 12 \
    --ldi_layers 10 \
    --render_first_n_scenes -1
```

### SCRREAM

No Blender needed — LDIs/point clouds are rendered directly from the provided
meshes and camera poses:

```bash
python ldi_distributed_scrream.py \
    --input_models_path /path/to/datasets/scrream/scrream \
    --object_path_file scrream/scrream_list.json \
    --render_first_n_scenes -1 \
    --render_timeout 2000 \
    --num_cpu_for_each_process 30
```

### ScanNet++

No Blender needed. `--point_priority_thres` skips views whose in-frustum mesh
exceeds the given vertex count (keeps per-view cost bounded):

```bash
python ldi_distributed_scannetpp.py \
    --input_models_path /path/to/datasets/scannetpp_v2/data \
    --object_path_file scannetpp/scannetpp_48k_list.json \
    --render_first_n_scenes -1 \
    --render_timeout 2000 \
    --num_cpu_for_each_process 30 \
    --point_priority_thres 10000
```

---

## 3. Directory structures

### Data lists (`--object_path_file`)

These JSON files ship in this repo, one subfolder per dataset:

| File                                      | Type   | Entry example                                  |
|-------------------------------------------|--------|------------------------------------------------|
| `gso/gso_object_list.json`                | list   | `"Mens_Mako_Canoe_Moc_..."` (object name)      |
| `objaverse/objaverse_lgm16K_list.json`                  | dict   | `"<uid>": "glbs/000-047/<uid>.glb"`            |
| `front3d/front3d_roomfloor_list.json`   | list   | `"<house_id> <room_id>"`                       |
| `scrream/scrream_list.json`           | list   | `"scene01/scene01_full_00 35"` (scene + frame) |
| `scannetpp/scannetpp_48k_list.json`            | list   | `"00777c41d4 DSC00860"` (scene + image id)     |

### Datasets on disk (`--input_models_path`)

All of the following live under `/media/.../research/datasets` (everything
except 3D-FRONT, whose raw archives live under `front3d_ori`).

```
datasets/
├── gso/files/                          # --input_models_path for GSO
│   └── <object_name>/
│       ├── meshes/model.obj            # mesh loaded for rendering
│       ├── materials/
│       └── metadata.pbtxt
│
├── objaverse/objaverse_object_lgm/     # --input_models_path for Objaverse
│   └── glbs/
│       └── 000-000/<uid>.glb           # path comes from the data-list dict
│
├── scrream/scrream/                    # --input_models_path for SCRREAM
│   └── scene01/
│       ├── meshes/*.obj                # objects composing the scene
│       └── scene01_full_00/
│           ├── camera_pose/000035.txt  # per-frame pose ({frame:06d}.txt)
│           ├── intrinsics.txt
│           ├── rgb/  depth_gt/  ...
│           └── ldi/                    # ← output (LDI .png/.npz + sampled .ply)
│
├── scannetpp_v2/data/                  # --input_models_path for ScanNet++
│   └── <scene_id>/                     # e.g. 00777c41d4
│       ├── scans/mesh_aligned_0.05.ply # mesh loaded for rendering
│       └── dslr/
│           ├── nerfstudio/transforms_2_undistorted.json  # camera params
│           └── ldi/                    # ← output (LDI .jpg/.npz + .npz cam + .ply)
│
└── 3d_front/                           # (extracted from front3d_ori archives)
    ├── 3D-FRONT/                       # --front  (per-house scene .json files)
    ├── 3D-FUTURE-model/                # --future_folder
    ├── 3D-FRONT-texture/               # --front_3D_texture_path
    └── cc_texture/                     # --cc_material_path (blenderproc download)
```

### Outputs

- **GSO / Objaverse / 3D-FRONT** write to `--save_res_path`, one folder per
  object/scene, containing per-view `{idx:03d}_ldi.png`, `{idx:03d}_ldi.npz`
  LDI depth layers, camera `.npy`, and (GSO) sampled point-cloud `.ply`/`.pth`.
- **SCRREAM / ScanNet++** write LDIs back **into the scene folder** (`ldi/` and
  `dslr/ldi/` respectively): an `_ldi.png`/`_ldi.jpg` preview, an `_ldi.npz`
  depth-layer matrix, and sampled point-cloud `.ply` files (rendered once per scene).




"""Blender script to render images of 3D models from GSO. [modified from Zero123]

Different from Objaverse rendering, this script corrects the axis for GSO and
saves a rescaled .obj for downstream point-cloud sampling / LDI rendering.

Example usage:
    blender -b -P zero123_gso_script.py -- \
        --object_path my_object.obj \
        --output_dir ./views \
        --num_images_per_ele 8 \
        --ele_angles 0 30 60
"""

import argparse
import math
import os
import random
import sys

import numpy as np
import bpy
from mathutils import Vector, Matrix

parser = argparse.ArgumentParser()
parser.add_argument("--object_path", type=str, required=True, help="Path to the object file")
parser.add_argument("--output_dir", type=str, default="./output")
parser.add_argument("--engine", type=str, default="CYCLES", choices=["CYCLES", "BLENDER_EEVEE"])
parser.add_argument("--ele_angles", type=float, nargs="+", default=[0, 30, 60])
parser.add_argument("--num_images_per_ele", type=int, default=8)
parser.add_argument("--camera_dist_low", type=float, default=1.5)
parser.add_argument("--camera_dist_high", type=float, default=2.0)
parser.add_argument("--only_northern_hemisphere", type=int, default=0)
parser.add_argument("--azimuths_offset_angle", type=int, default=0)

argv = sys.argv[sys.argv.index("--") + 1:]
args = parser.parse_args(argv)

context = bpy.context
scene = context.scene
render = scene.render

cam = scene.objects["Camera"]
cam.location = (0, 1.2, 0)
cam.data.lens = 35
cam.data.sensor_width = 32

cam_constraint = cam.constraints.new(type="TRACK_TO")
cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
cam_constraint.up_axis = "UP_Y"

# setup lighting
bpy.ops.object.light_add(type="AREA")
light2 = bpy.data.lights["Area"]
light2.energy = 3000
bpy.data.objects["Area"].location[2] = 0.5
bpy.data.objects["Area"].scale[0] = 100
bpy.data.objects["Area"].scale[1] = 100
bpy.data.objects["Area"].scale[2] = 100

render.engine = args.engine
render.image_settings.file_format = "PNG"
render.image_settings.color_mode = "RGBA"
render.resolution_x = 512
render.resolution_y = 512
render.resolution_percentage = 100

scene.cycles.device = "GPU"
scene.cycles.samples = 128
scene.cycles.diffuse_bounces = 1
scene.cycles.glossy_bounces = 1
scene.cycles.transparent_max_bounces = 3
scene.cycles.transmission_bounces = 3
scene.cycles.filter_width = 0.01
scene.cycles.use_denoising = True
scene.render.film_transparent = True

bpy.context.preferences.addons["cycles"].preferences.get_devices()
bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"


def set_eval_camera(phi=None, theta=None, r=1.0):
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)

    # only positive z
    if args.only_northern_hemisphere:
        z = abs(z)

    camera = bpy.data.objects["Camera"]
    camera.location = x, y, z

    direction = -camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()
    return camera


def randomize_lighting() -> None:
    light2.energy = random.uniform(300, 600)
    bpy.data.objects["Area"].location[0] = random.uniform(-1., 1.)
    bpy.data.objects["Area"].location[1] = random.uniform(-1., 1.)
    bpy.data.objects["Area"].location[2] = random.uniform(0.5, 1.5)


def reset_scene() -> None:
    """Resets the scene to a clean state."""
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


def load_object(object_path: str) -> None:
    """Loads a model into the scene."""
    if object_path.endswith(".glb"):
        bpy.ops.import_scene.gltf(filepath=object_path, merge_vertices=True)
    elif object_path.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=object_path)
    elif object_path.endswith(".obj"):
        # NOTE GSO shares Blender's coordinate setting (Z up, -Y forward), whereas a
        # typical .obj uses Y up and -Z forward; the axes below hold only for GSO.
        bpy.ops.wm.obj_import(filepath=object_path, forward_axis='NEGATIVE_Y', up_axis='Z')
    else:
        raise ValueError(f"Unsupported file type: {object_path}")


def scene_bbox(single_obj=None, ignore_matrix=False):
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)


def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj


# from https://github.com/panmari/stanford-shapenet-renderer/blob/master/render_blender.py
def get_3x4_RT_matrix_from_blender(cam):
    # use matrix_world to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()
    T_world2bcam = -1 * R_world2bcam @ location

    RT = Matrix((
        R_world2bcam[0][:] + (T_world2bcam[0],),
        R_world2bcam[1][:] + (T_world2bcam[1],),
        R_world2bcam[2][:] + (T_world2bcam[2],)
    ))
    return RT


def normalize_scene():
    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)
    for obj in scene_root_objects():
        obj.scale = obj.scale * scale
    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    for obj in scene_root_objects():
        obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")


def save_rescaled_obj(output_path):
    """Save the scene as a .obj after applying scale/translation transforms."""
    bpy.ops.object.mode_set(mode='OBJECT')

    # select only mesh objects, then bake their transforms
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            obj.select_set(True)
    bpy.ops.object.transform_apply(location=True, scale=True)

    bpy.ops.object.select_all(action='DESELECT')

    # export under the .obj default coordinate (Blender 4.0+ uses wm.obj_export)
    bpy.ops.wm.obj_export(filepath=output_path, forward_axis='NEGATIVE_Z', up_axis='Y')
    print(f"Saved rescaled OBJ to {output_path}")


def save_images(object_file: str) -> None:
    """Saves rendered images and camera RT matrices of the object in the scene."""
    os.makedirs(args.output_dir, exist_ok=True)

    reset_scene()
    load_object(object_file)
    normalize_scene()

    # create an empty object to track
    empty = bpy.data.objects.new("Empty", None)
    scene.collection.objects.link(empty)
    cam_constraint.target = empty

    # set pre-defined elevations and azimuths
    polar_angles = []
    for ang in args.ele_angles:
        polar_angles += [90 - ang] * args.num_images_per_ele

    angle_split = [*range(0 + args.azimuths_offset_angle, 360 + args.azimuths_offset_angle, 360 // args.num_images_per_ele)]
    angle_split = [angle % 360 for angle in angle_split]  # ensure all angles are within 360 degrees

    azimuths = np.radians(angle_split * len(args.ele_angles))
    polar_angles = np.radians(polar_angles)

    randomize_lighting()
    for i in range(args.num_images_per_ele * len(args.ele_angles)):
        # sample rendering distance, then set the camera
        distance = np.random.uniform(args.camera_dist_low, args.camera_dist_high)
        camera = set_eval_camera(polar_angles[i], azimuths[i], distance)

        # render the image
        render_path = os.path.join(args.output_dir, f"{i:03d}.png")
        scene.render.filepath = render_path
        bpy.ops.render.render(write_still=True)

        # save camera RT matrix
        RT = get_3x4_RT_matrix_from_blender(camera)
        RT_path = os.path.join(args.output_dir, f"{i:03d}.npy")
        np.save(RT_path, RT)

    save_rescaled_obj(os.path.join(args.output_dir, "res_normed.obj"))


if __name__ == "__main__":
    try:
        save_images(args.object_path)
    except Exception as e:
        print("Failed to render", args.object_path)
        print(e)

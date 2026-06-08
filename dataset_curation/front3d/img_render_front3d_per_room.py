import blenderproc as bproc
import argparse
import os
import numpy as np
import random
import sys
sys.path.append(os.path.dirname(__file__))
from utils_3dfront import invert_transformation_metrix, save_matrix
import bpy
from PIL import Image
import json

IMG_SIZE = 512
CAM_SENSOR_WIDTH = 32

parser = argparse.ArgumentParser()
parser.add_argument("--front", help="Path to the 3D front file")
parser.add_argument("--room_id", type=int, help="id of the room")
parser.add_argument("--cam_lens_range", nargs="+", type=int, help="camera focal length")
parser.add_argument("--future_folder", help="Path to the 3D Future Model folder.")
parser.add_argument("--front_3D_texture_path", help="Path to the 3D FRONT texture folder.")
parser.add_argument('--cc_material_path', nargs='?', default="./data/3d_front/cc_texture", help="Path to the CCTextures folder.")
parser.add_argument('--num_rendering', type=int, default=12)
parser.add_argument("--output_dir", nargs='?', default="./output", help="Path to where the data should be saved")
args = parser.parse_args()

if not os.path.exists(args.front) or not os.path.exists(args.future_folder):
    raise Exception("One of the two folders does not exist!")




def is_img_valid(img, valid_thres=0.3):
    '''
    check if image has enough valid regions
    
    img: numpy in [h w 3]
    '''
    total_area = img.shape[0] * img.shape[1] #  h*w
    invalid_area = (np.sum(img, axis=-1) < 10).sum() # black regions

    return False if (invalid_area/total_area) > valid_thres else True




def select_room_from_scene(loaded_objects, json_path, room_id):
    '''
    select mesh objects that belongs to one room by specifying the room ids

    loaded_objects: object mesh loaded by blenderproc
    obj_dict: object info loaded from .json file
    room_id: specify the room to be loaded

    '''
    with open(json_path, "r") as f:
        obj_dict = json.load(f)

    room_children = obj_dict["scene"]["room"][room_id]["children"]
    # for selecting furnitures
    room_children_uid = [child["ref"] for child in room_children]

    selected_model = []
    for obj in loaded_objects:
        assert "uid" in obj.get_all_cps().keys() # this applies for <patched> blenderproc code
        if obj.get_cp("uid") in room_children_uid:
            obj_name = obj.get_name().lower()
            if obj_name.split(".")[0] not in ["wallbottom", "walltop", "wallouter"]: # , "baseboard"
                selected_model.append(obj)

    return selected_model



def set_intrinsic_with_rand_lens(cam_len_range):

    assert cam_len_range[0] <= cam_len_range[1]

    cam_len = random.randint(cam_len_range[0], cam_len_range[1])

    # set the intrinsics
    focal = (cam_len / CAM_SENSOR_WIDTH) * IMG_SIZE
    K = np.array([
        [focal, 0., IMG_SIZE/2],
        [0., focal, IMG_SIZE/2],
        [0., 0., 1.]
    ])
    bproc.camera.set_intrinsics_from_K_matrix(K, IMG_SIZE, IMG_SIZE)
    
    return cam_len



def blender_remove_unselected_objects(loaded_objects):
    # Deselect all objects in Blender's scene first
    for obj in bpy.data.objects:
        obj.select_set(False)

    # Select only the objects you want to export
    for obj in loaded_objects:
        # Find the corresponding Blender object by name
        blender_obj = bpy.data.objects.get(obj.get_name())
        if blender_obj:
            blender_obj.select_set(True)

    # Step 1: Delete all unselected objects except cameras
    for obj in bpy.data.objects:
        # Skip cameras or keep selected objects
        if obj.type in {'CAMERA', 'LIGHT', 'EMPTY'} or obj.select_get():
            continue
        # Remove other unselected objects
        bpy.data.objects.remove(obj, do_unlink=True)

    # Step 2: Remove orphaned meshes (optional cleanup)
    for mesh in bpy.data.meshes:
        if mesh.users == 0:  # Only remove meshes not used by any object
            bpy.data.meshes.remove(mesh)




bproc.init()
mapping_file = bproc.utility.resolve_resource(os.path.join("front_3D", "3D_front_mapping.csv"))
mapping = bproc.utility.LabelIdMapping.from_csv(mapping_file)

# set the light bounces
bproc.renderer.set_light_bounces(diffuse_bounces=200, glossy_bounces=200, max_bounces=200,
                                  transmission_bounces=200, transparent_max_bounces=200)

# load the front 3D objects
loaded_objects = bproc.loader.load_front3d(
    json_path=args.front,
    future_model_path=args.future_folder,
    front_3D_texture_path=args.front_3D_texture_path,
    label_mapping=mapping
)


loaded_objects = select_room_from_scene(loaded_objects, args.front, room_id=args.room_id)

blender_remove_unselected_objects(loaded_objects)

# Init sampler for sampling locations inside the loaded front3D house
point_sampler = bproc.sampler.Front3DPointInRoomSampler(loaded_objects)

cc_materials = bproc.loader.load_ccmaterials(args.cc_material_path, ["Bricks", "Wood", "Carpet", "Tile", "Marble"])

floors = bproc.filter.by_attr(loaded_objects, "name", "Floor.*", regex=True)
for floor in floors:
    # For each material of the object
    for i in range(len(floor.get_materials())):
        # In 95% of all cases
        if np.random.uniform(0, 1) <= 0.95:
            # Replace the material with a random one
            floor.set_material(i, random.choice(cc_materials))


baseboards_and_doors = bproc.filter.by_attr(loaded_objects, "name", "Baseboard.*|Door.*", regex=True)
wood_floor_materials = bproc.filter.by_cp(cc_materials, "asset_name", "WoodFloor.*", regex=True)
for obj in baseboards_and_doors:
    # For each material of the object
    for i in range(len(obj.get_materials())):
        # Replace the material with a random one
        obj.set_material(i, random.choice(wood_floor_materials))


walls = bproc.filter.by_attr(loaded_objects, "name", "Wall.*", regex=True)
marble_materials = bproc.filter.by_cp(cc_materials, "asset_name", "Marble.*", regex=True)
for wall in walls:
    # For each material of the object
    for i in range(len(wall.get_materials())):
        # In 50% of all cases
        if np.random.uniform(0, 1) <= 0.1:
            # Replace the material with a random one
            wall.set_material(i, random.choice(marble_materials))



# Init bvh tree containing all mesh objects
bvh_tree = bproc.object.create_bvh_tree_multi_objects([o for o in loaded_objects if isinstance(o, bproc.types.MeshObject)])

poses = 0
tries = 0

def check_name(name):
    for category_name in ["chair", "sofa", "table", "bed"]:
        if category_name in name.lower():
            return True
    return False

# filter some objects from the loaded objects, which are later used in calculating an interesting score
special_objects = [obj.get_cp("category_id") for obj in loaded_objects if check_name(obj.get_name())]


# if we do not randomly sample cameras
if (args.cam_lens_range[0] == args.cam_lens_range[1]):
    cam_len = set_intrinsic_with_rand_lens(args.cam_lens_range)

cam_matrix_list = []
cam_len_list = []
proximity_checks = {"min": 1.5, "avg": {"min": 2.0, "max": 7.5}, "no_background": False}
while tries < 10000 and poses < args.num_rendering:
    if args.cam_lens_range[0] < args.cam_lens_range[1]:
        raise Exception("current logic only supports one <K>, ie the last K in the loop, \
                        as the images are simultaneously rendered using <bproc.renderer.render()>, which relies \
                        on the last updated <K>")

    # Sample point inside house
    height = np.random.uniform(1.4, 1.8)
    location = point_sampler.sample(height)
    # Sample rotation (fix around X and Y axis)
    rotation = np.random.uniform([1.2217, 0, 0], [1.338, 0, np.pi * 2])
    cam2world_matrix = bproc.math.build_transformation_mat(location, rotation)

    coverage_standard = bproc.camera.scene_coverage_score(cam2world_matrix, special_objects, special_objects_weight=2.0) > 0.3
    obstacle_standard = bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, proximity_checks, bvh_tree)

    # keep the pose only if the view has enough coverage and no close obstacles
    if coverage_standard and obstacle_standard:
        bproc.camera.add_camera_pose(cam2world_matrix)
        # save camera matrix
        cam_matrix_list.append(invert_transformation_metrix(cam2world_matrix))
        cam_len_list.append(cam_len)
        poses += 1
    tries += 1


if poses < args.num_rendering:
    raise SystemExit


# render the whole pipeline
rgb_data = bproc.renderer.render()['colors'] # list of numpy images in [0, 255]


os.makedirs(args.output_dir, exist_ok=True)
# save each png image
for id, img in enumerate(rgb_data):
    if True: #is_img_valid(img):
        Image.fromarray(img.astype(np.uint8)).save(os.path.join(args.output_dir, "{:03d}.png".format(id)))
        save_matrix(cam_matrix_list[id], cam_len_list[id], args.output_dir, id)     

## save to .obj
bpy.ops.wm.obj_export(filepath=os.path.join(args.output_dir, "res.obj"))
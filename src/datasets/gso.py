from PIL import Image
import os
import numpy as np
import torch
import json
import open3d as o3d
from src.datasets.base.base_dataset import BaseDataset
from src.datasets.utils.transforms import ImgNorm, ColorJitter

CAM_LENS = 35
CAM_SENSOR_WIDTH = 32
OBJA_IMG_SIZE = 512
OBJA_FOCAL = CAM_LENS / CAM_SENSOR_WIDTH * OBJA_IMG_SIZE


class GSO(BaseDataset):
    def __init__(self,
                 *args,
                 data_path,
                 train_list_path,
                 test_list_path,
                 img_per_obj,
                 num_pts,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)
        '''
        Dataset for GSO, 12 image for each object
        '''
        self.data_path = data_path
        self.data_list_path_dict = {"train": train_list_path, "test":test_list_path} # key: <train> or <test>

        self.intrinsic = np.array([[OBJA_FOCAL, 0, OBJA_IMG_SIZE/2, 0],
                            [0, OBJA_FOCAL, OBJA_IMG_SIZE/2, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]
                            ], dtype=np.float32)
        self.img_per_obj = img_per_obj
        self.num_pts = num_pts
        assert self.num_pts in [10000, 20000, 30000, 50000]

        self.scene_type = "object"

        self._load_data_list()

    def _load_data_list(self):

        with open(self.data_list_path_dict[self.split], "tr") as f:
            self._data_list = json.load(f) # "BREAKFAST_MENU"
        # a pre-defined value baesd on datasets
        self._NUM_IMG_PER_OBJ = self.img_per_obj


    def __len__(self):
        return len(self._data_list) * self._NUM_IMG_PER_OBJ

    def load_camera_params_obj(self, camera_path: str, cam_lens, cam_sensor_width, img_size):
        """
        Convert the world-to-camera transformation under the Blender coordinate system to the world-to-camera transformation under the OBJ world system and the Computer Vision camera system
        """
        res = np.load(camera_path, allow_pickle=True)
        if isinstance(res, np.ndarray):
            T_b_w2cam = res
            assert cam_lens is not None, "cam_lens must be provided if not included in file."
        elif isinstance(res.item(), dict):
            res = res.item()
            # In this case, assume the focal length is stored in the file.
            cam_lens = res["cam_len"]
            T_b_w2cam = res["T_b_w2cam"]
        else:
            raise NotImplementedError("Unsupported format in camera file.")

        # Convert T_b_w2cam to a 4x4 matrix.
        T_b_w2cam = np.concatenate((T_b_w2cam, np.array([[0, 0, 0, 1]])), axis=0)  # 4x4

        R_b2obj = np.array([
            [1, 0,  0, 0],
            [0, 0,  1, 0],
            [0, -1, 0, 0],
            [0, 0,  0, 1]
        ])
        
        # transform from Blender camera convention (-Z, Y) to Computer Vision camera convention (Z, -Y)
        R_bcam_to_cvcam = np.array([[1, 0,  0, 0],
                                [0, -1,  0, 0],
                                [0, 0, -1, 0],
                                [0, 0,  0, 1]
                                ])


        # Transformations:
        # 1. Transform OBJ point cloud into Blender coordinates using the inverse of R_b2obj.
        # 2. Apply the camera transformation T_b_w2cam.
        # 3. Convert from Blender to PyTorch3D (computer vision) coordinates using R_bcam2py3d.
        T_py_w2cam = R_bcam_to_cvcam @ T_b_w2cam @ np.linalg.inv(R_b2obj)

        R = T_py_w2cam[:3, :3]  # Shape (3, 3)
        T = T_py_w2cam[:3, -1]    # Shape (3,)

        return R, T, None



    def _get_image_and_ldi(self, idx):
        # identify the <object ID> then the <image ID>
        obj_id = (idx // self._NUM_IMG_PER_OBJ)
        img_id = idx % self._NUM_IMG_PER_OBJ

        # "BREAKFAST_MENU"
        obj_path = self._data_list[obj_id]

        
        try:
            # from RGBA to RGB (black background)
            img = Image.open(os.path.join(self.data_path, obj_path, "{:03d}.png".format(img_id))).convert("RGB") 
            # slice the target layers
            ldi = np.load(os.path.join(self.data_path, obj_path, "{:03d}_ldi.npz".format(img_id)))["ldi"][:,:,:self.n_ldi_layers]
            
            # point cloud 
            # Load the .ply file using Open3D.
            pcd = o3d.io.read_point_cloud(os.path.join(self.data_path, obj_path, "res_{}.ply".format(self.num_pts)))
            pcd = np.asarray(pcd.points).astype(np.float32)

            cam_file_path = os.path.join(self.data_path, obj_path, "{:03d}.npy".format(img_id))
            R, T, _ = self.load_camera_params_obj(cam_file_path, CAM_LENS, CAM_SENSOR_WIDTH, OBJA_IMG_SIZE)

            # left-multiplication for <R>
            pcd = (R @ pcd.T).T + T

        except Exception as e:
            # Log the error and file path for debugging
            print("[ERROR] data load error at path: {}, Error: {}".format(os.path.join(self.data_path, obj_path), e))
            raise

        sample_name = "{}_{}".format(obj_path, img_id)

        return img, ldi, pcd, sample_name, self.intrinsic
from PIL import Image
import os
import numpy as np
import json
import gzip
from src.datasets.base.base_dataset import BaseDataset

CAM_LENS = 35
CAM_SENSOR_WIDTH = 32
OBJA_IMG_SIZE = 512
OBJA_FOCAL = CAM_LENS / CAM_SENSOR_WIDTH * OBJA_IMG_SIZE



class Objaverse(BaseDataset):
    def __init__(self,
                 *args,
                 data_path,
                 train_list_path,
                 test_list_path,
                 num_pts=10000,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)
        '''
        Dataset for Objaverse, 12 image for each object
        '''
        self.data_path = data_path
        self.data_list_path_dict = {"train": train_list_path, "test":test_list_path} # key: <train> or <test>
        self.num_pts = num_pts # for validation

        self.intrinsic = np.array([[OBJA_FOCAL, 0, OBJA_IMG_SIZE/2, 0],
                                   [0, OBJA_FOCAL, OBJA_IMG_SIZE/2, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]
                                    ], dtype=np.float32)
   
        self.scene_type = "object"

        self._load_data_list()

    def _load_data_list(self):

        with gzip.open(self.data_list_path_dict[self.split], "tr") as f:
            # eg, <"0b6d53e3b2d048b38af4e27d74210a6c">: <"glb/000-007/0b6d53e3b2d048b38af4e27d74210a6c.glb">
            self._data_list = json.load(f)
            self._data_list = list(self._data_list.values())

        # a pre-defined value baesd on datasets
        self._NUM_IMG_PER_OBJ = 12

    def __len__(self):
        return len(self._data_list) * self._NUM_IMG_PER_OBJ


    def _get_image_and_ldi(self, idx):
        # identify the <object ID> then the <image ID>
        obj_id = (idx // self._NUM_IMG_PER_OBJ)
        img_id = idx % self._NUM_IMG_PER_OBJ

        # eg, <"glb/000-007/0b6d53e3b2d048b38af4e27d74210a6c.glb">
        obj_path = self._data_list[obj_id].split("/")[-2:]
        obj_path = "/".join(obj_path)[:-4]

        # from RGBA to RGB (black background)
        img = Image.open(os.path.join(self.data_path, obj_path, "{:03d}.png".format(img_id))).convert("RGB") 

        try:
            ldi = np.load(os.path.join(self.data_path, obj_path, "{:03d}_ldi.npz".format(img_id)))["ldi"][:,:,:self.n_ldi_layers] 
        except Exception as e:
            print("[ERROR] LDI load error at path: {}, Error: {}".format(os.path.join(self.data_path, obj_path, "{:03d}_ldi.npz".format(img_id)), e))
            raise


        intrinsic = self.intrinsic

        return img, ldi, intrinsic
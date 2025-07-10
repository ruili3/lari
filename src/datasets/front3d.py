from PIL import Image
import os
import numpy as np
import json

from src.datasets.base.base_dataset import BaseDataset

CAM_SENSOR_WIDTH = 32
OBJA_IMG_SIZE = 512


class Front3D(BaseDataset):
    def __init__(self,
                 *args,
                 data_path,
                 train_list_path,
                 test_list_path,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)
        '''
        Dataset for Front3D, 12 image for each object
        '''
        self.data_path = data_path
        self.data_list_path_dict = {"train": train_list_path, "test":test_list_path} # key: <train> or <test>

        # the intrinsic of each sample is different
        self.intrinsic = None

        self.scene_type = "indoor"
   
        self._load_data_list()

    def _load_data_list(self):

        with open(self.data_list_path_dict[self.split], "tr") as f:
            # eg, "646caacd-2202-49a5-8aee-8461238c4121.json 4": ["Floor.003", 4]
            self._data_list = json.load(f)
            self._data_list = list(self._data_list.keys()) # 646caacd-2202-49a5-8aee-8461238c4121.json 4

        self._NUM_IMG_PER_OBJ = 6


    def __len__(self):
        return len(self._data_list) * self._NUM_IMG_PER_OBJ


    def _get_image_and_ldi(self, idx):
        obj_id = (idx // self._NUM_IMG_PER_OBJ)
        img_id = idx % self._NUM_IMG_PER_OBJ

        # "646caacd-2202-49a5-8aee-8461238c4121.json 4"
        path = self._data_list[obj_id].split(" ")
        obj_path = path[0].split(".")[0]
        room_id = int(path[1])
        obj_path = "{}_{}".format(obj_path, room_id)

        # from RGBA to RGB (black background)
        img = Image.open(os.path.join(self.data_path, obj_path, "{:03d}.png".format(img_id))).convert("RGB") 
        
        try:
            ldi = np.load(os.path.join(self.data_path, obj_path, "{:03d}_ldi.npz".format(img_id)))["ldi"][:,:,:self.n_ldi_layers] 
        except Exception as e:
            print("[ERROR] LDI load error at path: {}, Error: {}".format(os.path.join(self.data_path, obj_path, "{:03d}_ldi.npz".format(img_id)), e))
            raise

        # load intrinsic
        cam_len = np.load(os.path.join(self.data_path, obj_path, "{:03d}.npy".format(img_id)), allow_pickle=True).item()["cam_len"]
        focal_length = (cam_len * OBJA_IMG_SIZE) / CAM_SENSOR_WIDTH
        intrinsic = np.array([[focal_length, 0, OBJA_IMG_SIZE/2, 0],
                                   [0, focal_length, OBJA_IMG_SIZE/2, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]
                                    ], dtype=np.float32)

        return img, ldi, intrinsic
    


    def __getitem__(self, idx):
        datadict = super().__getitem__(idx)
        # identify the <object ID> then the <image ID>
        obj_id = (idx // self._NUM_IMG_PER_OBJ)

        # "646caacd-2202-49a5-8aee-8461238c4121.json 4"
        path = self._data_list[obj_id].split(" ")
        obj_path = path[0].split(".")[0] # remove .json
        room_id = int(path[1])
        obj_path = "{}_{}".format(obj_path, room_id)

        datadict['name'] = obj_path

        return datadict
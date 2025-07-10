from PIL import Image
import os
import numpy as np
import torch
import json

from src.datasets.base.base_dataset import BaseDataset


class ScanNetPP(BaseDataset):
    def __init__(self,
                 *args,
                 data_path,
                 train_list_path,
                 test_list_path,
                 num_pts,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)
        '''
        Dataset for ScanNetPP
        '''
        self.data_path = data_path
        self.data_list_path_dict = {"train": train_list_path, "test":test_list_path} # key: <train> or <test>

        self.intrinsic = None
        self.num_pts = num_pts
        assert self.num_pts in [100000, 250000, 500000]
        
        self.scene_type = "indoor"

        self._load_data_list()

    def _load_data_list(self):
        # a list containing test sample path and image id
        with open(self.data_list_path_dict[self.split], "tr") as f:
            self._data_list = json.load(f)


    def __len__(self):
        return len(self._data_list)


    def _get_image_and_ldi(self, idx):

        # eg, "00777c41d4 DSC00920"
        item = self._data_list[idx].split(" ")
        obj_path, img_id = item

        try:
            # from RGBA to RGB (black background)
            img = Image.open(os.path.join(self.data_path, obj_path, "dslr/downscaled_undistorted_images", "{}.JPG".format(img_id))).convert("RGB") 
            # slice the target layers
            ldi = np.load(os.path.join(self.data_path, obj_path, "dslr/ldi", "{}_ldi.npz".format(img_id)))["ldi"][:,:,:self.n_ldi_layers]
            
            cam_params = np.load(os.path.join(self.data_path, obj_path, "dslr/ldi", "{}.npz".format(img_id)))
            intrinsics = cam_params["K"]
            intrinsics_4x4 = np.zeros((4,4)).astype(np.float32)
            intrinsics_4x4[:3,:3] = intrinsics
            intrinsics_4x4[3,3] = 1.0

        except Exception as e:
            print("[ERROR] data load error at path: {}, Error: {}".format(os.path.join(self.data_path, obj_path), e))
            raise

        return img, ldi, intrinsics_4x4
    

    def __getitem__(self, idx):
        datadict = super().__getitem__(idx)
        # eg, "00777c41d4 DSC00920"
        item = self._data_list[idx].split(" ")
        obj_path, img_id = item
        datadict['name'] = "{}_{}".format(obj_path, img_id)
        return datadict
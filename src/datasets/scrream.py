from PIL import Image
import os
import numpy as np
import torch
import json
from src.datasets.base.base_dataset import BaseDataset
from src.datasets.utils.transforms import ImgNorm, ColorJitter


class SCRREAM(BaseDataset):
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
        Dataset for SCRREAM
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
            self._data_list = json.load(f) # "BREAKFAST_MENU"


    def __len__(self):
        return len(self._data_list)


    def _load_intrinsics(self, scene_folder, output_tensor=True):
        """
        Loads the camera intrinsic matrix from intrinsics.txt as a PyTorch tensor.
        """
        intrinsics_file = os.path.join(scene_folder, "intrinsics.txt")

        if not os.path.exists(intrinsics_file):
            raise FileNotFoundError(f"Error: {intrinsics_file} does not exist!")

        # Read and parse the intrinsics matrix
        intrinsics = []
        with open(intrinsics_file, "r") as file:
            for line in file:
                row = list(map(float, line.strip().split()))
                intrinsics.append(row)

        if output_tensor:
            # Convert to a PyTorch tensor and reshape to (1, 3, 3)
            intrinsics_tensor = torch.tensor(intrinsics, dtype=torch.float32).unsqueeze(0)
            return intrinsics_tensor
        else:
            intrinsics = np.array(intrinsics).astype(np.float32) # 3, 3
            return intrinsics


    def _get_image_and_ldi(self, idx):

        # eg, "scene07/scene07_reduced_00 590"
        item = self._data_list[idx].split(" ")
        obj_path, img_id = item
        img_id = int(img_id)

        try:
            # from RGBA to RGB (black background)
            img = Image.open(os.path.join(self.data_path, obj_path, "rgb", "{:06d}.png".format(img_id))).convert("RGB") 
            # slice the target layers
            ldi = np.load(os.path.join(self.data_path, obj_path, "ldi", "{:06d}_ldi.npz".format(img_id)))["ldi"][:,:,:self.n_ldi_layers]
            
            intrinsics_path = os.path.join(self.data_path, obj_path)
            intrinsics = self._load_intrinsics(intrinsics_path, output_tensor=False)

        except Exception as e:
            print("[ERROR] data load error at path: {}, Error: {}".format(os.path.join(self.data_path, obj_path), e))
            raise

        sample_name = "{} {}".format(obj_path, img_id)

        return img, ldi, None, sample_name, intrinsics
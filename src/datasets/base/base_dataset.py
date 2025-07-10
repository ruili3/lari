import numpy as np
import torch


from src.datasets.base.easy_dataset import EasyDataset
from src.datasets.utils.transforms import ImgNorm
from src.utils.geometry import ldi_to_pts3d 
from src.datasets.utils.morphological_operation import morphological_close_cpu, morphological_open_cpu
from PIL import Image



class BaseDataset(EasyDataset):
    def __init__(self,
                 n_ldi_layers = 5,
                 split=None,
                 resolution=None,  # square_size or (width, height) or list of [(width,height), ...]
                 transform=ImgNorm,
                 aug_crop=False,
                 seed=None,
                 invalid_layer_pix_ratio = 0.03,
                 resize_mode = None,
                 refine_valid_mask = None,
                 complete_mask_first_layer = False,
                 enforce_img_reso_for_eval = None,
                 train_crop_range_h = None,
                 train_crop_range_w = None,
                 do_not_save_behind_for_eval = False
                 ):
        super().__init__()

        self.n_ldi_layers = n_ldi_layers
        self.split = split
        self.invalid_layer_pix_ratio = invalid_layer_pix_ratio

        assert isinstance(resolution, int) or isinstance(resolution, tuple)
        self.resolution = resolution if isinstance(resolution, tuple) else (resolution, resolution)

        self.transform = transform
        if isinstance(transform, str):
            transform = eval(transform)

        self.aug_crop = aug_crop
        self.seed = seed

        self.resize_mode = resize_mode

        self.refine_valid_mask = refine_valid_mask
        assert self.refine_valid_mask in [None, "morph_open", "morph_close"]

        # during eval, enforce image to be fixed resolution (eg, 512,512) for network feed forward
        self.enforce_img_reso_for_eval = enforce_img_reso_for_eval 


        # some dataset's (eg, FRONT3D) GT has undesired missing depth in the first layer, leading to inaccurate
        # mask annotation. We thereby force the first layer to be all-valid via this argument
        self.complete_mask_first_layer = complete_mask_first_layer
        
        self.intrinsic = None
        self.train_crop_range_h = train_crop_range_h
        self.train_crop_range_w = train_crop_range_w
        self.do_not_save_behind_for_eval = do_not_save_behind_for_eval

        self.scene_type = None
    

    def _get_image_and_ldi(self, idx):
        raise NotImplementedError()


    def train_random_crop(self, image, ldi, h_crop_size, w_crop_size, intrinsics):
        """
        Center crops the input image and layered depth image (ldi) by removing a total number
        of pixels specified by h_crop_size and w_crop_size from the height and width, respectively.
        It then updates the camera intrinsics accordingly.
        """
        # Get original dimensions (PIL returns size as (width, height))
        original_width, original_height = image.size

        if w_crop_size >= original_width or h_crop_size >= original_height:
            raise ValueError("Crop margins must be smaller than the image dimensions.")

        # Calculate crop margins on each side.
        left = w_crop_size // 2
        right_margin = w_crop_size - left

        top = h_crop_size // 2
        bottom_margin = h_crop_size - top

        # Define the crop box for center cropping.
        left_coord = left
        top_coord = top
        right_coord = original_width - right_margin
        bottom_coord = original_height - bottom_margin

        cropped_image = image.crop((left_coord, top_coord, right_coord, bottom_coord))
        cropped_ldi = ldi[top_coord:bottom_coord, left_coord:right_coord, :]

        # Update the camera intrinsics by shifting the principal point by the crop offsets.
        updated_intrinsics = intrinsics.copy()
        updated_intrinsics[0, 2] -= left_coord  # Adjust x-coordinate.
        updated_intrinsics[1, 2] -= top_coord   # Adjust y-coordinate.

        return cropped_image, cropped_ldi, updated_intrinsics



    def resize_and_completion(self, image, ldi, intrinsic, target_w, target_h):
        ''' 
        Given an image (in PIL Image format), a numpy array 'ldi' with shape (H, W, L),
        and an intrinsic matrix (3x3), resize all while maintaining aspect ratio,
        then complete the short side using a gray color for the image and -1 for 'ldi'.
        Adjust intrinsic matrix accordingly.
        '''

        w, h = image.size
        
        # Compute new size while maintaining aspect ratio
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize image while maintaining aspect ratio
        image = image.resize((new_w, new_h), Image.BICUBIC)

        new_image = Image.new("RGB", (target_w, target_h), (0, 0, 0))
        
        # Paste resized image onto the center of the new image
        paste_x = (target_w - new_w) // 2
        paste_y = (target_h - new_h) // 2
        new_image.paste(image, (paste_x, paste_y))
        
        if ldi is not None and intrinsic is not None:
            # Convert ldi to a PyTorch tensor
            ldi_tensor = torch.tensor(ldi, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # (1, L, H, W)
            
            # Resize ldi using PyTorch interpolation
            ldi_resized = torch.nn.functional.interpolate(ldi_tensor, size=(new_h, new_w), mode='nearest')
            # Create a new ldi tensor with the required resolution and fill missing values with -1
            new_ldi = torch.full((1, ldi.shape[2], target_h, target_w), -1, dtype=torch.float32)
            # Paste resized ldi onto the center of the new ldi tensor
            new_ldi[:, :, paste_y:paste_y + new_h, paste_x:paste_x + new_w] = ldi_resized

            # Convert back to numpy
            new_ldi = new_ldi.squeeze(0).permute(1, 2, 0).numpy()
            # mask = mask.squeeze(0).permute(1, 2, 0).numpy()

        
            # Adjust intrinsic matrix
            intrinsic[0, :] *= scale  # Scale focal length and principal point in x direction
            intrinsic[1, :] *= scale  # Scale focal length and principal point in y direction
            intrinsic[0, 2] += paste_x  # Adjust principal point for x-axis padding
            intrinsic[1, 2] += paste_y  # Adjust principal point for y-axis padding
            
            return new_image, new_ldi, intrinsic
        else:
            return new_image, None, None




    def filter_out_invalid_layers(self, valid_mask):
        '''
        to filter out layers with extremely small valid areas (such as smaller than 3% of that of the first layer)
        by marking the layered mask to zero
        '''
        area_first_layer = valid_mask[:,:,0].sum()
        area = np.sum(np.reshape(valid_mask, (-1, valid_mask.shape[-1])), axis=0) # L) 

        valid_layer_index = (area > self.invalid_layer_pix_ratio * area_first_layer)[None, None, ...] # 1 1 L
        res = (valid_mask * valid_layer_index).astype(bool) # B H L
        return res 



    def random_hw_crop_size(self, image):
        """
        Randomly generate crop sizes (height and width) within given ranges,
        ensuring the crop dimensions do not exceed the image dimensions.
        """
        # Get the dimensions of the image (PIL gives size as (width, height))
        image_width, image_height = image.size

        # Unpack the crop ranges
        assert self.train_crop_range_h is not None and self.train_crop_range_w is not None
        w_min, w_max = self.train_crop_range_w
        h_min, h_max = self.train_crop_range_h

        # Clamp the bounds to the image dimensions
        w_min = min(w_min, image_width)
        h_min = min(h_min, image_height)
        w_max = min(w_max, image_width)
        h_max = min(h_max, image_height)

        if w_min > w_max:
            w_crop_size = image_width
        else:
            w_crop_size = self._rng.integers(w_min, w_max + 1)

        if h_min > h_max:
            h_crop_size = image_height
        else:
            h_crop_size = self._rng.integers(h_min, h_max + 1)

        return h_crop_size, w_crop_size


    def adjust_resolution(self, image_ori, ldi, intrinsic):
        if self.resize_mode == "eval":
            image, ldi, intrinsic = self.resize_and_completion(image_ori, ldi, intrinsic, self.resolution[0], self.resolution[1])
            if isinstance(self.enforce_img_reso_for_eval, list):
                # For our evaluation, only resize image into fixed resolution (eg, 512x512) while keeping gt the original resolution
                image, _, _ = self.resize_and_completion(image_ori, None, None, 
                                                            self.enforce_img_reso_for_eval[0], self.enforce_img_reso_for_eval[1])

        elif self.resize_mode == "train":
            # first, randomly crop image 
            h_crop_size, w_crop_size = self.random_hw_crop_size(image_ori)
            image, ldi, intrinsic = self.train_random_crop(image_ori, 
                                                            ldi, 
                                                            h_crop_size, 
                                                            w_crop_size,
                                                            intrinsic)
            # then, resize image and complement it with gray area
            image, ldi, intrinsic = self.resize_and_completion(image, ldi, intrinsic, 
                                                                self.resolution[0], self.resolution[1])

        return image, ldi, intrinsic


    def sample_gt_point_cloud(self, pts3d, valid_mask, data):
        # sample the complete point cloud set
        valid_pts3d = pts3d[valid_mask] # N, 3
        n_valid_pts3d = valid_pts3d.shape[0]

        if n_valid_pts3d > self.num_pts:
            perm = torch.randperm(n_valid_pts3d)
            sampled_points = valid_pts3d[perm[:self.num_pts]]
            data["pcd_eval"] = sampled_points
        else:
            perm = torch.randint(0, n_valid_pts3d, (self.num_pts,))
            data["pcd_eval"] = valid_pts3d[perm]    


        if self.scene_type == "indoor":
            # sample the point cloud from unseen layers
            valid_pts3d_behind = pts3d[:,:,1:,:][valid_mask[:,:,1:]] # N', 3
            n_vpts3d_behind = valid_pts3d_behind.shape[0]
            # sample the point cloud from the visible layer
            valid_pts3d_first = pts3d[:,:,:1,:][valid_mask[:,:,:1]] # N', 3
            n_vpts3d_first = valid_pts3d_first.shape[0]


            if n_vpts3d_behind >= self.num_pts:
                perm = torch.randperm(n_vpts3d_behind)
                data["pcd_eval_unseen"] = valid_pts3d_behind[perm[:self.num_pts]]
            else:
                perm = torch.randint(0, n_vpts3d_behind, (self.num_pts,))
                data["pcd_eval_unseen"] = valid_pts3d_behind[perm]            

            if n_vpts3d_first >= self.num_pts:
                perm = torch.randperm(n_vpts3d_first)
                data["pcd_eval_visible"] = valid_pts3d_first[perm[:self.num_pts]]
            else:
                perm = torch.randint(0, n_vpts3d_first, (self.num_pts,))
                data["pcd_eval_visible"] = valid_pts3d_first[perm]     

        return data


    def __getitem__(self, idx):

        # set-up the rng
        if self.seed:
            self._rng = np.random.default_rng(seed=self.seed + idx)
        elif not hasattr(self, '_rng'):
            seed = torch.initial_seed()
            self._rng = np.random.default_rng(seed=seed)


        results =  self._get_image_and_ldi(idx)


        if len(results) == 3: # for training datasets
            image, ldi, intrinsic = results
            pcd = name = None
        elif len(results) == 5: # for testing datasets
            image, ldi, pcd, name, intrinsic = results
        else:
            raise NotImplementedError()


        # adjust resolution for both training and testing data
        if image.size[0] != self.resolution[1] or image.size[1] != self.resolution[0]:
            image, ldi, intrinsic = self.adjust_resolution(image, ldi, intrinsic)
                

        # construct the output dict
        image = self.transform(image) # normalized torch image
        ldi = ldi.astype(np.float32)
        pts3d, valid_mask = ldi_to_pts3d(ldi, intrinsic)
        pts3d = torch.from_numpy(pts3d)
        
        valid_mask = self.filter_out_invalid_layers(valid_mask)

        valid_mask = torch.from_numpy(valid_mask)
        if self.refine_valid_mask == "morph_open":
            valid_mask = morphological_open_cpu(valid_mask, kernel_size=5, iterations=1)
        elif self.refine_valid_mask == "morph_close":
            valid_mask = morphological_close_cpu(valid_mask, kernel_size=5, iterations=1)

        # setting the 3D point coordinate of the invalid pixels to (0,0,0)
        pts3d[~valid_mask] = 0


        # setting the first layer of valid mask to all-one (for room-level datasets, eg, FRONT-3D)
        if self.complete_mask_first_layer:
            valid_mask[:,:,0] = 1
            valid_mask = valid_mask.bool()


        intrinsic = torch.from_numpy(intrinsic)
        data = {}
        data["img"] = image
        data["pts3d"] = pts3d
        data["mask"] = valid_mask.unsqueeze(-1)
        data["intrinsic"] = intrinsic


        # only for evalution
        if pcd is not None: 
            pcd = torch.from_numpy(pcd).float()
            data["pcd_eval"] = pcd # GT of the complete point cloud

        if pcd is None and self.split == "test":
            data = self.sample_gt_point_cloud(pts3d, valid_mask, data) # GT of the unseen/visible point cloud

        if name is not None:
            data["name"] = name


        return data
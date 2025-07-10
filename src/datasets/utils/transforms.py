import torchvision.transforms as tvf
import torchvision.transforms.functional as F
import random
import torch
# from dust3r.utils.image import ImgNorm
ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# define the standard image transforms
ColorJitter = tvf.Compose([tvf.ColorJitter(0.5, 0.5, 0.5, 0.1), ImgNorm])



class LightingTransform:
    def __init__(self, brightness_range=(0.3, 0.7), gamma_range=(1.3, 1.7)):
        """
        Adjusts the lighting of an image.

        To simulate low-light conditions, use:
            brightness_range < 1 (e.g., (0.3, 0.7))
            gamma_range > 1 (e.g., (1.3, 1.7))

        To simulate high-light conditions, use:
            brightness_range > 1 (e.g., (1.3, 1.7))
            gamma_range < 1 (e.g., (0.3, 0.7))

        Args:
            brightness_range (tuple): Range of brightness factors.
            gamma_range (tuple): Range of gamma correction factors.
        """
        self.brightness_range = brightness_range
        self.gamma_range = gamma_range

    def __call__(self, img):
        # Adjust brightness
        brightness_factor = random.uniform(*self.brightness_range)
        img = F.adjust_brightness(img, brightness_factor)
        # Adjust gamma
        gamma = random.uniform(*self.gamma_range)
        img = F.adjust_gamma(img, gamma)
        return img
    
class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.05):
        """
        mean: Mean of the Gaussian noise.
        std: Standard deviation of the Gaussian noise.
        """
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise
    


LightNoisyAugmentation = tvf.Compose([
    tvf.ColorJitter(brightness=0, contrast=0.2, hue=(-0.1, 0.1)),
    LightingTransform(brightness_range=(0.6, 1.1), gamma_range=(0.8, 1.2)),
    tvf.ToTensor(),
    AddGaussianNoise(mean=0.0, std=0.01),
    tvf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

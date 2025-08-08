from torch.utils.data import Dataset
from tqdm import tqdm
import os
from torchvision import transforms
import numpy as np
from PIL import Image
import torch

class HybridTSImageDataset(Dataset):

    def __init__(self, image_dir, num_patches, num_patches_input, image_size):
        self.image_names = []
        self.image_dir = image_dir
        self.visible_size = (num_patches_input * image_size // num_patches)
        # Collect all image paths in the directory and its subdirectories
        for label_name in tqdm(os.listdir(image_dir), desc="Loading image dataset from disk"):
            dir_name = os.path.join(image_dir, label_name)
            if not os.path.isdir(dir_name):
                continue
            for file_name in os.listdir(dir_name):
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    self.image_names.append(os.path.join(image_dir, label_name, file_name))
        
        self.transform_train = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )
        self.generator = np.random.default_rng(0)
    def __getitem__(self, idx: int):
        # load image
        img_name = self.image_names[self.generator.choice(len(self.image_names))]
        # print(img_name)
        image = self.transform_train(Image.open(img_name).convert('RGB')) # [3, H, W]
        image = np.array(Image.open(img_name))
        
        # image_right = torch.mean(image, dim=0, keepdim=True)[:, :, self.visible_size:] 
        # image_mask = torch.ones_like(image_right)
        # image_right = torch.cat([image_right, image_mask], dim=0)

        # image[:, :, self.visible_size:] = 0

        return {
            'target': image,
            'target_img': image,
            'y': np.array([[0.0]]),
            'x': np.array([[0.0]]),
            'pad_left': 0,
            'context_len': 0,
            'pred_len': 0,
            'periodicity': 0,
            'scale_x': 0.0,
        }
    
    def __len__(self) -> int:
        """
        Length is the number of time series multiplied by dataset_weight
        """
        return len(self.image_names)


if __name__ == "__main__":

    import time

    st = time.time()
    n = 10000
    ds = HybridTSImageDataset("/data/mouxiangchen/home/data/imagenet1k/train", 14, 7, 224)
    # ds = HybridTSImageDataset("/dev/shm/imagenet1k/train", 14, 7, 224)
    # ds = HybridTSImageDataset("/home/mouxiangchen/imagenet1k/train", 14, 7, 224)
    # ds = HybridTSImageDataset("/data/imagenet/", 14, 7, 224)
    for _ in tqdm(range(n)):
        a = ds[0]
    
    
import torch
import torch.nn
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
import os
import numpy as np
from torch.utils.data import Dataset
import logging


class custom_dataset(Dataset):
    def __init__(self, mode="train", root="datasets/smoker-detection-dataset", transforms=None):
        super().__init__()
        self.mode = mode
        self.root = root
        self.transforms = transforms if transforms else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Select split
        self.folder = os.path.join(self.root, self.mode)
        if not os.path.exists(self.folder):
            raise FileNotFoundError(f"Dataset folder '{self.folder}' does not exist.")
        
        # Initialize lists
        self.image_list = []
        self.label_list = []
        
        # Save class lists
        self.class_list = [d for d in os.listdir(self.folder) if os.path.isdir(os.path.join(self.folder, d))]
        self.class_list.sort()
        
        if not self.class_list:
            raise ValueError(f"No class folders found in '{self.folder}'.")
        
        # Load images and labels
        for class_id, class_name in enumerate(self.class_list):
            class_folder = os.path.join(self.folder, class_name)
            for image_name in os.listdir(class_folder):
                image_path = os.path.join(class_folder, image_name)
                if not image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    self.logger.warning(f"Skipping non-image file: {image_path}")
                    continue
                self.image_list.append(image_path)
                label = np.zeros(len(self.class_list), dtype=np.float32)
                label[class_id] = 1.0
                self.label_list.append(label)
        
        if not self.image_list:
            raise ValueError(f"No valid images found in '{self.folder}'.")
        
        self.logger.info(f"Loaded {len(self.image_list)} images from '{self.folder}'.")

    def __getitem__(self, index):
        image_name = self.image_list[index]
        label = self.label_list[index]
        
        try:
            image = Image.open(image_name).convert("RGB")
        except UnidentifiedImageError:
            raise RuntimeError(f"Failed to open image: {image_name}")
        
        if self.transforms:
            image = self.transforms(image)
        
        label = torch.tensor(label, dtype=torch.float32)
        
        return image, label
            
    def __len__(self):
        return len(self.image_list)
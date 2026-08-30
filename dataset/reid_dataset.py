import os
import glob
import re
from PIL import Image
import torch
from torch.utils.data import Dataset

class ReIDImageDataset(Dataset):
    """
    Standard PyTorch Dataset wrapper for Person Re-ID.
    Each item returns (img_tensor, pid, camid, viewid).
    """
    def __init__(self, dataset, transform=None):
        self.dataset = dataset  # list of tuples (img_path, pid, camid, viewid)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, viewid = self.dataset[index]
        if isinstance(img_path, torch.Tensor):
            # Synthetic tensor image
            img = img_path
        else:
            img = Image.open(img_path).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
                
        return img, pid, camid, viewid

class ReIDDataset(Dataset):
    
    def __init__(self, num_identities=64, images_per_id=8, img_size=(256, 128), num_cameras=6, transform=None):
        self.num_identities = num_identities
        self.images_per_id = images_per_id
        self.transform = transform
        
        self.dataset = []
        for pid in range(num_identities):
            for img_idx in range(images_per_id):
                camid = img_idx % num_cameras
                viewid = (img_idx // num_cameras) % 2
                img_path = f"_pid_{pid}_img_{img_idx}.jpg"
                self.dataset.append((img_path, pid, camid, viewid))

    def __len__(self):
        return len(self.dataset)

def parse_market1501(dir_path, is_train=True):
    if not os.path.exists(dir_path):
        return []
    pattern = re.compile(r'([-\d]+)_c(\d)')
    dataset = []
    for root, _, files in os.walk(dir_path):
        for img_name in files:
            if not img_name.endswith('.jpg'):
                continue
            match = pattern.search(img_name)
            if not match:
                continue
            pid, camid = map(int, match.groups())
            if pid == -1:
                continue
            dataset.append((os.path.join(root, img_name), pid, camid - 1, 0))
    return dataset

def parse_dukemtmc(dir_path, is_train=True):

    if not os.path.exists(dir_path):
        return []
    pattern = re.compile(r'([-\d]+)_c(\d)')
    dataset = []
    for root, _, files in os.walk(dir_path):
        for img_name in files:
            if not img_name.endswith('.jpg'):
                continue
            match = pattern.search(img_name)
            if not match:
                continue
            pid, camid = map(int, match.groups())
            if pid == -1:
                continue
            dataset.append((os.path.join(root, img_name), pid, camid - 1, 0))
    return dataset

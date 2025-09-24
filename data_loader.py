import os
import random
import re
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from typing import List, Tuple

class Market1501(Dataset):
    def __init__(self, data_path: str, mode: str = 'train', image_size: Tuple[int, int] = (256, 128)):
        if not os.path.isdir(data_path): raise FileNotFoundError(f"Dataset path not found: {data_path}")
        self.mode = mode
        self.transform = transforms.Compose([
            transforms.Resize(image_size), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        if mode == 'train':
            self.paths, self.pids_raw, _ = self._load_paths(os.path.join(data_path, 'bounding_box_train'))
            if not self.paths: raise RuntimeError("No training images found.")
            
            # --- Remap PIDs to 0-indexed labels using the class attribute ---
            unique_pids = sorted(list(set(self.pids_raw)))
            self.pid_to_label = {pid: label for label, pid in enumerate(unique_pids)}
            self.labels = [self.pid_to_label[pid] for pid in self.pids_raw]
            self.num_classes = len(unique_pids)
            print(f"Found {len(self.paths)} training images with {self.num_classes} unique IDs.")
            
        else: # test mode
            self.query_paths, self.query_labels, self.query_cams = self._load_paths(os.path.join(data_path, 'query'))
            self.gallery_paths, self.gallery_labels, self.gallery_cams = self._load_paths(os.path.join(data_path, 'bounding_box_test'))
            if not self.query_paths or not self.gallery_paths: raise RuntimeError("No test/gallery images found.")

    def _load_paths(self, directory: str) -> Tuple[List[str], List[int], List[int]]:
        if not os.path.isdir(directory): return [], [], []
        paths, pids, cams = [], [], []
        pattern = re.compile(r'([-\d]+)_c(\d)')
        for filename in sorted(os.listdir(directory)):
            if filename.endswith('.jpg'):
                match = pattern.search(filename)
                if match:
                    pid, cam = map(int, match.groups())
                    if pid == -1 and self.mode == 'train': continue
                    paths.append(os.path.join(directory, filename))
                    pids.append(pid)
                    cams.append(cam)
        return paths, pids, cams

    def _process(self, path: str) -> torch.Tensor:
        with Image.open(path).convert('RGB') as img: return self.transform(img)

    def sample_episode(self, ns_gallery: int = 32):
        # This method is not actively used by the current 'train_baseline.py' or 'main.py'
        # but is kept for potential future use.
        pass

    def get_full_query(self): return [self._process(p) for p in self.query_paths], self.query_labels, self.query_cams
    def get_full_gallery(self): return [self._process(p) for p in self.gallery_paths], self.gallery_labels, self.gallery_cams

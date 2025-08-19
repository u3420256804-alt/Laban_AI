import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from augment import augment_sequence, AugmentCfg

class SkeletonSeqDataset(Dataset):
    def __init__(self, files, class_map_path, max_len=300, augment=False, augment_cfg=AugmentCfg()):
        super().__init__()
        self.files = files
        self.class_map = json.load(open(class_map_path, 'r', encoding='utf-8'))
        self.max_len = max_len
        self.augment = augment
        self.augment_cfg = augment_cfg

    def __len__(self):
        return len(self.files)

    @staticmethod
    def pad_or_crop(x: np.ndarray, max_len: int) -> np.ndarray:
        T = x.shape[0]
        if T == max_len:
            return x
        if T > max_len:
            return x[:max_len]
        pad = np.zeros((max_len - T, x.shape[1], x.shape[2]), dtype=x.dtype)
        return np.concatenate([x, pad], axis=0)

    def __getitem__(self, idx):
        path = self.files[idx]
        data = np.load(path)
        seq = data['features']
        if self.augment:
            seq = augment_sequence(seq, self.augment_cfg)
        seq = self.pad_or_crop(seq, self.max_len)
        cls_name = os.path.basename(os.path.dirname(path))
        y = self.class_map[cls_name]
        mask = (seq.sum(axis=(1, 2)) != 0).astype(np.float32)
        seq = seq.reshape(seq.shape[0], -1)
        return torch.from_numpy(seq).float(), torch.tensor(y).long(), torch.from_numpy(mask).float()
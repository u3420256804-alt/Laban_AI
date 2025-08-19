import os
import glob
import json
import random

def make_splits(proc_dir: str, splits_dir: str, train: float, val: float, test: float, seed: int = 42):
    assert abs(train + val + test - 1.0) < 1e-6
    random.seed(seed)
    os.makedirs(splits_dir, exist_ok=True)

    class_map = json.load(open(os.path.join(proc_dir, 'class_map.json'), 'r', encoding='utf-8'))
    splits = {'train': [], 'val': [], 'test': []}
    for c in class_map:
        files = sorted(glob.glob(os.path.join(proc_dir, c, '*.npz')))
        random.shuffle(files)
        n = len(files)
        n_train = int(round(n * train))
        n_val = int(round(n * val))
        n_test = n - n_train - n_val
        splits['train'] += files[:n_train]
        splits['val'] += files[n_train:n_train + n_val]
        splits['test'] += files[n_train + n_val:]

    for k, lst in splits.items():
        with open(os.path.join(splits_dir, f'{k}.txt'), 'w', encoding='utf-8') as f:
            for p in lst:
                f.write(p + '\n')
    print("Zapisano splity do:", splits_dir)
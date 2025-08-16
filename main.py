#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LMA Effort Actions – end‑to‑end pipeline (v2, stronger generalization)
---------------------------------------------------------------------
Co robi ten plik:
1) Ekstrakcja szkieletów z wideo (MediaPipe Pose) i zapis do .npz
2) Budowa zbioru (train/val/test) z MOCNĄ augmentacją sekwencji
3) Modele do wyboru: BiLSTM, TCN, TransformerEncoder
4) Trening z class weights / (opcjonalnie) balanced sampler, LR scheduler, early‑stopping
5) Ewaluacja (accuracy, F1, confusion matrix)
6) Predykcja okienkami + majority vote / softmax mean; możliwość zapisu timeline (CSV)

Kluczowe usprawnienia vs v1:
- normalizacja póz (center=mid‑hip, skala=rozstaw barków; opcjonalne wyrównanie yaw barków w 2D)
- augmentacje: time‑warp, jitter pozycji, dropout stawów, left‑right flip (z mapą par stawów), random crop w czasie
- nowe modele: TransformerEncoder, ulepszony TCN; wybór --model {lstm,tcn,transformer}
- class weights (automatycznie z train split), opcjonalny --balance_sampler
- predykcja segmentowa: --segment_len, --predict_stride, --vote {mean,majority}, zapis CSV z timeline

Wymagania (pip):
- mediapipe,
- opencv-python,
- torch, torchvision, torchaudio (zgodnie z CUDA/CPU),
- numpy, scipy, scikit-learn,
- tqdm

Przykłady użycia:
- Ekstrakcja:  python main_v2.py extract --data_root ./data_raw --out_dir ./data_proc --min_conf 0.5 --fps 25
- Split:       python main_v2.py split --proc_dir ./data_proc --splits_dir ./splits --train 0.7 --val 0.15 --test 0.15 --seed 42
- Trening:     python main_v2.py train --proc_dir ./data_proc --splits_dir ./splits \
                              --model transformer --epochs 70 --batch 8 --lr 3e-4 \
                              --save_dir ./runs/exp2 --augment 1 --max_len 300 \
                              --early_stop 10 --balance_sampler 0
- Ewaluacja:   python main_v2.py eval --proc_dir ./data_proc --splits_dir ./splits --ckpt ./runs/exp2/best.pt
- Predykcja:   python main_v2.py predict --video path/to/new.mp4 --ckpt ./runs/exp2/best.pt \
                              --class_map ./data_proc/class_map.json \
                              --segment_len 120 --predict_stride 60 --vote mean --save_timeline timeline.csv
"""

import os
import cv2
import json
import math
import glob
import time
import random
import argparse
import itertools
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

# Torch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# Metrics
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# Progress
from tqdm import tqdm

# -------------------------
# 0) Konfiguracje i stałe
# -------------------------
POSE_LANDMARKS = 33  # MediaPipe Pose
LEFT_RIGHT_PAIRS = [
    (11, 12), (13, 14), (15, 16),
    (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)
]
LEFT_INDEXES = {11,13,15,23,25,27,29,31}
RIGHT_INDEXES = {12,14,16,24,26,28,30,32}

# -------------------------
# 1) Pose extraction (MediaPipe)
# -------------------------
try:
    import mediapipe as mp
except ImportError:
    mp = None
    print("[WARN] mediapipe nie jest zainstalowane – komenda 'extract' nie zadziała.")


def normalize_landmarks(landmarks: np.ndarray, align_shoulders: bool = True) -> Tuple[np.ndarray, float]:
    """Normalizacja szkieletu: przesunięcie do środka miednicy i skalowanie
    rozstawem barków; opcjonalne wyrównanie yaw (obrót w 2D tak, by barki były poziome).
    landmarks: (33,4) -> (x,y,z,vis)
    Zwraca: norm_landmarks (33,4), scale (float)
    """
    LHIP, RHIP = 23, 24
    LSH, RSH = 11, 12

    center = (landmarks[LHIP, :3] + landmarks[RHIP, :3]) / 2.0
    shifted = landmarks.copy()
    shifted[:, :3] = landmarks[:, :3] - center

    if landmarks[LSH, 3] > 0 and landmarks[RSH, 3] > 0:
        scale = np.linalg.norm(landmarks[LSH, :3] - landmarks[RSH, :3]) + 1e-6
    else:
        scale = np.linalg.norm(landmarks[LHIP, :3] - landmarks[RHIP, :3]) + 1e-6
    shifted[:, :3] /= scale

    if align_shoulders and shifted[LSH, 3] > 0 and shifted[RSH, 3] > 0:
        # Obrót w 2D (x,y), tak aby linia barków była pozioma
        dx = shifted[RSH, 0] - shifted[LSH, 0]
        dy = shifted[RSH, 1] - shifted[LSH, 1]
        angle = math.atan2(dy, dx)
        ca, sa = math.cos(-angle), math.sin(-angle)
        xy = shifted[:, :2].copy()
        x_new = ca * xy[:, 0] - sa * xy[:, 1]
        y_new = sa * xy[:, 0] + ca * xy[:, 1]
        shifted[:, 0] = x_new
        shifted[:, 1] = y_new
    return shifted, float(scale)


def extract_sequence_from_video(video_path: str, min_conf: float = 0.5, fps: int = 25,
                                static_image_mode: bool = False,
                                align_shoulders: bool = True) -> Dict:
    if mp is None:
        raise RuntimeError("mediapipe nie jest zainstalowane")

    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Nie mogę otworzyć: {video_path}")

    orig_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    frame_interval = max(int(round(orig_fps / fps)), 1)

    pose = mp_pose.Pose(static_image_mode=static_image_mode,
                        model_complexity=1,
                        smooth_landmarks=True,
                        enable_segmentation=False,
                        min_detection_confidence=min_conf,
                        min_tracking_confidence=min_conf)

    landmarks_all = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_interval != 0:
            idx += 1
            continue
        idx += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        if res.pose_landmarks is None:
            continue
        lm = res.pose_landmarks.landmark
        arr = np.zeros((POSE_LANDMARKS, 4), dtype=np.float32)
        for i in range(POSE_LANDMARKS):
            arr[i, 0] = lm[i].x
            arr[i, 1] = lm[i].y
            arr[i, 2] = lm[i].z
            arr[i, 3] = lm[i].visibility
        arr, _ = normalize_landmarks(arr, align_shoulders=align_shoulders)
        landmarks_all.append(arr)

    cap.release()
    pose.close()

    if len(landmarks_all) == 0:
        raise ValueError(f"Brak wykrytych landmarków w {video_path}")

    L = np.stack(landmarks_all, axis=0)  # (T,33,4)

    # Pochodne: prędkości (różnice między klatkami)
    dL = np.diff(L[:, :, :3], axis=0, prepend=L[:1, :, :3])  # (T,33,3)
    features = np.concatenate([L, dL], axis=-1)  # (T,33,7) -> (x,y,z,vis, dx,dy,dz)

    return {
        'landmarks': L.astype(np.float32),
        'features': features.astype(np.float32),
        'orig_fps': float(orig_fps),
        'used_fps': float(fps)
    }


def extract_dir(data_root: str, out_dir: str, min_conf: float = 0.5, fps: int = 25):
    os.makedirs(out_dir, exist_ok=True)
    classes = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])
    class_map = {c: i for i, c in enumerate(classes)}
    with open(os.path.join(out_dir, 'class_map.json'), 'w', encoding='utf-8') as f:
        json.dump(class_map, f, ensure_ascii=False, indent=2)

    for c in classes:
        in_dir = os.path.join(data_root, c)
        out_c = os.path.join(out_dir, c)
        os.makedirs(out_c, exist_ok=True)
        videos = sorted(sum([glob.glob(os.path.join(in_dir, ext)) for ext in ('*.mp4','*.mov','*.avi')], []))
        print(f"[CLS {c}] {len(videos)} plików")
        for v in tqdm(videos, desc=f"Extract {c}"):
            base = os.path.splitext(os.path.basename(v))[0]
            dst = os.path.join(out_c, base + '.npz')
            if os.path.exists(dst):
                continue
            try:
                seq = extract_sequence_from_video(v, min_conf=min_conf, fps=fps)
                np.savez_compressed(dst, **seq)
            except Exception as e:
                print("[WARN] pomijam", v, "=>", e)

# -------------------------
# 2) Splity train/val/test
# -------------------------

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

# -------------------------
# 3) Augmentacje sekwencji i dataset
# -------------------------
@dataclass
class AugmentCfg:
    time_warp_prob: float = 0.5
    time_warp_strength: float = 0.3  # resample +/-30%
    jitter_prob: float = 0.6
    jitter_sigma: float = 0.015      # standard dev dla x,y,z po normalizacji
    dropout_prob: float = 0.5
    dropout_rate: float = 0.1        # procent punktów do wyzerowania
    flip_prob: float = 0.5           # losowe lustrzane odbicie L<->R
    random_crop_prob: float = 0.5
    crop_ratio_min: float = 0.6      # min proporcja długości po cropie


def time_resample(seq: np.ndarray, scale: float) -> np.ndarray:
    T = seq.shape[0]
    new_T = max(int(round(T * scale)), 2)
    xs = np.linspace(0, 1, T)
    xs_new = np.linspace(0, 1, new_T)
    out = np.empty((new_T, seq.shape[1], seq.shape[2]), dtype=seq.dtype)
    for j in range(seq.shape[1]):
        for k in range(seq.shape[2]):
            out[:, j, k] = np.interp(xs_new, xs, seq[:, j, k])
    return out


def flip_lr(seq: np.ndarray) -> np.ndarray:
    """Lustrzane odbicie w osi X oraz zamiana indeksów L<->R."""
    s = seq.copy()
    s[:, :, 0] *= -1.0  # x -> -x
    for l, r in LEFT_RIGHT_PAIRS:
        s[:, [l, r], :] = s[:, [r, l], :]
    return s


def random_time_crop(seq: np.ndarray, ratio_min: float) -> np.ndarray:
    T = seq.shape[0]
    keep = max(int(T * max(ratio_min, 0.05)), 2)
    if keep >= T:
        return seq
    st = random.randint(0, T - keep)
    return seq[st:st+keep]


def augment_sequence(seq: np.ndarray, cfg: AugmentCfg) -> np.ndarray:
    s = seq.copy()
    if random.random() < cfg.random_crop_prob:
        s = random_time_crop(s, cfg.crop_ratio_min)
    if random.random() < cfg.time_warp_prob:
        scale = 1.0 + random.uniform(-cfg.time_warp_strength, cfg.time_warp_strength)
        s = time_resample(s, scale)
    if random.random() < cfg.flip_prob:
        s = flip_lr(s)
    if random.random() < cfg.jitter_prob:
        s[:, :, :3] += np.random.normal(0, cfg.jitter_sigma, size=s[:, :, :3].shape)
    if random.random() < cfg.dropout_prob:
        mask = np.random.rand(*s[:, :, :3].shape) < cfg.dropout_rate
        s[:, :, :3][mask] = 0.0
    return s


class SkeletonSeqDataset(Dataset):
    def __init__(self, files: List[str], class_map_path: str,
                 max_len: int = 300, augment: bool = False, augment_cfg: AugmentCfg = AugmentCfg()):
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
            st = (T - max_len) // 2
            return x[st:st + max_len]
        pad = np.zeros((max_len - T, x.shape[1], x.shape[2]), dtype=x.dtype)
        return np.concatenate([x, pad], axis=0)

    def __getitem__(self, idx):
        path = self.files[idx]
        data = np.load(path)
        seq = data['features']  # (T,33,7)
        if self.augment:
            seq = augment_sequence(seq, self.augment_cfg)
        seq = self.pad_or_crop(seq, self.max_len)
        cls_name = os.path.basename(os.path.dirname(path))
        y = self.class_map[cls_name]
        mask = (seq.sum(axis=(1, 2)) != 0).astype(np.float32)
        seq = seq.reshape(seq.shape[0], -1)
        return torch.from_numpy(seq).float(), torch.tensor(y).long(), torch.from_numpy(mask).float()

# -------------------------
# 4) Modele: BiLSTM, TCN, Transformer
# -------------------------
class BiLSTMClassifier(nn.Module):
    def __init__(self, input_size: int, hidden: int, num_layers: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden, num_layers=num_layers,
                            bidirectional=True, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 2, num_classes)

    def forward(self, x, mask=None):  # x: (B,T,F)
        out, _ = self.lstm(x)
        if mask is not None:
            mask = mask.unsqueeze(-1)
            out = (out * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-6)
        else:
            out = out.mean(dim=1)
        out = self.dropout(out)
        return self.fc(out)


class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, d=1, p=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=d*(k-1)//2, dilation=d)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=k, padding=d*(k-1)//2, dilation=d)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(p)
        self.down = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):  # (B,F,T)
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.drop(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.drop(y)
        return self.relu(y + self.down(x))


class TCNClassifier(nn.Module):
    def __init__(self, input_size: int, num_classes: int, channels: List[int] = [256, 256, 256],
                 kernel: int = 3, dropout: float = 0.2):
        super().__init__()
        layers = []
        in_ch = input_size
        d = 1
        for ch in channels:
            layers.append(TemporalBlock(in_ch, ch, k=kernel, d=d, p=dropout))
            in_ch = ch
            d *= 2
        self.tcn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(in_ch, num_classes)

    def forward(self, x, mask=None):  # x: (B,T,F) -> (B,F,T)
        x = x.transpose(1, 2)
        y = self.tcn(x)
        y = self.pool(y).squeeze(-1)
        return self.fc(y)


class TransformerClassifier(nn.Module):
    def __init__(self, input_size: int, num_classes: int, d_model: int = 256, nhead: int = 8,
                 num_layers: int = 4, dim_feedforward: int = 512, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout,
                                                   batch_first=True, activation='gelu')
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls = nn.Linear(d_model, num_classes)

    def forward(self, x, mask=None):  # x: (B,T,F)
        h = self.proj(x)
        if mask is not None:
            # mask: 1 dla prawdziwych klatek -> trzeba zamienić na bool: True=pozycje do zmaskowania
            # Transformer oczekuje src_key_padding_mask: True where to mask (ignore)
            key_mask = (mask == 0.0)
            h = self.encoder(h, src_key_padding_mask=key_mask)
        else:
            h = self.encoder(h)
        if mask is not None:
            mask_exp = mask.unsqueeze(-1)
            pooled = (h * mask_exp).sum(dim=1) / (mask_exp.sum(dim=1) + 1e-6)
        else:
            pooled = h.mean(dim=1)
        return self.cls(pooled)

# -------------------------
# 5) Trening / Ewaluacja
# -------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_split_list(path: str) -> List[str]:
    with open(path, 'r', encoding='utf-8') as f:
        return [l.strip() for l in f if l.strip()]


def make_loaders(proc_dir: str, splits_dir: str, batch: int, max_len: int, augment: bool,
                 balance_sampler: bool = False):
    class_map_path = os.path.join(proc_dir, 'class_map.json')
    train_files = read_split_list(os.path.join(splits_dir, 'train.txt'))
    val_files = read_split_list(os.path.join(splits_dir, 'val.txt'))
    test_files = read_split_list(os.path.join(splits_dir, 'test.txt'))

    ds_train = SkeletonSeqDataset(train_files, class_map_path, max_len=max_len, augment=augment)
    ds_val = SkeletonSeqDataset(val_files, class_map_path, max_len=max_len, augment=False)
    ds_test = SkeletonSeqDataset(test_files, class_map_path, max_len=max_len, augment=False)

    in_size = 33 * 7

    # Sampler wagowy (opcjonalnie) – w razie lekkiej nierównowagi klas
    if balance_sampler:
        # policz częstości klas w train
        class_map = json.load(open(class_map_path, 'r', encoding='utf-8'))
        counts = {k: 0 for k in class_map}
        for p in train_files:
            cls_name = os.path.basename(os.path.dirname(p))
            counts[cls_name] += 1
        total = sum(counts.values())
        weights_per_class = {cls: total / (len(counts) * c) for cls, c in counts.items()}
        weights = []
        for p in train_files:
            cls_name = os.path.basename(os.path.dirname(p))
            weights.append(weights_per_class[cls_name])
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(ds_train, batch_size=batch, sampler=sampler, num_workers=2, drop_last=False)
    else:
        train_loader = DataLoader(ds_train, batch_size=batch, shuffle=True, num_workers=2, drop_last=False)

    val_loader = DataLoader(ds_val, batch_size=batch, shuffle=False, num_workers=2, drop_last=False)
    test_loader = DataLoader(ds_test, batch_size=batch, shuffle=False, num_workers=2, drop_last=False)
    return train_loader, val_loader, test_loader, in_size


def build_model(model_name: str, in_size: int, num_classes: int, args) -> nn.Module:
    if model_name == 'lstm':
        return BiLSTMClassifier(input_size=in_size, hidden=args.hidden, num_layers=args.layers,
                                num_classes=num_classes, dropout=args.dropout)
    elif model_name == 'tcn':
        return TCNClassifier(input_size=in_size, num_classes=num_classes, channels=[args.hidden]*3,
                             kernel=3, dropout=args.dropout)
    elif model_name == 'transformer':
        return TransformerClassifier(input_size=in_size, num_classes=num_classes, d_model=args.hidden,
                                     nhead=max(2, args.hidden // 64), num_layers=args.layers,
                                     dim_feedforward=args.hidden*2, dropout=args.dropout)
    else:
        raise ValueError("model must be one of: lstm, tcn, transformer")


def compute_class_weights(train_files: List[str], class_map_path: str) -> torch.Tensor:
    class_map = json.load(open(class_map_path, 'r', encoding='utf-8'))
    counts = np.zeros(len(class_map), dtype=np.int64)
    for p in train_files:
        cls_name = os.path.basename(os.path.dirname(p))
        counts[class_map[cls_name]] += 1
    # wagi ~ 1/freq (znormalizowane do średniej = 1)
    inv = 1.0 / np.clip(counts, 1, None)
    w = inv * (len(counts) / inv.sum())
    return torch.tensor(w, dtype=torch.float32)


def evaluate(model, loader, device, class_map_path: Optional[str] = None, ret_cm: bool = False):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y, m in loader:
            x, y, m = x.to(device), y.to(device), m.to(device)
            logits = model(x, mask=m)
            pred = logits.argmax(dim=1)
            ys += y.cpu().tolist()
            ps += pred.cpu().tolist()
    acc = accuracy_score(ys, ps)
    f1 = f1_score(ys, ps, average='macro')
    if not ret_cm:
        return acc, f1
    cm = confusion_matrix(ys, ps)
    target_names = None
    if class_map_path is not None:
        class_map = json.load(open(class_map_path, 'r', encoding='utf-8'))
        target_names = [k for k, _ in sorted(class_map.items(), key=lambda x: x[1])]
    report = classification_report(ys, ps, target_names=target_names)
    return acc, f1, cm, report


def train_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    train_loader, val_loader, test_loader, in_size = make_loaders(
        proc_dir=args.proc_dir,
        splits_dir=args.splits_dir,
        batch=args.batch,
        max_len=args.max_len,
        augment=bool(args.augment),
        balance_sampler=bool(args.balance_sampler),
    )

    class_map_path = os.path.join(args.proc_dir, 'class_map.json')
    class_map = json.load(open(class_map_path, 'r', encoding='utf-8'))
    num_classes = len(class_map)

    model = build_model(args.model, in_size, num_classes, args).to(device)

    # Class weights z train split
    train_files = read_split_list(os.path.join(args.splits_dir, 'train.txt'))
    class_weights = compute_class_weights(train_files, class_map_path).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_f1 = -1.0
    best_path = os.path.join(args.save_dir, 'best.pt')
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for x, y, m in pbar:
            x, y, m = x.to(device), y.to(device), m.to(device)
            logits = model(x, mask=m)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
            pbar.set_postfix({'loss': f"{np.mean(losses):.4f}"})

        # Walidacja
        val_acc, val_f1 = evaluate(model, val_loader, device, class_map_path=None)
        scheduler.step(val_f1)
        print(f"[VAL] acc={val_acc:.4f} f1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({'model': model.state_dict(), 'args': vars(args), 'class_map': class_map}, best_path)
            print("[SAVE] best ->", best_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if args.early_stop > 0 and epochs_no_improve >= args.early_stop:
                print(f"[EARLY STOP] brak poprawy przez {args.early_stop} epok.")
                break

    # Test końcowy
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    acc, f1, cm, report = evaluate(model, test_loader, device, class_map_path=class_map_path, ret_cm=True)
    with open(os.path.join(args.save_dir, 'test_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report + '\n')
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(cm))
    print("[TEST] acc=%.4f f1=%.4f" % (acc, f1))

# -------------------------
# 6) Predykcja segmentowa na nowym wideo
# -------------------------

def predict_on_video_segments(video_path: str, ckpt_path: str, class_map_path: str,
                              model_type: Optional[str] = None, max_len: int = 300,
                              segment_len: int = 120, predict_stride: int = 60,
                              vote: str = 'mean', save_timeline: Optional[str] = None) -> Dict:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seq = extract_sequence_from_video(video_path)
    X = seq['features']  # (T,33,7)

    # przygotuj okna [t, t+segment_len)
    T = X.shape[0]
    if T < 2:
        raise ValueError("Za krótkie wideo do predykcji")

    ckpt = torch.load(ckpt_path, map_location=device)
    class_map = json.load(open(class_map_path, 'r', encoding='utf-8'))
    inv_map = {v: k for k, v in class_map.items()}
    num_classes = len(class_map)

    if model_type is None:
        model_type = ckpt.get('args', {}).get('model', 'lstm')

    in_size = 33 * 7
    # odtwórz model
    if model_type == 'lstm':
        model = BiLSTMClassifier(input_size=in_size, hidden=ckpt.get('args', {}).get('hidden', 256),
                                 num_layers=ckpt.get('args', {}).get('layers', 2),
                                 num_classes=num_classes, dropout=ckpt.get('args', {}).get('dropout', 0.3))
    elif model_type == 'tcn':
        model = TCNClassifier(input_size=in_size, num_classes=num_classes,
                              channels=[ckpt.get('args', {}).get('hidden', 256)]*3,
                              kernel=3, dropout=ckpt.get('args', {}).get('dropout', 0.2))
    else:
        model = TransformerClassifier(input_size=in_size, num_classes=num_classes,
                                      d_model=ckpt.get('args', {}).get('hidden', 256),
                                      nhead=max(2, ckpt.get('args', {}).get('hidden', 256)//64),
                                      num_layers=ckpt.get('args', {}).get('layers', 4),
                                      dim_feedforward=ckpt.get('args', {}).get('hidden', 256)*2,
                                      dropout=ckpt.get('args', {}).get('dropout', 0.1))
    model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()

    # iteruj po segmentach
    probs_list = []
    seg_ranges = []
    t = 0
    while t < T:
        seg = X[t:t+segment_len]
        if seg.shape[0] == 0:
            break
        # pad/crop do max_len (użyjemy max_len aby dopasować do treningu)
        if seg.shape[0] > max_len:
            st = (seg.shape[0] - max_len) // 2
            seg = seg[st:st+max_len]
        else:
            pad = np.zeros((max_len - seg.shape[0], seg.shape[1], seg.shape[2]), dtype=seg.dtype)
            seg = np.concatenate([seg, pad], axis=0)
        mask = (seg.sum(axis=(1, 2)) != 0).astype(np.float32)
        x_t = torch.from_numpy(seg.reshape(max_len, -1)).unsqueeze(0).float().to(device)
        m_t = torch.from_numpy(mask).unsqueeze(0).float().to(device)
        with torch.no_grad():
            logits = model(x_t, mask=m_t)
            p = torch.softmax(logits, dim=1).cpu().numpy()[0]
        probs_list.append(p)
        seg_ranges.append((int(t), int(min(t+segment_len, T))))
        t += predict_stride if predict_stride > 0 else segment_len
        if predict_stride <= 0 and t >= T:
            break

    probs_arr = np.stack(probs_list, axis=0)  # (S,C)

    if vote == 'majority':
        votes = probs_arr.argmax(axis=1)
        # tie‑break: weź klasę o najwyższej średniej pewności
        vals, counts = np.unique(votes, return_counts=True)
        pred_idx = int(vals[np.argmax(counts)])
    else:  # mean softmax (domyślne, stabilniejsze)
        mean_probs = probs_arr.mean(axis=0)
        pred_idx = int(mean_probs.argmax())

    if save_timeline:
        import csv
        with open(save_timeline, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            header = ['seg_idx', 'start_frame', 'end_frame'] + [inv_map[i] for i in range(num_classes)]
            w.writerow(header)
            for i, (p, (s, e)) in enumerate(zip(probs_list, seg_ranges)):
                w.writerow([i, s, e] + list(map(lambda x: f"{x:.6f}", p)))
        print(f"[TIMELINE] zapisano: {save_timeline}")

    return {
        'pred_idx': pred_idx,
        'pred_class': inv_map[pred_idx],
        'probs': {inv_map[i]: float(p) for i, p in enumerate(probs_arr.mean(axis=0))},
        'segments': {
            'ranges': seg_ranges,
            'per_segment_probs': [list(map(float, p)) for p in probs_list],
            'classes': [inv_map[i] for i in range(num_classes)]
        }
    }

# -------------------------
# 7) CLI
# -------------------------

def build_argparser():
    p = argparse.ArgumentParser(description='LMA Effort Actions – end-to-end pipeline (v2)')
    sub = p.add_subparsers(dest='cmd')

    p_ext = sub.add_parser('extract', help='Ekstrakcja szkieletów z wideo -> .npz')
    p_ext.add_argument('--data_root', type=str, required=True)
    p_ext.add_argument('--out_dir', type=str, required=True)
    p_ext.add_argument('--min_conf', type=float, default=0.5)
    p_ext.add_argument('--fps', type=int, default=25)

    p_split = sub.add_parser('split', help='Podział na train/val/test')
    p_split.add_argument('--proc_dir', type=str, required=True)
    p_split.add_argument('--splits_dir', type=str, required=True)
    p_split.add_argument('--train', type=float, default=0.7)
    p_split.add_argument('--val', type=float, default=0.15)
    p_split.add_argument('--test', type=float, default=0.15)
    p_split.add_argument('--seed', type=int, default=42)

    p_train = sub.add_parser('train', help='Trening klasyfikatora')
    p_train.add_argument('--proc_dir', type=str, required=True)
    p_train.add_argument('--splits_dir', type=str, required=True)
    p_train.add_argument('--model', type=str, choices=['lstm', 'tcn', 'transformer'], default='transformer')
    p_train.add_argument('--epochs', type=int, default=70)
    p_train.add_argument('--batch', type=int, default=8)
    p_train.add_argument('--lr', type=float, default=3e-4)
    p_train.add_argument('--hidden', type=int, default=256)
    p_train.add_argument('--layers', type=int, default=4)
    p_train.add_argument('--dropout', type=float, default=0.1)
    p_train.add_argument('--max_len', type=int, default=300)
    p_train.add_argument('--save_dir', type=str, required=True)
    p_train.add_argument('--augment', type=int, default=1)
    p_train.add_argument('--balance_sampler', type=int, default=0)
    p_train.add_argument('--early_stop', type=int, default=10)
    p_train.add_argument('--seed', type=int, default=42)

    p_eval = sub.add_parser('eval', help='Ewaluacja na teście')
    p_eval.add_argument('--proc_dir', type=str, required=True)
    p_eval.add_argument('--splits_dir', type=str, required=True)
    p_eval.add_argument('--ckpt', type=str, required=True)

    p_pred = sub.add_parser('predict', help='Predykcja na nowym wideo (okna + głosowanie)')
    p_pred.add_argument('--video', type=str, required=True)
    p_pred.add_argument('--ckpt', type=str, required=True)
    p_pred.add_argument('--class_map', type=str, required=True)
    p_pred.add_argument('--max_len', type=int, default=300)
    p_pred.add_argument('--segment_len', type=int, default=120)
    p_pred.add_argument('--predict_stride', type=int, default=60)
    p_pred.add_argument('--vote', type=str, choices=['mean', 'majority'], default='mean')
    p_pred.add_argument('--save_timeline', type=str, default=None)

    return p


def main():
    parser = build_argparser()
    args = parser.parse_args()

    if args.cmd == 'extract':
        extract_dir(args.data_root, args.out_dir, min_conf=args.min_conf, fps=args.fps)

    elif args.cmd == 'split':
        make_splits(args.proc_dir, args.splits_dir, train=args.train, val=args.val, test=args.test, seed=args.seed)

    elif args.cmd == 'train':
        train_model(args)

    elif args.cmd == 'eval':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        class_map_path = os.path.join(args.proc_dir, 'class_map.json')
        train_loader, val_loader, test_loader, in_size = make_loaders(
            proc_dir=args.proc_dir, splits_dir=args.splits_dir, batch=16, max_len=300, augment=False)
        ckpt = torch.load(args.ckpt, map_location=device)
        class_map = ckpt.get('class_map', json.load(open(class_map_path)))
        num_classes = len(class_map)
        model_type = ckpt.get('args', {}).get('model', 'transformer')
        # zbuduj model spójny z checkpointem
        dummy = argparse.Namespace(hidden=ckpt.get('args', {}).get('hidden', 256),
                                   layers=ckpt.get('args', {}).get('layers', 4),
                                   dropout=ckpt.get('args', {}).get('dropout', 0.1))
        if model_type == 'lstm':
            model = BiLSTMClassifier(input_size=in_size, hidden=dummy.hidden, num_layers=dummy.layers,
                                     num_classes=num_classes, dropout=dummy.dropout)
        elif model_type == 'tcn':
            model = TCNClassifier(input_size=in_size, num_classes=num_classes,
                                  channels=[dummy.hidden]*3, kernel=3, dropout=dummy.dropout)
        else:
            model = TransformerClassifier(input_size=in_size, num_classes=num_classes,
                                          d_model=dummy.hidden,
                                          nhead=max(2, dummy.hidden//64),
                                          num_layers=dummy.layers,
                                          dim_feedforward=dummy.hidden*2,
                                          dropout=dummy.dropout)
        model.load_state_dict(ckpt['model'])
        model.to(device)
        acc, f1, cm, report = evaluate(model, test_loader, device, class_map_path=class_map_path, ret_cm=True)
        print(report)
        print("Confusion matrix:\n", cm)

    elif args.cmd == 'predict':
        out = predict_on_video_segments(
            video_path=args.video,
            ckpt_path=args.ckpt,
            class_map_path=args.class_map,
            max_len=args.max_len,
            segment_len=args.segment_len,
            predict_stride=args.predict_stride,
            vote=args.vote,
            save_timeline=args.save_timeline
        )
        print(json.dumps(out, ensure_ascii=False, indent=2))

    else:
        parser.print_help()


if __name__ == '__main__':
    main()

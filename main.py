#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LMA Effort Actions – end‑to‑end pipeline (Python, PyTorch, MediaPipe)
--------------------------------------------------------------------
Co robi ten plik:
1) Ekstrakcja szkieletów z wideo (MediaPipe Pose) i zapis do .npz
2) Budowa zbioru (train/val/test) z augmentacją sekwencji
3) Trenowanie klasyfikatora (BiLSTM lub TCN) na sekwencjach szkieletów
4) Ewaluacja (accuracy, F1, confusion matrix)
5) Inference na nowym wideo (predykcja klasy i prawdopodobieństw)

Struktura katalogów oczekiwana na wejściu (surowe wideo):
DATA_ROOT/
  punch/
    vid_001.mp4
    vid_002.mp4
    ...
  wring/
    vid_010.mp4
  press/
  flick/
  ... (dowolne klasy LMA, np. 8 effort actions)

Wywołania (przykłady):
1) Ekstrakcja
   python LMA_DeepLearning_Pipeline.py extract \
      --data_root ./data_raw \
      --out_dir ./data_proc \
      --min_conf 0.5 --fps 25

2) Split train/val/test
   python LMA_DeepLearning_Pipeline.py split \
      --proc_dir ./data_proc \
      --splits_dir ./splits \
      --train 0.7 --val 0.15 --test 0.15 --seed 42

3) Trening
   python LMA_DeepLearning_Pipeline.py train \
      --proc_dir ./data_proc --splits_dir ./splits \
      --model lstm --epochs 50 --batch 8 --lr 3e-4 \
      --save_dir ./runs/exp1 --augment 1

4) Ewaluacja na teście
   python LMA_DeepLearning_Pipeline.py eval \
      --proc_dir ./data_proc --splits_dir ./splits \
      --ckpt ./runs/exp1/best.pt

5) Predykcja na nowym wideo
   python LMA_DeepLearning_Pipeline.py predict \
      --video path/to/new.mp4 \
      --ckpt ./runs/exp1/best.pt \
      --class_map ./data_proc/class_map.json

Wymagania (pip):
- mediapipe
- opencv-python
- torch, torchvision, torchaudio (zgodnie z CUDA lub CPU)
- numpy, scipy, scikit-learn, matplotlib
- tqdm

Uwaga: MediaPipe Pose zwraca 33 punkty na klatkę (x,y,z,visibility). Tutaj
budujemy cechę  (x,y,z,vis, dx,dy,dz) znormalizowaną względem wymiaru ciała.
"""

import os
import cv2
import json
import math
import glob
import time
import random
import shutil
import argparse
import itertools
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple

# Torch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ML utils
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# Progress
from tqdm import tqdm

# -------------------------
# 1) Pose extraction (MediaPipe)
# -------------------------
try:
    import mediapipe as mp
except ImportError:
    mp = None
    print("[WARN] mediapipe nie jest zainstalowane – komenda 'extract' nie zadziała.")

POSE_LANDMARKS = 33  # MediaPipe


def pair_distance(a, b):
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def normalize_landmarks(landmarks: np.ndarray) -> Tuple[np.ndarray, float]:
    """Normalizacja szkieletu: przesunięcie do środka miednicy i skalowanie rozmiarem ciała.
    landmarks: (33, 4) -> (x,y,z,vis)
    Zwraca: norm_landmarks (33,4), scale (float)
    """
    # Użyjemy midpoint bioder (LEFT_HIP=23, RIGHT_HIP=24) jako centrum
    LHIP, RHIP = 23, 24
    center = (landmarks[LHIP, :3] + landmarks[RHIP, :3]) / 2.0
    shifted = landmarks.copy()
    shifted[:, :3] = landmarks[:, :3] - center

    # Skala: dystans barków (LEFT_SHOULDER=11, RIGHT_SHOULDER=12) lub bioder jako fallback
    LSH, RSH = 11, 12
    if landmarks[LSH, 3] > 0 and landmarks[RSH, 3] > 0:
        scale = np.linalg.norm(landmarks[LSH, :3] - landmarks[RSH, :3]) + 1e-6
    else:
        scale = np.linalg.norm(landmarks[LHIP, :3] - landmarks[RHIP, :3]) + 1e-6

    shifted[:, :3] /= scale
    return shifted, float(scale)


def extract_sequence_from_video(video_path: str, min_conf: float = 0.5, fps: int = 25,
                                static_image_mode: bool = False) -> Dict:
    """Przetwarza wideo -> sekwencja landmarków i cech pochodnych.
    Zwraca dict z kluczami: 'landmarks' (T,33,4), 'features' (T,33,7), 'orig_fps', 'used_fps'
    """
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
        arr, _ = normalize_landmarks(arr)
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
    class_map = {}
    classes = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])
    for ci, c in enumerate(classes):
        class_map[c] = ci
    with open(os.path.join(out_dir, 'class_map.json'), 'w', encoding='utf-8') as f:
        json.dump(class_map, f, ensure_ascii=False, indent=2)

    for c in classes:
        in_dir = os.path.join(data_root, c)
        out_c = os.path.join(out_dir, c)
        os.makedirs(out_c, exist_ok=True)
        videos = sorted(glob.glob(os.path.join(in_dir, '*.mp4')) +
                        glob.glob(os.path.join(in_dir, '*.mov')) +
                        glob.glob(os.path.join(in_dir, '*.avi')))
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

    items = []
    class_map = json.load(open(os.path.join(proc_dir, 'class_map.json'), 'r', encoding='utf-8'))
    for c in class_map:
        files = sorted(glob.glob(os.path.join(proc_dir, c, '*.npz')))
        items += [(f, c) for f in files]

    # per-class split dla balansu
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
# 3) Dataset + augmentacja sekwencji
# -------------------------
@dataclass
class AugmentCfg:
    time_warp_prob: float = 0.2
    time_warp_strength: float = 0.2  # resample +/-20%
    jitter_prob: float = 0.3
    jitter_sigma: float = 0.01
    dropout_prob: float = 0.2
    dropout_rate: float = 0.1  # procent punktów do wyzerowania (symuluje brak detekcji)


def time_resample(seq: np.ndarray, scale: float) -> np.ndarray:
    """Resampling w osi czasu przez interpolację liniową.
    seq: (T,33,7) -> (int(T*scale),33,7)
    """
    T = seq.shape[0]
    new_T = max(int(round(T * scale)), 2)
    xs = np.linspace(0, 1, T)
    xs_new = np.linspace(0, 1, new_T)
    out = np.empty((new_T, seq.shape[1], seq.shape[2]), dtype=seq.dtype)
    for j in range(seq.shape[1]):
        for k in range(seq.shape[2]):
            out[:, j, k] = np.interp(xs_new, xs, seq[:, j, k])
    return out


def augment_sequence(seq: np.ndarray, cfg: AugmentCfg) -> np.ndarray:
    s = seq.copy()
    # Time-warp
    if random.random() < cfg.time_warp_prob:
        scale = 1.0 + random.uniform(-cfg.time_warp_strength, cfg.time_warp_strength)
        s = time_resample(s, scale)
    # Jitter
    if random.random() < cfg.jitter_prob:
        s[:, :, :3] += np.random.normal(0, cfg.jitter_sigma, size=s[:, :, :3].shape)
    # Dropout landmarków
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
        self.inv_map = {v: k for k, v in self.class_map.items()}
        self.max_len = max_len
        self.augment = augment
        self.augment_cfg = augment_cfg

    def __len__(self):
        return len(self.files)

    def pad_or_crop(self, x: np.ndarray) -> np.ndarray:
        T = x.shape[0]
        if T == self.max_len:
            return x
        if T > self.max_len:
            # central crop
            st = (T - self.max_len) // 2
            return x[st:st + self.max_len]
        # pad
        pad = np.zeros((self.max_len - T, x.shape[1], x.shape[2]), dtype=x.dtype)
        return np.concatenate([x, pad], axis=0)

    def __getitem__(self, idx):
        path = self.files[idx]
        data = np.load(path)
        seq = data['features']  # (T,33,7)
        if self.augment:
            seq = augment_sequence(seq, self.augment_cfg)
        seq = self.pad_or_crop(seq)
        # z ścieżki klasy
        cls_name = os.path.basename(os.path.dirname(path))
        y = self.class_map[cls_name]
        # maska (1 tam gdzie są rzeczywiste klatki)
        mask = (seq.sum(axis=(1, 2)) != 0).astype(np.float32)
        # flatten per-frame: (T, 33*7)
        seq = seq.reshape(seq.shape[0], -1)
        return torch.from_numpy(seq).float(), torch.tensor(y).long(), torch.from_numpy(mask).float()


# -------------------------
# 4) Modele: BiLSTM i TCN
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
            # masked mean over time
            mask = mask.unsqueeze(-1)  # (B,T,1)
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


# -------------------------
# 5) Trening / Ewaluacja
# -------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loaders(proc_dir: str, splits_dir: str, batch: int, max_len: int, augment: bool) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    class_map_path = os.path.join(proc_dir, 'class_map.json')
    with open(os.path.join(splits_dir, 'train.txt'), 'r', encoding='utf-8') as f:
        train_files = [l.strip() for l in f if l.strip()]
    with open(os.path.join(splits_dir, 'val.txt'), 'r', encoding='utf-8') as f:
        val_files = [l.strip() for l in f if l.strip()]
    with open(os.path.join(splits_dir, 'test.txt'), 'r', encoding='utf-8') as f:
        test_files = [l.strip() for l in f if l.strip()]

    ds_train = SkeletonSeqDataset(train_files, class_map_path, max_len=max_len, augment=augment)
    ds_val = SkeletonSeqDataset(val_files, class_map_path, max_len=max_len, augment=False)
    ds_test = SkeletonSeqDataset(test_files, class_map_path, max_len=max_len, augment=False)

    in_size = 33 * 7

    return (
        DataLoader(ds_train, batch_size=batch, shuffle=True, num_workers=2, drop_last=False),
        DataLoader(ds_val, batch_size=batch, shuffle=False, num_workers=2, drop_last=False),
        DataLoader(ds_test, batch_size=batch, shuffle=False, num_workers=2, drop_last=False),
        in_size
    )


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
    )

    class_map = json.load(open(os.path.join(args.proc_dir, 'class_map.json'), 'r', encoding='utf-8'))
    num_classes = len(class_map)

    if args.model == 'lstm':
        model = BiLSTMClassifier(input_size=in_size, hidden=args.hidden, num_layers=args.layers,
                                 num_classes=num_classes, dropout=args.dropout)
    elif args.model == 'tcn':
        model = TCNClassifier(input_size=in_size, num_classes=num_classes, channels=[args.hidden]*3,
                              kernel=3, dropout=args.dropout)
    else:
        raise ValueError("model must be 'lstm' or 'tcn'")

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    criterion = nn.CrossEntropyLoss()

    best_f1 = -1.0
    best_path = os.path.join(args.save_dir, 'best.pt')

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

        # Val
        val_acc, val_f1 = evaluate(model, val_loader, device)
        scheduler.step(val_f1)
        print(f"[VAL] acc={val_acc:.4f} f1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({'model': model.state_dict(), 'args': vars(args), 'class_map': class_map}, best_path)
            print("[SAVE] best ->", best_path)

    # Test końcowy
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    test_acc, test_f1, cm, report = evaluate(model, test_loader, device, ret_cm=True)
    with open(os.path.join(args.save_dir, 'test_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report + '\n')
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(cm))
    print("[TEST] acc=%.4f f1=%.4f" % (test_acc, test_f1))


def evaluate(model, loader, device, ret_cm: bool = False):
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
    # Dodano argument class_map_path, aby nie używać args
    target_names = None
    if hasattr(evaluate, 'class_map_path') and evaluate.class_map_path is not None:
        class_map = json.load(open(evaluate.class_map_path, 'r', encoding='utf-8'))
        target_names = [k for k, _ in sorted(class_map.items(), key=lambda x: x[1])]
    report = classification_report(ys, ps, target_names=target_names)
    return acc, f1, cm, report


# -------------------------
# 6) Predykcja na jednym wideo
# -------------------------

def predict_on_video(video_path: str, ckpt_path: str, class_map_path: str, model_type: str = None,
                     max_len: int = 300) -> Dict:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seq = extract_sequence_from_video(video_path)
    x = seq['features']  # (T,33,7)
    T = x.shape[0]
    if T > max_len:
        st = (T - max_len) // 2
        x = x[st:st + max_len]
    else:
        pad = np.zeros((max_len - T, x.shape[1], x.shape[2]), dtype=x.dtype)
        x = np.concatenate([x, pad], axis=0)
    mask = (x.sum(axis=(1, 2)) != 0).astype(np.float32)
    x = torch.from_numpy(x.reshape(max_len, -1)).unsqueeze(0).float().to(device)
    mask = torch.from_numpy(mask).unsqueeze(0).float().to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    class_map = json.load(open(class_map_path, 'r', encoding='utf-8'))
    num_classes = len(class_map)

    # inferuj typ modelu jeśli nie podano
    if model_type is None:
        model_type = ckpt.get('args', {}).get('model', 'lstm')

    in_size = 33 * 7
    if model_type == 'lstm':
        model = BiLSTMClassifier(input_size=in_size, hidden=ckpt.get('args', {}).get('hidden', 256),
                                 num_layers=ckpt.get('args', {}).get('layers', 2),
                                 num_classes=num_classes, dropout=ckpt.get('args', {}).get('dropout', 0.3))
    else:
        model = TCNClassifier(input_size=in_size, num_classes=num_classes,
                              channels=[ckpt.get('args', {}).get('hidden', 256)]*3,
                              kernel=3, dropout=ckpt.get('args', {}).get('dropout', 0.2))
    model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()

    with torch.no_grad():
        logits = model(x, mask=mask)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred = int(probs.argmax())

    inv_map = {v: k for k, v in class_map.items()}
    return {
        'pred_idx': pred,
        'pred_class': inv_map[pred],
        'probs': {inv_map[i]: float(p) for i, p in enumerate(probs)}
    }


# -------------------------
# 7) CLI
# -------------------------

def build_argparser():
    p = argparse.ArgumentParser(description='LMA Effort Actions – end-to-end pipeline')
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
    p_train.add_argument('--model', type=str, choices=['lstm', 'tcn'], default='lstm')
    p_train.add_argument('--epochs', type=int, default=50)
    p_train.add_argument('--batch', type=int, default=8)
    p_train.add_argument('--lr', type=float, default=3e-4)
    p_train.add_argument('--hidden', type=int, default=256)
    p_train.add_argument('--layers', type=int, default=2)
    p_train.add_argument('--dropout', type=float, default=0.3)
    p_train.add_argument('--max_len', type=int, default=300)
    p_train.add_argument('--save_dir', type=str, required=True)
    p_train.add_argument('--augment', type=int, default=1)
    p_train.add_argument('--seed', type=int, default=42)

    p_eval = sub.add_parser('eval', help='Ewaluacja na teście')
    p_eval.add_argument('--proc_dir', type=str, required=True)
    p_eval.add_argument('--splits_dir', type=str, required=True)
    p_eval.add_argument('--ckpt', type=str, required=True)

    p_pred = sub.add_parser('predict', help='Predykcja na nowym wideo')
    p_pred.add_argument('--video', type=str, required=True)
    p_pred.add_argument('--ckpt', type=str, required=True)
    p_pred.add_argument('--class_map', type=str, required=True)
    p_pred.add_argument('--max_len', type=int, default=300)

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
        # quick evaluation using saved best
        train_loader, val_loader, test_loader, _ = make_loaders(
            proc_dir=args.proc_dir, splits_dir=args.splits_dir, batch=16, max_len=300, augment=False)
        ckpt = torch.load(args.ckpt, map_location=device)
        class_map = ckpt.get('class_map', json.load(open(os.path.join(args.proc_dir, 'class_map.json'))))
        num_classes = len(class_map)
        model_type = ckpt.get('args', {}).get('model', 'lstm')
        in_size = 33 * 7
        if model_type == 'lstm':
            model = BiLSTMClassifier(input_size=in_size, hidden=ckpt.get('args', {}).get('hidden', 256),
                                     num_layers=ckpt.get('args', {}).get('layers', 2), num_classes=num_classes,
                                     dropout=ckpt.get('args', {}).get('dropout', 0.3))
        else:
            model = TCNClassifier(input_size=in_size, num_classes=num_classes,
                                  channels=[ckpt.get('args', {}).get('hidden', 256)]*3,
                                  kernel=3, dropout=ckpt.get('args', {}).get('dropout', 0.2))
        model.load_state_dict(ckpt['model'])
        model.to(device)
        # przekazujemy ścieżkę do class_map.json przez atrybut funkcji
        evaluate.class_map_path = os.path.join(args.proc_dir, 'class_map.json')
        acc, f1, cm, report = evaluate(model, test_loader, device, ret_cm=True)
        print(report)
        print("Confusion matrix:\n", cm)

    elif args.cmd == 'predict':
        out = predict_on_video(args.video, args.ckpt, args.class_map, max_len=args.max_len)
        print(json.dumps(out, ensure_ascii=False, indent=2))

    else:
        parser.print_help()


if __name__ == '__main__':
    main()

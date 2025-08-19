import numpy as np
import random
from dataclasses import dataclass

LEFT_RIGHT_PAIRS = [
    (11, 12), (13, 14), (15, 16),
    (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)
]

@dataclass
class AugmentCfg:
    time_warp_prob: float = 0.5
    time_warp_strength: float = 0.3
    jitter_prob: float = 0.6
    jitter_sigma: float = 0.015
    dropout_prob: float = 0.5
    dropout_rate: float = 0.1
    flip_prob: float = 0.5
    random_crop_prob: float = 0.5
    crop_ratio_min: float = 0.6

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
    s = seq.copy()
    s[:, :, 0] *= -1.0
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
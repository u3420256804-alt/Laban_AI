import os
import cv2
import glob
import json
import numpy as np
import math

POSE_LANDMARKS = 33
LEFT_RIGHT_PAIRS = [
    (11, 12), (13, 14), (15, 16),
    (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)
]

try:
    import mediapipe as mp
except ImportError:
    mp = None
    print("[WARN] mediapipe nie jest zainstalowane – komenda 'extract' nie zadziała.")

def normalize_landmarks(landmarks: np.ndarray, align_shoulders: bool = True):
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
                               align_shoulders: bool = True):
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

    dL = np.diff(L[:, :, :3], axis=0, prepend=L[:1, :, :3])  # (T,33,3)
    features = np.concatenate([L, dL], axis=-1)  # (T,33,7)

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

    from tqdm import tqdm
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
                print(f"[ERR] {v}: {e}")
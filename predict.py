import torch
import numpy as np
import json
from extract import extract_sequence_from_video
from models import BiLSTMClassifier, TCNClassifier, TransformerClassifier

def predict_on_video_segments(video_path: str, ckpt_path: str, class_map_path: str,
                              model_type: str = None, max_len: int = 300,
                              segment_len: int = 120, predict_stride: int = 60,
                              vote: str = 'mean', save_timeline: str = None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seq = extract_sequence_from_video(video_path)
    X = seq['features']

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

    probs_list = []
    seg_ranges = []
    t = 0
    while t < T:
        seg = X[t:t+segment_len]
        if seg.shape[0] == 0:
            break
        if seg.shape[0] > max_len:
            seg = seg[:max_len]
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

    probs_arr = np.stack(probs_list, axis=0)

    if vote == 'majority':
        votes = probs_arr.argmax(axis=1)
        vals, counts = np.unique(votes, return_counts=True)
        pred_idx = int(vals[np.argmax(counts)])
    else:
        mean_probs = probs_arr.mean(axis=0)
        pred_idx = int(mean_probs.argmax())

    if save_timeline:
        with open(save_timeline, 'w', newline='', encoding='utf-8') as f:
            import csv
            writer = csv.writer(f)
            writer.writerow(['start', 'end'] + [inv_map[i] for i in range(num_classes)])
            for (start, end), p in zip(seg_ranges, probs_list):
                writer.writerow([start, end] + list(map(float, p)))
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
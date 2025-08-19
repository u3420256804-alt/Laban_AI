import os
import torch
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from dataset import SkeletonSeqDataset
from models import BiLSTMClassifier, TCNClassifier, TransformerClassifier

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def read_split_list(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return [l.strip() for l in f if l.strip()]

def make_loaders(proc_dir: str, splits_dir: str, batch: int, max_len: int, augment: bool, balance_sampler: bool = False):
    import json
    import os
    class_map_path = os.path.join(proc_dir, 'class_map.json')
    train_files = read_split_list(os.path.join(splits_dir, 'train.txt'))
    val_files = read_split_list(os.path.join(splits_dir, 'val.txt'))
    test_files = read_split_list(os.path.join(splits_dir, 'test.txt'))

    ds_train = SkeletonSeqDataset(train_files, class_map_path, max_len=max_len, augment=augment)
    ds_val = SkeletonSeqDataset(val_files, class_map_path, max_len=max_len, augment=False)
    ds_test = SkeletonSeqDataset(test_files, class_map_path, max_len=max_len, augment=False)

    in_size = 33 * 7

    if balance_sampler:
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

def build_model(model_name: str, in_size: int, num_classes: int, args):
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

def compute_class_weights(train_files, class_map_path):
    import json
    class_map = json.load(open(class_map_path, 'r', encoding='utf-8'))
    counts = np.zeros(len(class_map), dtype=np.int64)
    for p in train_files:
        cls_name = os.path.basename(os.path.dirname(p))
        counts[class_map[cls_name]] += 1
    inv = 1.0 / np.clip(counts, 1, None)
    w = inv * (len(counts) / inv.sum())
    return torch.tensor(w, dtype=torch.float32)

def evaluate(model, loader, device, class_map_path=None, ret_cm=False):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y, m in loader:
            x = x.to(device)
            m = m.to(device)
            logits = model(x, mask=m)
            pred = logits.argmax(dim=1).cpu().numpy()
            ys.extend(y.cpu().numpy())
            ps.extend(pred)
    acc = accuracy_score(ys, ps)
    f1 = f1_score(ys, ps, average='macro')
    if not ret_cm:
        return acc, f1
    cm = confusion_matrix(ys, ps)
    target_names = None
    if class_map_path is not None:
        import json
        class_map = json.load(open(class_map_path, 'r', encoding='utf-8'))
        target_names = [k for k, _ in sorted(class_map.items(), key=lambda x: x[1])]
    report = classification_report(ys, ps, target_names=target_names)
    return acc, f1, cm, report

def train_model(args):
    from tqdm import tqdm
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

    import json
    class_map_path = os.path.join(args.proc_dir, 'class_map.json')
    class_map = json.load(open(class_map_path, 'r', encoding='utf-8'))
    num_classes = len(class_map)

    model = build_model(args.model, in_size, num_classes, args).to(device)

    train_files = read_split_list(os.path.join(args.splits_dir, 'train.txt'))
    class_weights = compute_class_weights(train_files, class_map_path).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    best_f1 = -1.0
    best_path = os.path.join(args.save_dir, 'best.pt')
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for x, y, m in pbar:
            x = x.to(device)
            y = y.to(device)
            m = m.to(device)
            optimizer.zero_grad()
            logits = model(x, mask=m)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            pbar.set_postfix(loss=np.mean(losses))

        val_acc, val_f1 = evaluate(model, val_loader, device, class_map_path=None)
        scheduler.step(val_f1)
        print(f"[VAL] acc={val_acc:.4f} f1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({'model': model.state_dict(), 'args': vars(args), 'class_map': class_map}, best_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.early_stop:
                print(f"Early stopping at epoch {epoch}")
                break

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    acc, f1, cm, report = evaluate(model, test_loader, device, class_map_path=class_map_path, ret_cm=True)
    with open(os.path.join(args.save_dir, 'test_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report + '\n')
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(cm))
    print("[TEST] acc=%.4f f1=%.4f" % (acc, f1))
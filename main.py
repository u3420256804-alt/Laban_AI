import argparse
import os
from extract import extract_dir
from split import make_splits
from train import train_model
from predict import predict_on_video_segments

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
    p_train.add_argument('--save_dir', type=str, required=True)
    p_train.add_argument('--augment', type=int, default=1)
    p_train.add_argument('--max_len', type=int, default=300)
    p_train.add_argument('--early_stop', type=int, default=10)
    p_train.add_argument('--balance_sampler', type=int, default=0)
    p_train.add_argument('--hidden', type=int, default=256)
    p_train.add_argument('--layers', type=int, default=4)
    p_train.add_argument('--dropout', type=float, default=0.2)
    p_train.add_argument('--seed', type=int, default=42)

    p_eval = sub.add_parser('eval', help='Ewaluacja modelu')
    p_eval.add_argument('--proc_dir', type=str, required=True)
    p_eval.add_argument('--splits_dir', type=str, required=True)
    p_eval.add_argument('--ckpt', type=str, required=True)

    p_pred = sub.add_parser('predict', help='Predykcja na nowym wideo')
    p_pred.add_argument('--video', type=str, required=True)
    p_pred.add_argument('--ckpt', type=str, required=True)
    p_pred.add_argument('--class_map', type=str, required=True)
    p_pred.add_argument('--segment_len', type=int, default=120)
    p_pred.add_argument('--predict_stride', type=int, default=60)
    p_pred.add_argument('--vote', type=str, choices=['mean', 'majority'], default='mean')
    p_pred.add_argument('--save_timeline', type=str, default=None)
    p_pred.add_argument('--model_type', type=str, choices=['lstm', 'tcn', 'transformer'], default=None)
    p_pred.add_argument('--max_len', type=int, default=300)
    return p

def main():
    parser = build_argparser()
    args = parser.parse_args()

    if args.cmd == 'extract':
        extract_dir(args.data_root, args.out_dir, min_conf=args.min_conf, fps=args.fps)
    elif args.cmd == 'split':
        make_splits(args.proc_dir, args.splits_dir, args.train, args.val, args.test, seed=args.seed)
    elif args.cmd == 'train':
        train_model(args)
    elif args.cmd == 'eval':
        from train import make_loaders, evaluate
        import torch
        import json
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        class_map_path = os.path.join(args.proc_dir, 'class_map.json')
        _, _, test_loader, _ = make_loaders(
            proc_dir=args.proc_dir,
            splits_dir=args.splits_dir,
            batch=8,
            max_len=300,
            augment=False,
            balance_sampler=False
        )
        ckpt = torch.load(args.ckpt, map_location=device)
        model_type = ckpt.get('args', {}).get('model', 'transformer')
        from models import BiLSTMClassifier, TCNClassifier, TransformerClassifier
        num_classes = len(json.load(open(class_map_path, 'r', encoding='utf-8')))
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
        acc, f1, cm, report = evaluate(model, test_loader, device, class_map_path=class_map_path, ret_cm=True)
        print(report)
        print("Confusion matrix:\n", cm)
        print(f"Test acc={acc:.4f} f1={f1:.4f}")
    elif args.cmd == 'predict':
        result = predict_on_video_segments(
            video_path=args.video,
            ckpt_path=args.ckpt,
            class_map_path=args.class_map,
            model_type=args.model_type,
            max_len=args.max_len,
            segment_len=args.segment_len,
            predict_stride=args.predict_stride,
            vote=args.vote,
            save_timeline=args.save_timeline
        )
        print("Predicted class:", result['pred_class'])
        print("Probabilities (mean over all segments):", result['probs'])
        print("Segments:")
        for (start, end), seg_probs in zip(result['segments']['ranges'], result['segments']['per_segment_probs']):
            print(f"  Frames {start}-{end}:")
            for cls, prob in zip(result['segments']['classes'], seg_probs):
                print(f"    {cls}: {prob:.4f}")
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
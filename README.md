# Laban_AI

How to setup venv:

Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt

Commands to run project
Extract skeletons from video:
python main.py extract --data_root ./data_raw --out_dir ./data_proc --min_conf 0.5 --fps 25

Split data into train/val/test:
python main.py split --proc_dir ./data_proc --splits_dir ./splits --train 0.7 --val 0.15 --test 0.15

Train the classifier:
python main.py train --proc_dir ./data_proc --splits_dir ./splits --model transformer --epochs 70 --batch 8 --lr 3e-4 --hidden 256 --layers 4 --dropout 0.1 --max_len 300 --augment 1 --balance_sampler 0 --early_stop 10 --save_dir ./runs/exp2

Evaluate on test set:
python main.py eval --proc_dir ./data_proc --splits_dir ./splits --ckpt ./runs/exp2/best.pt

Predict on a new video:
python main.py predict --video path/to/new.mp4 --ckpt ./runs/exp2/best.pt --class_map ./data_proc/class_map.json --segment_len 120 --predict_stride 60 --vote mean --save_timeline timeline.csv

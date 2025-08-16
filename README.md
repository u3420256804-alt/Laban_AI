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
python main.py split --proc_dir ./data_proc --splits_dir ./splits --train 0.7 --val 0.15 --test 0.15 --seed 42

Train the classifier:
python main.py train --proc_dir ./data_proc --splits_dir ./splits --model lstm --epochs 50 --batch 8 --lr 3e-4 --save_dir ./runs/exp1 --augment 1

Evaluate on test set:
python main.py eval --proc_dir ./data_proc --splits_dir ./splits --ckpt ./runs/exp1/best.pt

Predict on a new video:
python main.py predict --video path/to/new.mp4 --ckpt ./runs/exp1/best.pt --class_map ./data_proc/class_map.json

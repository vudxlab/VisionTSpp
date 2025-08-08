import hashlib
import subprocess
import sys
import os
import pandas as pd
import random
from tempfile import TemporaryDirectory

dir_path = sys.argv[-1]

VM_ARCH = "mae_base"

def process_csv(csv_path):

    if "weighted_5" in csv_path:
        num_patch_input = 5
    elif "weighted_7" in csv_path:
        num_patch_input = 7
    elif "first_run" in csv_path:
        num_patch_input = 6
    else:
        raise ValueError(csv_path)

    df = pd.read_csv(csv_path)
    
    
    # ! 250401 adds: 还要设置模型size
    vm_arch = VM_ARCH  # 默认为'mae_base'
    if 'large' in csv_path:
        vm_arch = 'mae_large'
    elif 'huge' in csv_path:
        vm_arch = 'mae_huge'
        
    
    # ! 20250410 adds:
    quantile = 'quantile' in csv_path
    quantile_int = 1 if quantile else 0

    if 'clip_input_new' in csv_path: clip_input = 2
    elif 'clip_input' in csv_path: clip_input = 1
    else: clip_input = 0
    
    if 'complete_no_clip' in csv_path: complete_no_clip = 1
    else: complete_no_clip = 0
    
    
    
    def map_process(row):
        if "without_missing" not in row['dataset'] and row['dataset'] != 'bitcoin':
            return row
        with TemporaryDirectory() as tmp_dir:
            tmp_file = os.path.join(tmp_dir, "temp.csv")
            print(tmp_file)
            cmd = [
                "python", "run.py", 
                "--dataset", row['dataset'].replace("without_missing", "with_missing") if row['dataset'] != 'bitcoin' else 'bitcoin_with_missing',
                "--save_name", tmp_file,
                "--periodicity", "autotune", 
                "--context_len", "1000",
                "--no_periodicity_context_len", "300",
                "--batch_size", "256",
                "--checkpoint_path", csv_path.replace(".csv", ".ckpt").replace("_missing", ""),
                "--num_patch_input", str(num_patch_input),
                "--vm_arch", vm_arch,
                "--quantile", str(quantile_int),
                "--clip_input", str(clip_input),
                "--complete_no_clip", str(complete_no_clip),
            ]
            print(cmd)
            subprocess.run(cmd, cwd="/home/mouxiangchen/uni2ts/scripts/VisionTS/eval_gluonts")
            new_df = pd.read_csv(tmp_file)
        return new_df.iloc[0]

    df: pd.DataFrame = df.apply(map_process, axis=1)
    if "_missing" in csv_path:
        df.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path.replace(".csv", "_missing.csv"), index=False)

    # for row in df.iterrows():
    #     df.iloc[]

    # for line in lines[st:ed]:
    #     line = line.strip()
    #     if line == "": continue
    #     if os.path.exists(line.replace(".ckpt", ".csv")):
    #         continue

    #     if "weighted_5" in line:
    #         num_patch_input = 5
    #     elif "weighted_7" in line:
    #         num_patch_input = 7
    #     elif "first_run" in line:
    #         num_patch_input = 6
    #     else:
    #         raise ValueError(line)

    #     for ds in datasets:
    #         cmd = [
    #             "python", "run.py", 
    #             "--dataset", ds,
    #             "--save_name", line.replace(".ckpt", ".csv"),
    #             "--periodicity", "autotune", 
    #             "--context_len", "1000",
    #             "--no_periodicity_context_len", "300",
    #             "--batch_size", "256",
    #             "--checkpoint_path", line,
    #             "--num_patch_input", str(num_patch_input)
    #         ]
    #         print(cmd)

    #         subprocess.run(cmd, cwd="/home/mouxiangchen/VisionTS/eval_gluonts")



for root, dirs, files in os.walk(dir_path):
    for file in files:
        print(f"{root = }, {file = }")
        if file.endswith(".csv") and '_missing' not in file:
            path = os.path.join(root, file)
            if not os.path.exists(path.replace(".csv", "_missing.csv")):
                process_csv(path)
import hashlib
import subprocess
import sys
import os


# datasets_ltsf = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Electricity", "Weather"]
# datasets_ltsf = ["ETTm1"]
datasets_ltsf = ["ETTm2"]
# datasets_ltsf = ["electricity", "weather"]
# ltsf_pred_len_list = [96]
ltsf_pred_len_list = [336]
# ltsf_pred_len_list = [1200]
# ltsf_pred_len_list = [96, 720]
# ltsf_pred_len_list = [96, 192, 336, 720]


ltsf_metadata = {
    "ETTh1": {"CONTEXT_LEN": 2880, "PERIODICITY": 24, "LOOKBACK_LEN_VISUAL": 300},
    "ETTh2": {"CONTEXT_LEN": 1728, "PERIODICITY": 24, "LOOKBACK_LEN_VISUAL": 300},
    "ETTm1": {"CONTEXT_LEN": 2304, "PERIODICITY": 96, "LOOKBACK_LEN_VISUAL": 1200},
    "ETTm2": {"CONTEXT_LEN": 4032, "PERIODICITY": 96, "LOOKBACK_LEN_VISUAL": 1200},
    "Electricity": {"CONTEXT_LEN": 2880, "PERIODICITY": 24, "LOOKBACK_LEN_VISUAL": 300},
    "Weather": {"CONTEXT_LEN": 4032, "PERIODICITY": 144, "LOOKBACK_LEN_VISUAL": 1800},
}
ALIGN_CONST=0.4
NORM_CONST=0.4
VM_ARCH="mae_base"


# change to your own path
line = "/home/mouxiangchen/uni2ts/outputs/pretrain/visionts/lotsa_v1_weighted/weighted_7_quantile_512_multi_color_9_heads/checkpoints/processed_epoch=999-step=100000.ckpt"



print("line:", line)
# 以及如果line对应的ckpt不存在的话，也跳过
if not os.path.exists(line):
    print(f"Ckpt {line} not exists, skip...")


if "weighted_5" in line:
    num_patch_input = 5
elif "weighted_7" in line:
    num_patch_input = 7
elif "first_run" in line:
    num_patch_input = 6
else:
    raise ValueError(line)


# 对每个pred_len长度都遍历过去！！
for pred_len in ltsf_pred_len_list:
    for ds in datasets_ltsf:
        
        # ! 还要设置模型size
        vm_arch = VM_ARCH  # 默认为'mae_base'
        if 'large' in line:
            vm_arch = 'mae_large'
        elif 'huge' in line:
            vm_arch = 'mae_huge'
            
        
        # ! 20250410 adds:
        quantile = 'quantile' in line
        quantile_int = 1 if quantile else 0

        if 'clip_input_new' in line: clip_input = 2
        elif 'clip_input' in line: clip_input = 1
        else: clip_input = 0
        
        if 'complete_no_clip' in line: complete_no_clip = 1
        else: complete_no_clip = 0
        
        # ! 250426 adds:
        if 'multi' in line: multivariate = 1
        else: multivariate = 0
        # ! 250429 adds:
        if 'color' in line: color = 1
        else: color = 0
        
        
        # ! 250505 adds: 临时加的用于解决univariate忘加color的问题！！
        if '9_heads' in line: color = 1
                
        
        # 一些参数
        data_form = ds if "ETT" in ds else "custom"
        # 否则跑这个数据集
        cmd = [
            "python", "-u", "run.py",
            "--task_name", "long_term_forecast",
            "--is_training", "1",  # 虽然设置了is_training为1，但由于train_epochs为0，所以仍然没训～
            "--model", "VisionTS",
            # "--root_path", "./datasets/",
            "--root_path", "~/VisionTSpp/datasets/",
            "--data_path", f"{ds}.csv",
            "--save_dir", f"./test_visual/{ds}_{pred_len}",
            "--model_id", f"VisionTS_{ds}_{pred_len}_{line}",
            "--data", f"{data_form}",
            "--features", "M",
            "--train_epochs", "0",
            "--vm_arch", vm_arch,
            "--checkpoint_path", line,
            "--seq_len", str(ltsf_metadata[ds]["CONTEXT_LEN"]),
            "--periodicity", str(ltsf_metadata[ds]["PERIODICITY"]),
            "--pred_len", str(pred_len),
            "--norm_const", str(NORM_CONST),
            "--align_const", str(ALIGN_CONST),
            "--save_name", line.replace(".ckpt", f"_ltsf_pred{pred_len}.csv"),
            "--quantile", str(quantile_int),
            "--clip_input", str(clip_input),
            "--complete_no_clip", str(complete_no_clip),
            "--color", str(color),
            "--num_patch_input", str(num_patch_input),
            "--visual", "1",
            "--num_workers", "0",
            "--LOOKBACK_LEN_VISUAL", str(ltsf_metadata[ds]["LOOKBACK_LEN_VISUAL"])
        ]
        print(cmd)

        # subprocess.run(cmd, cwd="/home/mouxiangchen/uni2ts/scripts/VisionTS/eval_ltsf")
        subprocess.run(cmd, cwd="/home/lefeishen/uni2ts/scripts/VisionTS/eval_ltsf")
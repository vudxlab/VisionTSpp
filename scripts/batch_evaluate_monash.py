import hashlib
import subprocess
import sys
import os

split = int(sys.argv[-2])
count = int(sys.argv[-1])

datasets = "m1_monthly monash_m3_monthly monash_m3_other m4_monthly m4_weekly m4_daily m4_hourly tourism_quarterly tourism_monthly cif_2016_6 cif_2016_12 australian_electricity_demand bitcoin pedestrian_counts vehicle_trips_without_missing kdd_cup_2018_without_missing weather nn5_daily_without_missing nn5_weekly car_parts_without_missing fred_md traffic_hourly traffic_weekly rideshare_without_missing hospital covid_deaths temperature_rain_without_missing sunspot_without_missing saugeenday us_births".split(" ")

datasets_pf = "electricity solar-energy walmart jena_weather istanbul_traffic turkey_power".split(" ")

datasets_ltsf = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Electricity", "Weather"]
# datasets_ltsf = ["ETTh1", "ETTh2"]
# datasets_ltsf = ["electricity", "weather"]
# ltsf_pred_len_list = [96, 720]
ltsf_pred_len_list = [96, 192, 336, 720]


ltsf_metadata = {
    "ETTh1": {"CONTEXT_LEN": 2880, "PERIODICITY": 24},
    "ETTh2": {"CONTEXT_LEN": 1728, "PERIODICITY": 24},
    "ETTm1": {"CONTEXT_LEN": 2304, "PERIODICITY": 96},
    "ETTm2": {"CONTEXT_LEN": 4032, "PERIODICITY": 96},
    "Electricity": {"CONTEXT_LEN": 2880, "PERIODICITY": 24},
    "Weather": {"CONTEXT_LEN": 4032, "PERIODICITY": 144},
}
ALIGN_CONST=0.4
NORM_CONST=0.4
VM_ARCH="mae_base"



lines = open("ckpt_path.txt").readlines()
lines = [x for x in lines]
st = int(len(lines) / count * split)
ed = int(len(lines) / count * (split + 1))


print(f"Total task: {len(lines)}, Range: [{st}, {ed})")
print(lines)



# 1.先做pf的测试：
for line in lines[st:ed]:
    line = line.strip()
    if line == "": continue
    if os.path.exists(line.replace(".ckpt", "_pf.csv")):
        continue

    if "weighted_5" in line:
        num_patch_input = 5
    elif "weighted_7" in line:
        num_patch_input = 7
    elif "first_run" in line:
        num_patch_input = 6
    else:
        raise ValueError(line)

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
    
    
    
    # if not ('epoch=999' in line): 
    #     continue
    
    # ! 还要设置模型size
    vm_arch = VM_ARCH  # 默认为'mae_base'
    if 'large' in line:
        vm_arch = 'mae_large'
    elif 'huge' in line:
        vm_arch = 'mae_huge'
    

    for ds in datasets_pf:
        cmd = [
            "python", "run.py", 
            "--dataset", ds,
            "--save_name", line.replace(".ckpt", "_pf.csv"),
            "--periodicity", "freq", 
            "--context_len", "2000",
            "--batch_size", "256",
            "--checkpoint_path", line,
            "--num_patch_input", str(num_patch_input),
            "--vm_arch", vm_arch,
            "--quantile", str(quantile_int),
            "--clip_input", str(clip_input),
            "--complete_no_clip", str(complete_no_clip),
            "--multivariate", str(multivariate),
            "--color", str(color),
        ]
        print(cmd)

        subprocess.run(cmd, cwd="/home/mouxiangchen/uni2ts/scripts/VisionTS/eval_gluonts")


# 2. 再做monash的测试：
for line in lines[st:ed]:
    line = line.strip()
    if line == "": continue
    if os.path.exists(line.replace(".ckpt", ".csv")):
        continue

    if "weighted_5" in line:
        num_patch_input = 5
    elif "weighted_7" in line:
        num_patch_input = 7
    elif "first_run" in line:
        num_patch_input = 6
    else:
        raise ValueError(line)

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
    
    
    # if not ('epoch=999' in line): 
    #     continue
    
    # ! 还要设置模型size
    vm_arch = VM_ARCH  # 默认为'mae_base'
    if 'large' in line:
        vm_arch = 'mae_large'
    elif 'huge' in line:
        vm_arch = 'mae_huge'
    

    for ds in datasets:
        cmd = [
            "python", "run.py", 
            "--dataset", ds,
            "--save_name", line.replace(".ckpt", ".csv"),
            "--periodicity", "autotune",
            "--context_len", "1000",
            "--no_periodicity_context_len", "300",
            "--batch_size", "256",
            "--checkpoint_path", line,
            "--num_patch_input", str(num_patch_input),
            "--vm_arch", vm_arch,
            "--quantile", str(quantile_int),
            "--clip_input", str(clip_input),
            "--complete_no_clip", str(complete_no_clip),
            "--multivariate", str(multivariate),
            "--color", str(color),
        ]
        print(cmd)

        subprocess.run(cmd, cwd="/home/mouxiangchen/uni2ts/scripts/VisionTS/eval_gluonts")



# 3. 最后做ltsf的测试：
for line in lines[st:ed]:
    line = line.strip()
    # 如果line为空，则跳过
    if line == "": continue
    
    print("line:", line)
    # 以及如果line对应的ckpt不存在的话，也跳过
    if not os.path.exists(line):
        print(f"Ckpt {line} not exists, skip...")
        continue
    
    
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
        
        runned_dataset_list = []
        if os.path.exists(line.replace(".ckpt", f"_ltsf_pred{pred_len}.csv")):
            with open(line.replace(".ckpt", f"_ltsf_pred{pred_len}.csv"), "r") as f:
                for content in f:
                    if content.strip() == "": 
                        continue
                    
                    runned_dataset = content.strip().split(",")[0]
                    if runned_dataset == "dataset":
                        continue
                    runned_dataset_list.append(runned_dataset)
            # continue

        for ds in datasets_ltsf:
            # 如果之前已经跑过了就不跑了
            if ds in runned_dataset_list:
                continue
            
            
            
            # 这里多加一些判断，因为ETTm1和ETTm2比较慢，所以只跑300、600和1000的ckpt，以及只跑96的pred_len
            # 而"Electricity", "Weather"更慢，所以只跑最后一个ckpt；也只跑96的pred_len！
            
            # if ds == "ETTh1" or ds == "ETTh2":
            #     if pred_len == 96 or pred_len == 720:
            #         pass
            #     else:
            #         if not ('epoch=299' in line or 'epoch=599' in line or 'epoch=999' in line): 
            #             continue
            
            if ds == "ETTm1" or ds == "ETTm2":
                if pred_len == 96:
                    if not ('epoch=299' in line or 'epoch=599' in line or 'epoch=999' in line): 
                        continue
                else:
                    if not 'epoch=999' in line: 
                        continue
            
            if ds == "Electricity" or ds == "Weather":
                # if pred_len > 96: continue
                if not 'epoch=999' in line: 
                    continue

            
            # # 统一只跑最后一个epoch？
            # if not ('epoch=999' in line): 
            #     continue
            
            
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
                "--root_path", "./datasets/",
                "--data_path", f"{ds}.csv",
                "--save_dir", f"./save/{ds}_{pred_len}",
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
            ]
            print(cmd)

            subprocess.run(cmd, cwd="/home/mouxiangchen/uni2ts/scripts/VisionTS/eval_ltsf")
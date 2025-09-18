import hashlib
import subprocess
import sys
import os
import pandas as pd
import numpy as np


# naive_result = """0.881
# 1.203
# 1.236
# 0.782
# 1.137
# 0.906""".split("\n")
# naive_result = np.asarray([float(x) for x in naive_result])

# name_step_result = {}

# zero_shot = np.array([0.755, 1.141, 0.949, 0.737, 0.706, 0.856])
# print(float((zero_shot / naive_result).prod() ** (1.0 / len(zero_shot))))

# moirai_small = np.array([0.981, 1.465, 1.048, 0.521, 0.99, 0.948])
# moirai_base = np.array([0.792, 1.292, 0.964, 0.487, 0.644, 0.888])
# moirai_large = np.array([0.751, 1.237, 1.007, 0.515, 0.631, 0.87])
# moirai_small = float((moirai_small / naive_result).prod() ** (1.0 / len(zero_shot)))
# moirai_base = float((moirai_base / naive_result).prod() ** (1.0 / len(zero_shot)))
# moirai_large = float((moirai_large / naive_result).prod() ** (1.0 / len(zero_shot)))


# ! 250501 adds:
datasets_pf = ["electricity", "solar-energy", "walmart", "jena_weather", "istanbul_traffic", "turkey_power"]

name_step_result_crps = {}
name_step_result_msis = {}


naive_crps = [0.070, 0.512, 0.151, 0.068, 0.257, 0.085]
naive_mase = [0.881, 1.203, 1.236, 0.782, 1.137, 0.906]


moirai_small_crps = np.array([0.072, 0.471, 0.103, 0.049, 0.173, 0.048])
moirai_base_crps = np.array([0.055, 0.419, 0.093, 0.041, 0.116, 0.040])
moirai_large_crps = np.array([0.050, 0.406, 0.098, 0.051, 0.112, 0.036])
moirai_small_msis = np.array([7.999, 8.425, 9.371, 5.236, 5.937, 7.127])
moirai_base_msis = np.array([6.172, 7.011, 8.421, 5.136, 4.461, 6.766])
moirai_large_msis = np.array([5.875, 6.250, 8.520, 4.962, 4.277, 6.341])


for root, dirs, files in os.walk("../outputs"):
    for file in files:
        try:
            if file.endswith("_pf.csv") and 'last' not in file:
                csv_path = os.path.join(root, file)
                df = pd.read_csv(csv_path)
                
                raw_result_crps = df['mean_weighted_sum_quantile_loss']
                raw_result_msis = df['MSIS']
                cur_datasets = df['dataset']
                
                result_crps = np.asarray(raw_result_crps).tolist()
                result_msis = np.asarray(raw_result_msis).tolist()
                cur_datasets = np.asarray(cur_datasets).tolist()
                print(cur_datasets)
                print(result_crps)
                print(result_msis)
                
                # result = np.round(result, 2)
                # if 'last' in file:
                #     step = "110000"
                # else:
                
                step = file.split("-step=")[-1].split(".csv")[0]
                name = root.split("/")[-2]
                if step.endswith("_pf"):
                    step = step.replace("_pf", "")

                # ! 20250416 adds:
                if "-v" in step:  # 一个目录下多次重复训练的ckpt会带有"-v1","-v2"等后缀，故需要去掉！
                    print(f"'-v' in step:{step}, skipping...")
                    continue
                
                # 打印一下
                if step == "0":
                    # print(float((result / naive_result).prod() ** (1.0 / len(result))))
                    # print(result.tolist())
                    continue
                
                if name not in name_step_result_crps:
                    name_step_result_crps[name] = {}
                if name not in name_step_result_msis:
                    name_step_result_msis[name] = {}
                
                for i in range(len(cur_datasets)):
                    dataset = cur_datasets[i]
                    crps_value = result_crps[i]
                    msis_value = result_msis[i]
                    print(dataset, crps_value, msis_value)
                    print(step)
                    
                    if dataset not in name_step_result_crps[name]:
                        name_step_result_crps[name][dataset] = {}
                    name_step_result_crps[name][dataset][step] = crps_value
                    
                    if dataset not in name_step_result_msis[name]:
                        name_step_result_msis[name][dataset] = {}
                    name_step_result_msis[name][dataset][step] = msis_value
        except Exception as e:
            print(f"In file={file}, Exception:", e)
            pass

import matplotlib.pyplot as plt

# del name_step_result['weighted_5_rd']
# for deleted in ['weighted_5', 'weighted_7_full', 'unweighted_7', 'weighted_7_gauss_px1.0_huber_missing', 'weighted_7_gauss_missing', 'weighted_7_missing', 'weighted_7_gauss_px0.2_huber_512_missing', 'weighted_7_gauss_px0.5_huber_512_top7_missing', 'weighted_7_gauss_px0.5_huber_512_top5_missing']:
# for deleted in ['weighted_5', 'weighted_7_full', 'unweighted_7', 'weighted_7_gauss_px1.0_huber_missing', 'weighted_7_gauss_missing', 'weighted_7_missing', 'weighted_7_gauss_px0.2_huber_512_missing', 'weighted_7_gauss_px0.5_huber_missing']:
#     if deleted in name_step_result:# or 'weighted_7_gauss_px0.5_huber_missing' not in deleted:
#         del name_step_result[deleted]

# for deleted in list(name_step_result.keys()):
#     if 'weighted_7_gauss_px0.5_huber_missing' in deleted:
#         del name_step_result[deleted]


# showed = ['weighted_7_gauss_px0.5_huber_512', 
#           'weighted_7_gauss_px0.5_huber_512_im2', 
#           'weighted_7_nlgauss_px0.5_huber_512']

# showed = [
#         #   'weighted_7_gauss_px0.5_huber_512', 
#         #   'weighted_7_gauss_px0.5_huber_2048', 
        
#         #   'weighted_7_gauss_px0.5_huber_2048_im2',
#         #   "weighted_7_gauss_px0.5_huber_512_im1",
#         #   "weighted_7_gauss_px0.5_huber_512_im2",
#         #   'weighted_7_gauss_px0.5_huber_512_im3',
#         #   'weighted_7_nlgauss_px0.5_huber_512',
        
#         #   'weighted_7_quantile_px0.5_huber_512',
#           'weighted_7_quantile_px0_512',
#           'weighted_7_quantile_px0.5_quantile_512_im0.2',
#           'weighted_7_quantile_px0.5_quantile_1024_im0.2',
#           'weighted_7_quantile_px0.5_quantile_1024_im0.4',
#         #   'weighted_7_quantile_px0.5_quantile_2048_im0.2',
#           'weighted_7_1024_large',
#         #   'weighted_7_512_huge',
#           'weighted_7_1024_huge',
#           'weighted_7_quantile_1024',
#           'weighted_7_quantile_512_clip_input',
#           'weighted_7_quantile_512_clip_input_new',
#           'weighted_7_quantile_512_complete_no_clip',
#         #   'weighted_7_quantile_px0.5_huber_512_im3',
#           ]


showed = [
          'weighted_7_quantile_px0_512',
          'weighted_7_quantile_512_clip_input',
          'weighted_7_quantile_512_clip_input_new',
        #   'weighted_7_quantile_1024_clip_input',
        #   'weighted_7_quantile_512_complete_no_clip',
          'weighted_7_quantile_512_filter',
        #   'weighted_7_quantile_1024_filter',
        #   'weighted_7_quantile_512_multi',
          'weighted_7_quantile_512_multi_color',
          'weighted_7_quantile_512_multi_color_9_heads',
          'weighted_7_quantile_512_multi_color_9_heads_large',
          'weighted_7_quantile_512_9_heads',
          'weighted_7_quantile_512_multi_color_9_heads_huge',
        #   'weighted_7_quantile_512_multi_color_9_heads_rand_init',
        #   'weighted_7_quantile_128_multi_color_9_heads_rand_init',
        #   'weighted_7_quantile_512_multi_color_9_heads_ln_only',
          
          'weighted_7_quantile_512_multi_color_9_heads_gift_eval',
          
        #   'weighted_7_quantile_1024_clip_input',
          'weighted_7_quantile_1024_filter',
        #   'weighted_7_1024_large',
        #   'weighted_7_1024_huge',
          'weighted_7_quantile_px0.5_quantile_512_im0.2',
        #   'weighted_7_quantile_px0.5_quantile_1024_im0.2',
        #   'weighted_7_quantile_px0.5_quantile_1024_im0.4',
          ]


for deleted in list(name_step_result_crps.keys()):
    print(deleted)
    if deleted not in showed:
        del name_step_result_crps[deleted]

for deleted in list(name_step_result_msis.keys()):
    print(deleted)
    if deleted not in showed:
        del name_step_result_msis[deleted]


# 画图！
for dataset in datasets_pf:
    try:
        plt.figure(figsize=(6, 8))
        # plt.tight_layout(h_pad=3.0)
        
        # 1. 先画CRPS
        plt.subplot(2, 1, 1)
        plt.title(f"CRPS on {dataset}")
        
        for name in name_step_result_crps:
            y = []
            x = []
            
            # 取出当前pred_len和dataset：
            cur_step_list = name_step_result_crps[name][dataset].items()
            
            for step, result in sorted(cur_step_list, key=lambda x: int(x[0])):
                x.append(int(step))
                y.append(result)
            
            print(name, y)
            plt.plot(x, y, label=name, markersize=5, marker='o')
        
        plt.xlabel("Training Step")
        plt.grid()
        plt.legend(fontsize=7)

        cur_dataset_index = datasets_pf.index(dataset)
        
        crps_small = moirai_small_crps[cur_dataset_index]
        plt.axhline(y=crps_small, color='C0', linestyle='--')
        plt.text(90000, crps_small + 0.02, 'Moirai (Small)', color='black', ha='center')
        
        crps_base = moirai_base_crps[cur_dataset_index]
        plt.axhline(y=crps_base, color='C1', linestyle='--')
        plt.text(90000, crps_base + 0.001, 'Moirai (Base)', color='black', ha='center')
        
        crps_large = moirai_large_crps[cur_dataset_index]
        plt.axhline(y=crps_large, color='C2', linestyle='--')
        plt.text(90000, crps_large - 0.005, 'Moirai (Large)', color='black', ha='center')



        # 2. 再画MSIS
        plt.subplot(2, 1, 2)
        plt.title(f"MSIS on {dataset}")
        
        for name in name_step_result_msis:
            # if n == 'first_run' or 'unweigh' in n or 'full' in n: continue
            # if 'rd' not in n:
            #     y = [0.8374937152805807]
            #     x = [0]
            # else:
            y = []
            x = []
            
            # 取出当前pred_len和dataset：
            cur_step_list = name_step_result_msis[name][dataset].items()
            
            for step, result in sorted(cur_step_list, key=lambda x: int(x[0])):
                x.append(int(step))
                y.append(result)
            
            print(name, y)
            plt.plot(x, y, label=name, markersize=5, marker='o')
        
        plt.xlabel("Training Step")
        plt.grid()
        plt.legend(fontsize=7)
        
        
        cur_dataset_index = datasets_pf.index(dataset)
        
        msis_small = moirai_small_msis[cur_dataset_index]
        plt.axhline(y=msis_small, color='C0', linestyle='--')
        plt.text(90000, msis_small + 0.02, 'Moirai (Small)', color='black', ha='center')
        
        msis_base = moirai_base_msis[cur_dataset_index]
        plt.axhline(y=msis_base, color='C1', linestyle='--')
        plt.text(90000, msis_base + 0.001, 'Moirai (Base)', color='black', ha='center')
        
        msis_large = moirai_large_msis[cur_dataset_index]
        plt.axhline(y=msis_large, color='C2', linestyle='--')
        plt.text(90000, msis_large - 0.005, 'Moirai (Large)', color='black', ha='center')


        # plt.tight_layout(h_pad=3.0)
        plt.tight_layout()
        plt.savefig(f"result_pf_{dataset}_crps_msis.pdf")
    
    except Exception as e:
        print(f"Exception occured when visualing {dataset}:", e)
        # continue
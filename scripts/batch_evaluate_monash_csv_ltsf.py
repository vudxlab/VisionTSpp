import hashlib
import subprocess
import sys
import os
import pandas as pd
import numpy as np


datasets_ltsf = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Electricity", "Weather"]
# datasets_ltsf = ["ETTh1", "ETTh2"]
pred_len_list = [96, 192, 336, 720]
# pred_len_list = [96, 720]


# naive_result = """0.881
# 1.203
# 1.236
# 0.782
# 1.137
# 0.906""".split("\n")
# naive_result = np.asarray([float(x) for x in naive_result])

# zero_shot = np.array([0.755, 1.141, 0.949, 0.737, 0.706, 0.856])
# print(float((zero_shot / naive_result).prod() ** (1.0 / len(zero_shot))))



# moirai_small = np.array([0.981, 1.465, 1.048, 0.521, 0.99, 0.948])
# moirai_base = np.array([0.792, 1.292, 0.964, 0.487, 0.644, 0.888])
# moirai_large = np.array([0.751, 1.237, 1.007, 0.515, 0.631, 0.87])
# moirai_small = float((moirai_small / naive_result).prod() ** (1.0 / len(zero_shot)))
# moirai_base = float((moirai_base / naive_result).prod() ** (1.0 / len(zero_shot)))
# moirai_large = float((moirai_large / naive_result).prod() ** (1.0 / len(zero_shot)))



name_step_result_mse = {}
name_step_result_mae = {}

for root, dirs, files in os.walk("../outputs"):
    for file in files:
        for pred_len in pred_len_list:
            try:
                if file.endswith(f"_ltsf_pred{pred_len}.csv") and 'last' not in file:
                    csv_path = os.path.join(root, file)
                    df = pd.read_csv(csv_path)
                    raw_result_mse = df['MSE']
                    raw_result_mae = df['MAE']
                    cur_datasets = df['dataset']
                    # breakpoint()
                    result_mse = np.asarray(raw_result_mse).tolist()
                    result_mae = np.asarray(raw_result_mae).tolist()
                    cur_datasets = np.asarray(cur_datasets).tolist()
                    print(cur_datasets)
                    print(result_mse)
                    print(result_mae)
                    
                    # result = np.round(result, 2)
                    # if 'last' in file:
                    #     step = "110000"
                    # else:
                    step = file.split("-step=")[-1].split(".csv")[0]
                    name = root.split("/")[-2]
                    if step.endswith(f"_ltsf_pred{pred_len}"):
                        step = step.replace(f"_ltsf_pred{pred_len}", "")

                    if step == "0":
                        # print(float((result / naive_result).prod() ** (1.0 / len(result))))
                        # print(result.tolist())
                        continue
                    
                    # ! 20250416 adds:
                    if "-v" in step:  # 一个目录下多次重复训练的ckpt会带有"-v1","-v2"等后缀，故需要去掉！
                        print(f"'-v' in step:{step}, skipping...")
                        continue
                    
                    if name not in name_step_result_mse:
                        name_step_result_mse[name] = {}
                    if pred_len not in name_step_result_mse[name]:
                        name_step_result_mse[name][pred_len] = {}
                    if name not in name_step_result_mae:
                        name_step_result_mae[name] = {}
                    if pred_len not in name_step_result_mae[name]:
                        name_step_result_mae[name][pred_len] = {}
                    
                    
                    # mse_dict = dict(zip(cur_datasets, result_mse))
                    # mae_dict = dict(zip(cur_datasets, result_mae))
                    for i in range(len(cur_datasets)):
                        dataset = cur_datasets[i]
                        mse_value = result_mse[i]
                        mae_value = result_mae[i]
                        print(dataset, mse_value, mae_value)
                        print(step)
                        if dataset not in name_step_result_mse[name][pred_len]:
                            name_step_result_mse[name][pred_len][dataset] = {}
                        name_step_result_mse[name][pred_len][dataset][step] = mse_value
                        # [    
                        #     # float((result / naive_result).prod() ** (1.0 / len(result))),
                        #     # (result > zero_shot).mean()
                        # ]
                        if dataset not in name_step_result_mae[name][pred_len]:
                            name_step_result_mae[name][pred_len][dataset] = {}
                        name_step_result_mae[name][pred_len][dataset][step] = mae_value
                    # TODO: 注意：所以输出形式为：mse[name][pred_len][dataset][step] = value
            except Exception as e:
                print(f"In pred_len={pred_len} for file={file}, Exception:", e)
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


for deleted in list(name_step_result_mse.keys()):
    print("deleted:", deleted)
    if deleted not in showed:
        del name_step_result_mse[deleted]

for deleted in list(name_step_result_mae.keys()):
    print("deleted:", deleted)
    if deleted not in showed:
        del name_step_result_mae[deleted]


print("name_step_result_mse:", name_step_result_mse)


# 画图！
for pred_len in pred_len_list:
    for dataset in datasets_ltsf:
        try:
            plt.figure(figsize=(6, 8))
            
            # 1. 先画MSE
            plt.subplot(2, 1, 1)
            plt.title(f"MSE on {dataset} with pred_len={pred_len}")
            
            for n in name_step_result_mse:
                # if n == 'first_run' or 'unweigh' in n or 'full' in n: continue
                # if 'rd' not in n:
                #     y = [0.8374937152805807]
                #     x = [0]
                # else:
                y = []
                x = []
                
                # 取出当前pred_len和dataset：
                cur_step_list = name_step_result_mse[n][pred_len][dataset].items()
                
                for step, result in sorted(cur_step_list, key=lambda x: int(x[0])):
                    x.append(int(step))
                    y.append(result)
                print(n, y)
                plt.plot(x, y, label=n, markersize=5, marker='o')
            plt.xlabel("Training Step")
            plt.grid()
            plt.legend(fontsize=7)

            # plt.axhline(y=moirai_small, color='C0', linestyle='--')
            # plt.text(0.5, moirai_small + 0.02, 'Moirai (S)', color='C0', ha='center')
            
            # plt.axhline(y=moirai_base, color='black', linestyle='--')
            # plt.text(90000, moirai_base + 0.001, 'Moirai (Base)', color='black', ha='center')
            # plt.axhline(y=moirai_large, color='black', linestyle='--')
            # plt.text(90000, moirai_large - 0.005, 'Moirai (Large)', color='black', ha='center')


            # 2. 再画MAE
            plt.subplot(2, 1, 2)
            plt.title(f"MAE on {dataset} with pred_len={pred_len}")
            
            for n in name_step_result_mae:
                # if n == 'first_run' or 'unweigh' in n or 'full' in n: continue
                # if 'rd' not in n:
                #     y = [0.8374937152805807]
                #     x = [0]
                # else:
                y = []
                x = []
                
                # 取出当前pred_len和dataset：
                cur_step_list = name_step_result_mae[n][pred_len][dataset].items()
                
                for step, result in sorted(cur_step_list, key=lambda x: int(x[0])):
                    x.append(int(step))
                    y.append(result)
                print(n, y)
                plt.plot(x, y, label=n, markersize=5, marker='o')
            plt.xlabel("Training Step")
            plt.grid()
            plt.legend(fontsize=7)
            

            plt.tight_layout()
            plt.savefig(f"result_ltsf_{dataset}_pred{pred_len}.pdf")
        except Exception as e:
            print(f"Exception occured when visualing {dataset} with pred_len={pred_len}:", e)
            # continue

# plt.subplot(2, 1, 2)
# plt.title("Loss% v.s zero-shot")

# for i, n in enumerate(name_step_result):
#     # if n == 'first_run' or 'unweigh' in n or 'full' in n: continue
#     y = []
#     x = []
#     for step, result in sorted(name_step_result[n].items(), key=lambda x: int(x[0])):
#         x.append(int(step))
#         y.append(result[1] + i / 1000)
#     print(n, y)
#     plt.plot(x, y, label=n, markersize=5, marker='o')
# plt.grid()

# plt.tight_layout()
# plt.savefig("result_pf.pdf")
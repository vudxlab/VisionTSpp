import hashlib
import subprocess
import sys
import os
import pandas as pd
import numpy as np


naive_result = """2707.75
837.14
278.43
671.27
347.99
180.83
1218.06
15845.1
5636.83
578596.53
659.6
7.78E+17
170.88
31.42
42.13
2.36
8.26
16.71
0.65
2825.67
0.03
1.19
6.29
24.07
353.71
9.39
3.93
21.5
1152.67""".split("\n")
naive_result = np.asarray([float(x) for x in naive_result])


seasonal_naive = """2011.96
788.95
375.13
700.24
347.99
180.83
353.86
11405.45
1980.21
743512.31
455.96
7.78E+17
65.6
32.48
47.09
2.36
8.26
16.71
0.67
5385.53
0.013
1.19
1.6
20.01
353.71
9.39
3.93
21.5
1152.67""".split("\n")
seasonal_naive = np.asarray([float(x) for x in seasonal_naive])
print(f"{seasonal_naive = }")


# old VisionTS!!
zero_shot = np.asarray([1987.688524778489, 737.9286053731669, 315.85190011671443, 666.542285729571, 404.2335016314153, 215.6305384347969, 288.3720925388093, 12931.875315263896, 2560.1894177780323, 570907.24, 237.435282621234, 2.3263519906884823e+18, 52.00575514256805, 22.082236005438084, 38.16027990301013, 2.062313813461159, 3.514067169133817, 14.673289428601764, 0.5750384417170816, 1893.668241959648, 0.0120616098617277, 1.1392348558854624, 5.924230305868256, 19.362289049205128, 137.50615390907925, 6.372606752077468, 2.806585902972492, 30.217300610643388, 519.9384686380703])
# new VisionTS!!
zero_shot = np.asarray([1970.2121511918376, 740.1431896053211, 332.59920825701846, 665.7360103647181, 404.39254689438735, 215.0333546004561, 291.4275249196256, 12935.103790213889, 2564.39512326331, 618203.2392828392, 237.47544678133767, 2.3079811633160617e+18, 52.01517385240229, 22.4257183439435, 40.347538028337, 2.062225979335069, 3.5397552421196408, 14.56019899965986, 0.6052659306692014, 1807.42341742993, 0.0120608855068674, 1.139243161977859, 1.4075668775335717, 19.36181897912828, 154.08986048358565, 6.414361967002922, 2.8059039982798164, 30.21501088889398, 519.9338015417562])


name_step_result = {}

for root, dirs, files in os.walk("../outputs"):
    print(f"{root = }, {dirs = }, {files = }")
    if root.endswith("_backup") or root.endswith("_backup_missing"):
        print(f"'_backup' in root:{root}, continue...")
        continue
    for file in files:
        try:
            if file.endswith(".csv") and 'last' not in file and "missing" in file:
                # ! 250327 adds:
                tmp_file_name = file.split(".")[0]
                # print("file:", file)
                # print("tmp_file_name:", tmp_file_name)
                if tmp_file_name.endswith("_quantile") or tmp_file_name.endswith("_quantile_missing"):
                    print(f"'_quantile' in {tmp_file_name}, continue...")
                    continue
                # ! 250410 adds:
                if tmp_file_name.endswith("_backup") or tmp_file_name.endswith("_backup_missing"):
                    print(f"'_backup' in tmp_file_name:{tmp_file_name}, continue...")
                    continue
                
                csv_path = os.path.join(root, file)
                raw_result = pd.read_csv(csv_path)['MAE[0.5]']
                # ! 注意cif2016的计算！！！
                result = np.asarray(raw_result[:9].tolist() + \
                    [((raw_result[9] * 15 + raw_result[10] * 57) / (15+57))] + \
                    raw_result[11:].tolist())
                
                # result = np.round(result, 2)
                # if 'last' in file:
                #     step = "110000"
                # else:
                step = file.split("-step=")[-1].split(".csv")[0]
                name = root.split("/")[-2]
                if step.endswith("_missing"):
                    step = step.replace("_missing", "")
                    name = name + "_missing"
                    
                # ! 20250416 adds:
                if "-v" in step:  # 一个目录下多次重复训练的ckpt会带有"-v1","-v2"等后缀，故需要去掉！
                    print(f"'-v' in step:{step}, skipping...")
                    continue

                if step == "0":
                    print(float((result / naive_result).prod() ** (1.0 / len(result))))
                    print(result.tolist())
                    continue
                if name not in name_step_result:
                    name_step_result[name] = {}
                name_step_result[name][step] = [
                    float((result / naive_result).prod() ** (1.0 / len(result))),
                    (result > zero_shot).mean()
                ]
        except: pass

import matplotlib.pyplot as plt

print(name_step_result)
# del name_step_result['weighted_5_rd']
# for deleted in ['weighted_5', 'weighted_7_full', 'unweighted_7', 'weighted_7_gauss_px1.0_huber_missing', 'weighted_7_gauss_missing', 'weighted_7_missing', 'weighted_7_gauss_px0.2_huber_512_missing', 'weighted_7_gauss_px0.5_huber_512_top7_missing', 'weighted_7_gauss_px0.5_huber_512_top5_missing']:
# for deleted in ['weighted_5', 'weighted_7_full', 'unweighted_7', 'weighted_7_gauss_px1.0_huber_missing', 'weighted_7_gauss_missing', 'weighted_7_missing', 'weighted_7_gauss_px0.2_huber_512_missing', 'weighted_7_gauss_px0.5_huber_missing']:
#     if deleted in name_step_result:# or 'weighted_7_gauss_px0.5_huber_missing' not in deleted:
#         del name_step_result[deleted]

# for deleted in list(name_step_result.keys()):
#     if 'weighted_7_gauss_px0.5_huber_missing' in deleted:
#         del name_step_result[deleted]


# showed = ['weighted_7_gauss_px0.5_huber_512_missing', 
#           'weighted_7_gauss_px0.5_huber_512_im2_missing', 
#           'weighted_7_nlgauss_px0.5_huber_512_missing']

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
        #   'weighted_7_quantile_512_9_heads',
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


showed = [x + "_missing" for x in showed]

for deleted in list(name_step_result.keys()):
    if deleted not in showed:
        print("deleted:", deleted)
        del name_step_result[deleted]
    # if "-v" in deleted:
    #     print("deleted:", deleted)
    #     del name_step_result[deleted]

plt.figure(figsize=(6, 8))
plt.subplot(2, 1, 1)

plt.title("Normalized MAE on Monash")
for n in name_step_result:
    # if n == 'first_run' or 'unweigh' in n or 'full' in n: continue
    if 'rd' not in n:
        y = [0.7054353364902101]
        x = [0]
    else:
        y = []
        x = []
    for step, result in sorted(name_step_result[n].items(), key=lambda x: int(x[0])):
        x.append(int(step))
        y.append(result[0])
    print(n, y)
    print(x)
    # plt.plot(x, y, label=n, markersize=5, marker='o')
    
    # Plot the line and get its color
    line, = plt.plot(x, y, label=n, markersize=5, marker='o')
    color = line.get_color()  # Get the color of the line
    # Add text labels near each point
    for xi, yi in zip(x, y):
        plt.text(xi+0.1, yi+0.005, f'{yi:.3f}', color=color, fontsize=5,
                 ha='right', va='bottom')  # Adjust text alignment as needed
    
plt.xlabel("Training Step")
plt.grid()
plt.legend(fontsize=7)

plt.subplot(2, 1, 2)


plt.title("Loss% v.s zero-shot")

for i, n in enumerate(name_step_result):
    # if n == 'first_run' or 'unweigh' in n or 'full' in n: continue
    y = []
    x = []
    for step, result in sorted(name_step_result[n].items(), key=lambda x: int(x[0])):
        x.append(int(step))
        y.append(result[1] + i / 1000)
    print(n, y)
    plt.plot(x, y, label=n, markersize=5, marker='o')
plt.grid()

plt.tight_layout()
plt.savefig("result.pdf")
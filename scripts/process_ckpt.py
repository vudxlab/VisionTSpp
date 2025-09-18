import os
import torch

# N = 100
# p_good = torch.rand((1, N))
# p_good = p_good / torch.sum(p_good)
# p_bad = torch.rand((1, N))
# p_bad = p_bad / torch.sum(p_bad)


# for eps in range(1, 100):
#     eps = 10 ** (-eps)
#     p_t = torch.ones((1, N)) * eps
#     p_t[0, 0] = 1
#     p_t = p_t / torch.sum(p_t)
#     # print(p_t)
#     print(f"eps = {eps}", torch.sum((p_good - p_bad) * torch.log(p_t) + p_bad * torch.log(p_bad) + p_good * torch.log(p_good)))


def process_ckpt_files(input_dir_list):
    processed_files = []
    
    # 递归扫描目录
    for input_dir in input_dir_list:
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.endswith(".ckpt") and 'last' not in file and "processed_" not in file:
                    ckpt_path = os.path.join(root, file)
                    output_file = os.path.join(root, f"processed_{file}")
                    # processed_files.append(output_file)
                    if os.path.exists(output_file):
                        if not os.path.exists(output_file.replace('.ckpt', '.csv')) or \
                          not os.path.exists(output_file.replace('.ckpt', '_pf.csv')) or \
                          not os.path.exists(output_file.replace('.ckpt', '_missing.csv')) or \
                          not os.path.exists(output_file.replace('.ckpt', '_pf_missing.csv')) or \
                          not os.path.exists(output_file.replace('.ckpt', '_ltsf_pred192.csv')) or \
                          not os.path.exists(output_file.replace('.ckpt', '_quantile.csv')) or \
                          not os.path.exists(output_file.replace('.ckpt', '_pf_quantile.csv')) or \
                          not os.path.exists(output_file.replace('.ckpt', '_ltsf_pred192_quantile.csv')):
                            # 只要有一个没完成，就记录处理一下
                            # print("Processing: ", ckpt_path, "->", output_file)
                            # print(not os.path.exists(output_file.replace('.ckpt', '.csv')), \
                            #     not os.path.exists(output_file.replace('.ckpt', '_pf.csv')), \
                            #     not os.path.exists(output_file.replace('.ckpt', '_missing.csv')), \
                            #     not os.path.exists(output_file.replace('.ckpt', '_pf_missing.csv')))
                            processed_files.append(output_file)
                        continue
                    try:
                        # 加载ckpt文件
                        # checkpoint = torch.load(ckpt_path, map_location="cpu")
                        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                        
                        # 提取state_dict
                        if 'state_dict' in checkpoint or 'epoch=0-step=00000.ckpt' in file:
                            if 'epoch=0-step=00000.ckpt' in file:
                                new_dict = checkpoint
                            else:
                                new_dict = {"model": {}}
                                state_dict = checkpoint['state_dict']
                                for n, p in state_dict.items():
                                    new_dict['model'][n.replace("module.", "")] = p
                                del new_dict['model']['mask']
                            
                            # 构造新文件路径
                            
                            # 保存state_dict到新的文件
                            torch.save(new_dict, output_file)
                            # print("processed_files:", processed_files, "\n")
                            
                            # 记录处理过的文件路径
                            processed_files.append(output_file)
                            
                            print(f"Processed: {ckpt_path} -> {output_file}")
                    except Exception as e:
                        raise
                        print(f"Error processing {ckpt_path}: {e}")
    
    # 将处理过的文件路径保存到ckpt_path.txt
    with open("ckpt_path.txt", "w") as f:
        # print("processed_files:", processed_files, "\n")
        for path in processed_files:
            # print("path:", path)
            # 只要有一个没完成，就记录处理一下
            # if not os.path.exists(path.replace(".ckpt", ".csv")):
            if not os.path.exists(path.replace('.ckpt', '.csv')) or \
              not os.path.exists(path.replace('.ckpt', '_pf.csv')) or \
              not os.path.exists(path.replace('.ckpt', '_missing.csv')) or \
              not os.path.exists(path.replace('.ckpt', '_pf_missing.csv')) or \
              not os.path.exists(output_file.replace('.ckpt', '_ltsf_pred192.csv')) or \
              not os.path.exists(output_file.replace('.ckpt', '_quantile.csv')) or \
              not os.path.exists(output_file.replace('.ckpt', '_pf_quantile.csv')) or \
              not os.path.exists(output_file.replace('.ckpt', '_ltsf_pred192_quantile.csv')):
                # print("Here")
                f.write(f"{path}\n")
    
    print("Processing completed. File paths saved in 'ckpt_path.txt'.")

# 使用示例

# input_directory = "/home/lefeishen/VisionTSpp/outputs/pretrain/visionts"  # 修改为你的输入目录路径
# input_directory = "/home/lefeishen/VisionTSpp/outputs/pretrain/visionts_nopre/lotsa_v1_weighted/weighted_5_rd"  # 修改为你的输入目录路径

# input_directory = ["/home/lefeishen/VisionTSpp/outputs/pretrain/visionts/lotsa_v1_weighted/weighted_7_gauss_px0.5_huber_512_top5", 
#                     "/home/lefeishen/VisionTSpp/outputs/pretrain/visionts/lotsa_v1_weighted/weighted_7_gauss_px0.5_huber_512_top7", 
#                     "/home/lefeishen/VisionTSpp/outputs/pretrain/visionts/lotsa_v1_weighted/weighted_7_gauss_px0.5_huber_512", 
#                     "/home/lefeishen/VisionTSpp/outputs/pretrain/visionts/lotsa_v1_weighted/weighted_7_gauss_px0.2_huber_512", 
#                     "/home/lefeishen/VisionTSpp/outputs/pretrain/visionts/lotsa_v1_weighted/weighted_7_gauss_px0.5_huber", 
#                     "/home/lefeishen/VisionTSpp/outputs/pretrain/visionts/lotsa_v1_weighted/weighted_7_gauss", 
#                     "/home/lefeishen/VisionTSpp/outputs/pretrain/visionts/lotsa_v1_weighted/weighted_7"]

# input_directory = ["/home/lefeishen/VisionTSpp/outputs/pretrain/visionts/lotsa_v1_weighted_image/weighted_7_gauss_px0.5_huber_2048_im2/checkpoints", 
#                    "/home/lefeishen/VisionTSpp/outputs/pretrain/visionts/lotsa_v1_weighted/weighted_7_gauss_px0.5_huber_2048"]

input_directory = [
                #    "/home/lefeishen/VisionTSpp/outputs/pretrain/visionts/lotsa_v1_weighted_image/weighted_7_gauss_px0.5_huber_2048_im2", 
                   "/home/lefeishen/VisionTSpp/outputs/pretrain/visionts/lotsa_v1_weighted/weighted_7_gauss_px0.5_huber_512_im3",
                   "/home/lefeishen/VisionTSpp/outputs/pretrain/visionts/lotsa_v1_weighted/weighted_7_gauss_px0.5_huber_512",
                   "/home/lefeishen/VisionTSpp/outputs/pretrain/visionts/lotsa_v1_weighted/weighted_7_gauss_px0.2_huber_512",
                   "/home/lefeishen/VisionTSpp/outputs/pretrain/visionts/lotsa_v1_weighted/weighted_7_gauss_px0.5_huber_2048",
                #    "/home/lefeishen/VisionTSpp/outputs/pretrain/visionts/lotsa_v1_weighted/weighted_7_quantile_px0.5_huber_2048",
                   "/home/lefeishen/VisionTSpp/outputs/pretrain/visionts/lotsa_v1_weighted/weighted_7_quantile_px0.5_huber_512",
                   "/home/lefeishen/VisionTSpp/outputs/pretrain/visionts/lotsa_v1_weighted/weighted_7_quantile_px0_512",
                   "/home/lefeishen/VisionTSpp/outputs/pretrain/visionts/lotsa_v1_weighted_image/weighted_7_quantile_px0.5_quantile_512_im0.2",
                   "/home/lefeishen/VisionTSpp/outputs/pretrain/visionts/lotsa_v1_weighted_image/weighted_7_quantile_px0.5_quantile_1024_im0.2",
                   "/home/lefeishen/VisionTSpp/outputs/pretrain/visionts/lotsa_v1_weighted_image/weighted_7_quantile_px0.5_quantile_1024_im0.4", 
                #    "/home/lefeishen/VisionTSpp/outputs/pretrain/visionts/lotsa_v1_weighted_image/weighted_7_quantile_px0.5_quantile_2048_im0.2", 
                   "/home/lefeishen/VisionTSpp/outputs/pretrain/visionts_large/lotsa_v1_weighted/weighted_7_1024_large", 
                #    "/home/lefeishen/VisionTSpp/outputs/pretrain/visionts_huge/lotsa_v1_weighted/weighted_7_512_huge", 
                   "/home/lefeishen/VisionTSpp/outputs/pretrain/visionts_huge/lotsa_v1_weighted/weighted_7_1024_huge", 
                   "/home/lefeishen/VisionTSpp/outputs/pretrain/visionts/lotsa_v1_weighted/weighted_7_quantile_1024", 
                   "/home/lefeishen/VisionTSpp/outputs/pretrain/visionts/lotsa_v1_weighted/weighted_7_quantile_512_clip_input",
                   "/home/lefeishen/VisionTSpp/outputs/pretrain/visionts/lotsa_v1_weighted/weighted_7_quantile_512_clip_input_new",
                   "/home/lefeishen/VisionTSpp/outputs/pretrain/visionts/lotsa_v1_weighted/weighted_7_quantile_512_complete_no_clip",
                   "/home/lefeishen/VisionTSpp/outputs/pretrain/visionts/lotsa_v1_weighted/weighted_7_quantile_1024_clip_input",
                   "/home/lefeishen/VisionTSpp/outputs/pretrain/visionts/lotsa_v1_weighted/weighted_7_quantile_512_filter",
                   "/home/lefeishen/VisionTSpp/outputs/pretrain/visionts/lotsa_v1_weighted/weighted_7_quantile_1024_filter",
                   "/home/lefeishen/VisionTSpp/outputs/pretrain/visionts/lotsa_v1_weighted/weighted_7_quantile_512_multi",
                   "/home/lefeishen/VisionTSpp/outputs/pretrain/visionts/lotsa_v1_weighted/weighted_7_quantile_512_multi_color",
                   "/home/lefeishen/VisionTSpp/outputs/pretrain/visionts/lotsa_v1_weighted/weighted_7_quantile_512_multi_color_9_heads",
                   "/home/lefeishen/VisionTSpp/outputs/pretrain/visionts_large/lotsa_v1_weighted/weighted_7_quantile_512_multi_color_9_heads_large",
                   "/home/lefeishen/VisionTSpp/outputs/pretrain/visionts/lotsa_v1_weighted/weighted_7_quantile_512_9_heads",
                   "/home/lefeishen/VisionTSpp/outputs/pretrain/visionts_huge/lotsa_v1_weighted/weighted_7_quantile_512_multi_color_9_heads_huge",
                   "/home/lefeishen/VisionTSpp/outputs/pretrain/visionts/lotsa_v1_weighted/weighted_7_quantile_512_multi_color_9_heads_rand_init",
                   "/home/lefeishen/VisionTSpp/outputs/pretrain/visionts/lotsa_v1_weighted/weighted_7_quantile_128_multi_color_9_heads_rand_init",
                   "/home/lefeishen/VisionTSpp/outputs/pretrain/visionts/lotsa_v1_weighted/weighted_7_quantile_512_multi_color_9_heads_ln_only",
                   
                   # 250909 adds:
                   "/home/lefeishen/VisionTSpp/outputs/pretrain/visionts/lotsa_v1_weighted/weighted_7_quantile_512_multi_color_9_heads_gift_eval"
                    
                #    "/home/lefeishen/VisionTSpp/outputs/pretrain/visionts/lotsa_v1_weighted_image/weighted_7_quantile_px0.5_huber_512_im3",
                  ]

process_ckpt_files(input_directory)

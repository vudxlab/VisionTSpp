import time
import subprocess

while True:
    cmd = [
        "python", "process_ckpt.py"
    ]
    subprocess.run(cmd)
    if open("ckpt_path.txt").read().strip() != "":
        cmd = [
            "python", "batch_evaluate_monash.py", 
            # "python", "batch_evaluate_monash_new.py",  # 250322 adds
            "0", "1"
        ]
        subprocess.run(cmd)
        cmd = [
            "python", "batch_evaluate_monash_withmissing.py", 
            # "/home/mouxiangchen/uni2ts/outputs/pretrain/visionts/lotsa_v1_weighted"
            "/home/lefeishen/VisionTSpp/outputs/pretrain/visionts/lotsa_v1_weighted"
        ]
        subprocess.run(cmd)
        cmd = [
            "python", "batch_evaluate_monash_withmissing.py", 
            "/home/lefeishen/VisionTSpp/outputs/pretrain/visionts/lotsa_v1_weighted_image"
        ]
        # ! 250401 adds: 新增mae_large!!
        subprocess.run(cmd)
        cmd = [
            "python", "batch_evaluate_monash_withmissing.py", 
            "/home/lefeishen/VisionTSpp/outputs/pretrain/visionts_large/lotsa_v1_weighted"
        ]
        subprocess.run(cmd)
        # subprocess.run(cmd)
        # cmd = [
        #     "python", "batch_evaluate_monash_withmissing.py", 
        #     "/home/lefeishen/VisionTSpp/outputs/pretrain/visionts_large/lotsa_v1_weighted_image"
        # ]
        cmd = [
            "python", "batch_evaluate_monash_withmissing.py", 
            "/home/lefeishen/VisionTSpp/outputs/pretrain/visionts_huge/lotsa_v1_weighted"
        ]
        subprocess.run(cmd)
        
        cmd = [
            "python", "batch_evaluate_monash_csv.py", 
        ]
        subprocess.run(cmd)
        
        cmd = [
            "python", "batch_evaluate_monash_csv_pf.py", 
        ]
        subprocess.run(cmd)
        
        # ! 250501 adds:
        cmd = [
            "python", "batch_evaluate_monash_csv_pf_crps_msis.py", 
        ]
        subprocess.run(cmd)
        
        # ! 250315 adds:
        cmd = [
            "python", "batch_evaluate_monash_csv_ltsf.py", 
        ]
        subprocess.run(cmd)
    time.sleep(60)
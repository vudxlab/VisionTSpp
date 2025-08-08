
for ds in ETTh2 ETTm2; do
  for cl in 1728 2304 2880 4032; do
    for pl in 96 192 336 720; do
        CUDA_VISIBLE_DEVICES=2 python run_visionts.py --dataset=${ds}  --run_name=visionts --save_dir=lsf_results --test_setting=lsf --model=visionts --periodicity=freq --context_len=${cl} --pred_length=${pl} --batch_size 256 --split_name=val

        CUDA_VISIBLE_DEVICES=2 python run_visionts.py --dataset=${ds}  --run_name=visionts --save_dir=lsf_results --test_setting=lsf --model=visionts --periodicity=freq --context_len=${cl} --pred_length=${pl} --batch_size 256 --split_name=test
      done
    done
done

# for pl in 96 192 336 720; do
#   python run_visionts.py --dataset=ETTh2  --run_name=visionts --save_dir=lsf_results --test_setting=lsf --model=visionts --periodicity=freq --context_len=1728 --pred_length=${pl} --batch_size 512
# done

# for pl in 96 192 336 720; do
#   python run_visionts.py --dataset=ETTm1  --run_name=visionts --save_dir=lsf_results --test_setting=lsf --model=visionts --periodicity=freq --context_len=2304 --pred_length=${pl} --batch_size 512
# done

# for pl in 96 192 336 720; do
#   python run_visionts.py --dataset=ETTm2  --run_name=visionts --save_dir=lsf_results --test_setting=lsf --model=visionts --periodicity=freq --context_len=4032 --pred_length=${pl} --batch_size 512
# done

# for pl in 96 192 336 720; do
#   python run_visionts.py --dataset=electricity  --run_name=visionts --save_dir=lsf_results --test_setting=lsf --model=visionts --periodicity=freq --context_len=2880 --pred_length=${pl} --batch_size 512
# done

# for pl in 96 192 336 720; do
#   python run_visionts.py --dataset=weather  --run_name=visionts --save_dir=lsf_results --test_setting=lsf --model=visionts --periodicity=freq --context_len=4032 --pred_length=${pl} --batch_size 512
# done

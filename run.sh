# Run one of the following commands to start training:
# The MAE-base model is recommended.

# base model: 
# setting: "multivariate + color_list + 9 quantile heads", batch_size=512
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m cli.train -cp conf/pretrain run_name=weighted_7_quantile_512_multi_color_9_heads  model=visionts data=lotsa_v1_weighted

# large model:
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m cli.train -cp conf/pretrain run_name=weighted_7_quantile_512_multi_color_9_heads_large  model=visionts_large data=lotsa_v1_weighted

# huge model:
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m cli.train -cp conf/pretrain run_name=weighted_7_quantile_512_multi_color_9_heads_huge  model=visionts_huge data=lotsa_v1_weighted
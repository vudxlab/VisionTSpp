import math
import sys
import os
import argparse
sys.path.append("../")


import einops
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from gluonts.model.forecast import SampleForecast, QuantileForecast
from gluonts.ev.metrics import MAE, MASE, MSE, ND, NRMSE, SMAPE, MSIS, MeanWeightedSumQuantileLoss
from gluonts.model.evaluation import evaluate_forecasts

from dataset import get_gluonts_test_dataset
from visionts import VisionTS, freq_to_seasonality_list

POSSIBLE_SEASONALITIES = {
    "S": [3600],  # 1 hour
    "T": [1440, 10080],  # 1 day or 1 week
    "H": [24],  # 1 day or 1 week
    "D": [7, 30, 365],  # 1 week, 1 month or 1 year
    "W": [52, 4], # 1 year or 1 month
    "M": [12, 6, 3], # 3 months, 6 months or 1 year
    "B": [5],
    "Q": [4, 2], # 6 months or 1 year
}


def imputation_nan(array):
    """
    Impute missing value using Naive forecasting.
    """
    not_nan_mask = ~np.isnan(array)
    if not_nan_mask.all():
        return array
    if not not_nan_mask.any():
        return np.zeros_like(array)

    array_imputed = np.copy(array)
    for i in range(len(array)):
        if not not_nan_mask[i]:
            array_imputed[i] = array_imputed[i - 1]
    return array_imputed


def forecast(model: VisionTS, train_list: list, test_list: list, batch_size, device, periodicity, num_patch_input, quantile):
    # We combine testing data with the context lengths
    seq_len_to_group_data = {}
    for i in range(len(train_list)):
        train_len = len(train_list[i])
        if train_len not in seq_len_to_group_data:
            seq_len_to_group_data[train_len] = [[], [], []] # index, input, output
        # 对于当前train_len，添加当前的index, input, output
        seq_len_to_group_data[train_len][0].append(i)
        seq_len_to_group_data[train_len][1].append(train_list[i])
        seq_len_to_group_data[train_len][2].append(test_list[i])
    
    forecast_np = {}  # raw index -> forecast
    forecast_np_quantiles = {}  # list of "raw index -> forecast quantiles"
    
    # 遍历每个train_len
    for train_len in seq_len_to_group_data:
        cur_idx_list, cur_train, cur_test = seq_len_to_group_data[train_len]
        convert = lambda array: torch.FloatTensor(
            einops.rearrange(np.array(array), 'b t -> b t 1')
        ).to(device)  # 就是在最后加一维nvars=1
        
        cur_train = convert(cur_train)  # shape: [sample_num=2247, context_len=1992, 1]
        cur_test = convert(cur_test)  # shape: [sample_num=2247, pred_len=24, 1]
        context_len = cur_train.shape[1]
        pred_len = cur_test.shape[1]
        
        # 更新模型设置？
        model.update_config(context_len=context_len, pred_len=pred_len, periodicity=periodicity, 
                            num_patch_input=num_patch_input, padding_mode='constant')

        # 按照batch大小遍历数据集？
        for batch_i in range(int(math.ceil(len(cur_train) / batch_size))):
            batch_start = batch_i * batch_size
            if batch_start >= len(cur_train):
                continue
            batch_end = batch_start + batch_size
            if batch_end > len(cur_train):
                batch_end = len(cur_train)

            cur_batch_train = cur_train[batch_start:batch_end]  # shape: [bs=256, context_len=1992, 1]
            if not quantile:
                cur_batch_pred = model(cur_batch_train, fp64=True)  # [bs=256, pred_len=24, 1]
            else:
                # cur_batch_pred, cur_batch_pred_25, cur_batch_pred_75 = model(cur_batch_train, fp64=True)
                cur_batch_pred, cur_batch_pred_quantile_list = model(cur_batch_train, fp64=True)
                num_quantiles = len(cur_batch_pred_quantile_list)
            
            for i in range(len(cur_batch_pred)):
                cur_idx = cur_idx_list[batch_start + i]  # 0, 1, 2, ..., 2246
                assert cur_idx not in forecast_np
                
                if not quantile:
                    forecast_np[cur_idx] = cur_batch_pred[i, :, 0].detach().cpu().numpy()
                else:
                    pred = cur_batch_pred[i, :, 0].detach().cpu()
                    # pred_25 = cur_batch_pred_25[i, :, 0].detach().cpu()
                    # pred_75 = cur_batch_pred_75[i, :, 0].detach().cpu()
                    
                    # ! 250430 adds: 这里如何返回最后结果上存在两个方案！！！
                    
                    # # ! solution 1: 用mean之后的结果返回？
                    # pred_quantile_list = [
                    #     cur_batch_pred_quantile_list[j][i, :, 0].detach().cpu() 
                    #     for j in range(len(cur_batch_pred_quantile_list))
                    # ]
                    
                    # all_preds = [pred] + pred_quantile_list  # 包含了所有预测值
                    # stacked_preds = torch.stack(all_preds)  # 堆叠张量，增加一个新维度
                    # mean_preds = torch.mean(stacked_preds, dim=0)  # 沿新维度（dim=0）计算平均值
                    # # ! 250430 finds bug: 尝试用emsemble之后的结果返回？
                    # # ! 但实际上应该只用pred值？因为我们希望用中位数而不是平均值来计算？
                    # forecast_np[cur_idx] = mean_preds.numpy()
                    
                    # ! solution 2: 用中位数？也即只返回pred自身！！
                    forecast_np[cur_idx] = pred.numpy()
                    
                    # ! 然后将各个分位数的结果同样存入forecast_np_quantile_list？
                    quantiles_list = []
                    for j in range(num_quantiles // 2):  # 10% - 40%
                        cur_batch_pred_quantile = cur_batch_pred_quantile_list[j][i, :, 0].detach().cpu().numpy()
                        quantiles_list.append(cur_batch_pred_quantile)
                    quantiles_list.append(pred.numpy())  # 50%
                    for j in range(num_quantiles // 2, num_quantiles):  # 60% - 90%
                        cur_batch_pred_quantile = cur_batch_pred_quantile_list[j][i, :, 0].detach().cpu().numpy()
                        quantiles_list.append(cur_batch_pred_quantile)
                    
                    quantiles_np = np.array(quantiles_list)  # [num_quantiles+1=9, pred_len=24]
                    forecast_np_quantiles[cur_idx] = quantiles_np
    
    # 最后得到总的预测值
    if not quantile:
        # 点预测：[sample_num=2247, pred_len=24]
        return np.stack([forecast_np[i] for i in range(len(train_list))])
    else:
        # 概率分布预测：[sample_num=2247, num_quantiles=9, pred_len=24]
        return np.stack([forecast_np[i] for i in range(len(train_list))]), \
            np.stack([forecast_np_quantiles[i] for i in range(len(train_list))])


def convert_context_len(context_len, no_periodicity_context_len, periodicity):
    if periodicity == 1:
        context_len = no_periodicity_context_len
    # Round context length to the integer multiples of the period
    context_len = int(round(context_len / periodicity)) * periodicity
    return context_len


def evaluate(
    dataset,
    save_path,
    context_len,
    no_periodicity_context_len,
    num_patch_input=None,
    device="cuda:0",
    checkpoint_dir="./ckpt",
    checkpoint_path=None,
    mae_arch="mae_base",
    batch_size=512,
    periodicity="autotune",
    quantile_int=0,
    # clip_input=False,
    clip_input=0,
    complete_no_clip=False,
    multivariate=0,
    color=0,
):
    # model = VisionTS(mae_arch, ckpt_dir=checkpoint_dir, ckpt_path=checkpoint_path).to(device)
    
    # ! 20250416 adds:
    quantile_bool = True if quantile_int > 0 else False
    # clip_input_bool = True if clip_input > 0 else False
    complete_no_clip_bool = True if complete_no_clip > 0 else False
    color_bool = True if color > 0 else False
    
    model = VisionTS(mae_arch, ckpt_dir=checkpoint_dir, ckpt_path=checkpoint_path, quantile=quantile_bool, 
                     clip_input=clip_input, complete_no_clip=complete_no_clip_bool, color=color_bool).to(device)
    
    datasets_ltsf = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Electricity", "Weather"]
    # test_data, metadata = get_gluonts_test_dataset(dataset)
    if dataset not in datasets_ltsf:
        # 用glonts读
        test_data, metadata = get_gluonts_test_dataset(dataset)
    else:
        pass
        # # 用本地数据读！
        # from gluonts.dataset.common import ListDataset
        # from gluonts.dataset.field_names import FieldName
        # # 加载LTSF数据集并提取测试集
        # def load_ltsf_test_data(file_path, target_column, freq, test_ratio=0.2):
        #     df = pd.read_csv(file_path)
            
        #     # 确保日期列为 datetime 类型
        #     df['date'] = pd.to_datetime(df['date'])
            
        #     # 分割训练集和测试集
        #     total_length = len(df)
        #     test_length = int(total_length * test_ratio)
        #     test_df = df.iloc[-test_length:]  # 取最后 20% 作为测试集
            
        #     # 提取目标变量和时间戳
        #     start_time = test_df['date'].iloc[0]  # 测试集的起始时间
        #     target_series = test_df[target_column].values.tolist()  # 目标列
            
        #     # 如果有协变量，提取协变量
        #     dynamic_features = [
        #         test_df[col].values.tolist() for col in df.columns if col not in ['date', target_column]
        #     ]
            
        #     # 创建 GluonTS 格式的测试集
        #     test_dataset = ListDataset(
        #         [
        #             {
        #                 FieldName.START: start_time,
        #                 FieldName.TARGET: target_series,
        #                 FieldName.FEAT_DYNAMIC_REAL: dynamic_features,  # 动态特征（可选）
        #             }
        #         ],
        #         freq=freq  # 时间频率（如 "H" 表示小时级）
        #     )
            
        #     return test_dataset

        # # 示例：加载 ETTh1 测试集
        # file_path = "ETTh1.csv"  # 替换为你的文件路径
        # target_column = "OT"     # 目标列（Out Temperature）
        # freq = "H"               # 小时级频率

        # test_dataset = load_ett_test_data(file_path, target_column, freq)

        # # 查看测试集内容
        # for entry in test_dataset:
        #     print("Start Time:", entry[FieldName.START])
        #     print("Target Length:", len(entry[FieldName.TARGET]))
        #     print("Dynamic Features Shape:", 
        #         [(len(feat),) for feat in entry.get(FieldName.FEAT_DYNAMIC_REAL, [])])

        
    print("test_data:", test_data)
    print("metadata:", metadata)
    print("test_data.input:", test_data.input)
    print("test_data.label:", test_data.label)
    
    
    data_train = [imputation_nan(x['target']) for x in test_data.input]
    data_test = [x['target'] for x in test_data.label]
    pred_len = len(data_test[0])
    
    if periodicity == "autotune":
        # ! 如果是autotune的话，那么会根据采样频率先获得所有可能的周期
        # ! 然后遍历eval后选择最佳周期作为最终的周期。
        seasonality_list = freq_to_seasonality_list(metadata.freq, POSSIBLE_SEASONALITIES)
        best_valid_mae = float('inf')
        best_valid_p = 1
        for periodicity in tqdm(seasonality_list, desc='validate seasonality'):
            cur_context_len = convert_context_len(context_len, no_periodicity_context_len, periodicity)

            val_train = [x[-cur_context_len-pred_len:-pred_len] for x in data_train]
            val_test = [x[-pred_len:] for x in data_train]
            
            # 主要的预测函数！
            # val_train 和 val_test 分别为长为ctx_len和pred_len的tensors！！
            val_forecast = forecast(model, val_train, val_test, batch_size, device, periodicity, num_patch_input, quantile=quantile_bool)
            # ! 250503 adds: 增加对quantile_bool的判断
            if quantile_bool:
                val_forecast, val_forecast_quantiles  = val_forecast  # forecast_quantiles.shape: [sample_num=2247, num_quantiles=9, pred_len=24]
            
            
            val_mae = np.abs(np.asarray(val_test) - val_forecast).mean()
            if val_mae < best_valid_mae:
                best_valid_p = periodicity
                best_valid_mae = val_mae
                tqdm.write(f"autotune: P = {periodicity} | valid mae = {val_mae}, accept!")
            else:
                tqdm.write(f"autotune: P = {periodicity} | valid mae = {val_mae}, reject!")
        periodicity = best_valid_p
    elif periodicity == "freq":
        # ! 如果是freq的话，那么默认选取list的第一个，一般也是最常用的那个周期？
        periodicity = freq_to_seasonality_list(metadata.freq, POSSIBLE_SEASONALITIES)[0]
    else:
        # ! 如果是整数的话，那么直接使用该整数作为周期
        periodicity = int(periodicity)

    with open("p.txt", "a+") as f:
        f.write(f"{dataset},{metadata.freq},{periodicity}\n")
    cur_context_len = convert_context_len(context_len, no_periodicity_context_len, periodicity)  # 将context_len规约到periodicity的整数倍；如果周期为1的话，则直接使用no_periodicity_context_len代替？
    train = [x[-cur_context_len:] for x in data_train]  # shape: (sample_num=2247, context_len=1992)
    
    forecast_values = forecast(model, train, data_test, batch_size, device, periodicity, num_patch_input, quantile=quantile_bool)  # shape: [sample_num=2247, pred_len=24]
    if quantile_bool:
        forecast_values, forecast_quantiles  = forecast_values  # forecast_quantiles.shape: [sample_num=2247, num_quantiles=9, pred_len=24]
        
    
    sample_forecasts = []
    quantile_sample_forecasts = []
    
    # ! 250430 adds: 在这里设置quantile_levels！！
    QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    QUANTILE_LEVELS_STR = [str(key) for key in QUANTILE_LEVELS]
    
    # for item, ts in zip(forecast_values, test_data.input):
    #     forecast_start_date = ts["start"] + len(ts["target"])  # 获取当前样本
    #     sample_forecasts.append(
    #         SampleForecast(
    #             samples=np.reshape(item, (1, -1)), start_date=forecast_start_date
    #         )  # 其shape为(1, pred_len=24)，对应于需要的(num_samples, prediction_length)或(num_samples, prediction_length, target_dim)
    #     )
    
    # ! 250503 adds: 增加计算各个分位数的预测值
    for idx, ts in enumerate(test_data.input):
        forecast_start_date = ts["start"] + len(ts["target"])
        # 1. 点预测 (用于传统指标)
        sample_forecasts.append(
            SampleForecast(
                samples=np.reshape(forecast_values[idx], (1, -1)),
                start_date=forecast_start_date
            )
        )
        # 2. 分位数预测 (用于分位数指标)
        if quantile_bool:
            quantile_sample_forecasts.append(
                QuantileForecast(
                    forecast_arrays=forecast_quantiles[idx],  # shape: (num_quantiles, pred_len, nvars)
                    start_date=forecast_start_date,
                    forecast_keys=QUANTILE_LEVELS_STR,  # 设定的分位数列表，如 [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                )
            )
            
    # 定义要评估的指标
    POINT_METRICS = [
        MSE(),
        MAE(),
        SMAPE(),
        MASE(),
        ND(),
        NRMSE(),
        MSIS(),
    ]
    QUANTILE_METRICS = [
        MeanWeightedSumQuantileLoss(quantile_levels=QUANTILE_LEVELS),
        # 可以添加其他分位数相关指标，如分位数覆盖度等
    ]
    
    # metrics_df = evaluate_forecasts(
    #     sample_forecasts,
    #     test_data=test_data,
    #     metrics=[
    #         MSE(),
    #         MAE(),
    #         SMAPE(),
    #         MASE(),
    #         ND(),
    #         NRMSE(),
    #         MSIS(),
    #         MeanWeightedSumQuantileLoss(quantile_levels=QUANTILE_LEVELS),
    #     ],
    # )
    
    # ! 250503 adds: 评估点预测指标
    point_metrics_df = evaluate_forecasts(
        sample_forecasts,
        test_data=test_data,
        metrics=POINT_METRICS,
    )

    # 评估分位数预测指标
    # for input_, label_, forecast_ in zip(test_data.input, test_data.label, quantile_sample_forecasts):
    #     print(input_, label_, forecast_)
    if quantile_bool:
        quantile_metrics_df = evaluate_forecasts(
            quantile_sample_forecasts,
            test_data=test_data,
            metrics=QUANTILE_METRICS,
        )
    
    
    # # 提取分位数预测值
    # quantile_forecasts_array = forecast_quantiles.transpose(0, 2, 1)  # 形状为 (sample_num=2247, pred_len=24, num_quantiles=9)
    # print(f"{quantile_forecasts_array.shape = }")

    # # 初始化 MeanWeightedSumQuantileLoss 指标
    # mwsql_metric = MeanWeightedSumQuantileLoss(quantile_levels=QUANTILE_LEVELS)

    # # 提取真实值
    # true_values = np.array([ts["target"][-pred_len:] for ts in test_data.input])  # 形状为 (sample_num=2247, pred_len=24)

    # # 计算 MeanWeightedSumQuantileLoss
    # mwsql_score = mwsql_metric(true_values, quantile_forecasts_array)
    # print(f"Mean Weighted Sum Quantile Loss (CRPS approximation): {mwsql_score}")

    
    # 合并结果
    if quantile_bool:
        metrics_df = pd.concat([point_metrics_df, quantile_metrics_df], axis=1)
    else:
        metrics_df = point_metrics_df.copy()
    print(f"{metrics_df = }")
    
    # # 插入 MeanWeightedSumQuantileLoss
    # metrics_df = point_metrics_df.copy()
    # metrics_df.insert(loc=len(metrics_df.columns), column='MeanWeightedSumQuantileLoss', value=[mwsql_score])
    
    metrics_df.insert(loc=0, column='dataset', value=[dataset])  # 在第一列插入dataset！！
    
    # 保存结果
    if os.path.exists(save_path):
        old_metrics_df = pd.read_csv(save_path)
        metrics_df = pd.concat([old_metrics_df, metrics_df], ignore_index=True)
    
    metrics_df.to_csv(save_path, index=False)
    
    print(metrics_df)
    print(f"Results saved to {save_path}")
    print("-" * 5, f"Evaluation of {dataset} complete", "-" * 5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate VisionTS on Monash or PF"
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Name of the dataset to use"
    )
    parser.add_argument(
        "--periodicity",
        type=str,
        required=True,
        help=(
            "Time series periodicity length. Can be the following param: "
            + "(1) 'autotune': find the best periodicity on the validation set based on frequency "
            + "(2) 'freq': use the pre-defined periodicity based on frequency "
            + "(3) An integer: use the given periodicity."
        ),
    )
    parser.add_argument(
        "--save_name", type=str, default="result.csv", help="Directory to save the results"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="../ckpt/",
        help="Dir to load the model. Auto download if not exists.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to load the model (Override --checkpoint_dir)",
    )
    parser.add_argument("--context_len", type=int, default=1000, help="Context length.")
    parser.add_argument(
        "--no_periodicity_context_len",
        type=int,
        default=1000,
        help="Context length for data with periodicity = 1.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=512, help="Batch size for generating samples"
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device. cuda or cpu"
    )
    parser.add_argument(
        "--num_patch_input", type=int, default=None, help="Number of input patches (X-axis) for MAE. None: compute with the ratio between context/pred length"
    )
    parser.add_argument(
        "--quantile", type=int, default=1, help="0: no quantile, 1: quantile"
    )
    parser.add_argument(
        "--clip_input", type=int, default=0, help="0: no clip input, 1: clip input"
    )
    parser.add_argument(
        "--complete_no_clip", type=int, default=0, help="0: clip, 1: complete_no_clip"
    )
    parser.add_argument(
        "--vm_arch", type=str, default='mae_base', help="mae_base, mae_large, mae_huge"
    )
    parser.add_argument(
        "--multivariate", type=int, default=0, help="0: univariate, 1: multivariate"
    )
    parser.add_argument(
        "--color", type=int, default=0, help="0: no color, 1: use color"
    )

    args = parser.parse_args()
    print('Args in experiment:')
    print(f"{args = }")

    with torch.no_grad():
        evaluate(
            args.dataset,
            args.save_name,
            context_len=args.context_len,
            no_periodicity_context_len=args.no_periodicity_context_len,
            num_patch_input=args.num_patch_input,
            device=args.device,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_path=args.checkpoint_path,
            mae_arch=args.vm_arch,
            batch_size=args.batch_size,
            periodicity=args.periodicity,
            quantile_int=args.quantile,
            clip_input=args.clip_input,
            complete_no_clip=args.complete_no_clip,
            multivariate=args.multivariate,
            color=args.color,
        )
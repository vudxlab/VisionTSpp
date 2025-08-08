#  Copyright (c) 2024, Salesforce, Inc.
#  SPDX-License-Identifier: Apache-2
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from enum import Enum
from typing import Any

import numpy as np
import os
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image
import torch

from uni2ts.common.sampler import Sampler, get_sampler
from uni2ts.common.typing import (
    BatchedData,
    BatchedDateTime,
    BatchedString,
    Data,
    FlattenedData,
    MultivarTimeSeries,
    UnivarTimeSeries,
)
from uni2ts.data.indexer import Indexer
from uni2ts.transform import Transformation
import torchvision.transforms as transforms
from uni2ts.common.env import env

import copy
import pprint
import json
from ..global_var import get_value, set_value, lock


class SampleTimeSeriesType(Enum):
    """
    How to sample from the dataset.
    - none: do not sample, return the current index.
    - uniform: each time series sampled with equal probability
    - proportional: each time series sampled with probability proportional to it's length
    """

    NONE = "none"
    UNIFORM = "uniform"
    PROPORTIONAL = "proportional"


class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        indexer: Indexer[dict[str, Any]],
        transform: Transformation,
        sample_time_series: SampleTimeSeriesType = SampleTimeSeriesType.NONE,
        dataset_weight: float = 1.0,
        IS_RECORD_DATA: bool = False,
        # IS_RECORD_DATA: bool = True,
        
        # DO_FILTER: bool = False,
        DO_FILTER: bool = True,
        
        # max_detect_iter = 30,
        max_detect_iter = 10,
        # max_detect_iter = 5,
    ):
        """
        :param indexer: Underlying Indexer object
        :param transform: Transformation to apply to time series
        :param sample_time_series: defines how a time series is obtained from the dataset
        :param dataset_weight: multiplicative factor to apply to dataset size
        """
        self.indexer = indexer
        self.transform = transform
        self.sample_time_series = sample_time_series
        self.dataset_weight = dataset_weight
        
        self.IS_RECORD_DATA = IS_RECORD_DATA
        self.DO_FILTER = DO_FILTER
        self.max_detect_iter = max_detect_iter

        if sample_time_series == SampleTimeSeriesType.NONE:
            self.probabilities = None
        elif sample_time_series == SampleTimeSeriesType.UNIFORM:
            self.probabilities = indexer.get_uniform_probabilities()
        elif sample_time_series == SampleTimeSeriesType.PROPORTIONAL:
            self.probabilities = indexer.get_proportional_probabilities()
        else:
            raise ValueError(f"Unknown sample type {sample_time_series}")

    def __getitem__(self, idx: int) -> dict[str, FlattenedData]:
        """
        Obtain a time series from the dataset, flatten
        :param idx: index of time series to retrieve. if sample_time_series is specified, this will be ignored.
        :return: transformed time series data
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(
                f"Index {idx} out of range for dataset of length {len(self)}"
            )

        if self.sample_time_series != SampleTimeSeriesType.NONE:
            idx = np.random.choice(len(self.probabilities), p=self.probabilities)

        # 原来流程：(1)根据下标idx获得数据，(2)然后展平，(3)最后做一份transform。
        data = self._get_data(idx)
        is_tuple = False
        if isinstance(data, tuple):
            is_tuple = True
            data, dataset_name = data
        
        # # # ! 250417 adds: 
        # # # ! 这里的data_copy是为了防止原数据data在transfomations中被修改了，所以必须深拷贝来备份一份
        # data_copy = copy.deepcopy(data)
        
        flatten_data = self._flatten_data(data)
        
        result = self.transform(flatten_data)
        
        
        total_samples = 0
        filter_samples = 0
        
        
        # ! 20250418 adds: 判断是否超出上下界，并记录在文件里
        if self.IS_RECORD_DATA:
            # * 1. 计算当前数据是否超出了threshold边界？用target_out和target_img_out记录。
            target_data = torch.tensor(result["target"])
            target_img_data = torch.tensor(result["target_img"])
            thres_down, thres_up = -1.8044, 2.2489
            # thres_down, thres_up = -5, 5
            
            target_out = torch.any((target_data > thres_up)) or torch.any((target_data < thres_down))
            target_img_out = torch.any((target_img_data > thres_up)) or torch.any((target_img_data < thres_down))
            # target_out = np.any(target_data > thres_up) or np.any(target_data < thres_down)
            # target_img_out = np.any(target_img_data > thres_up) or np.any(target_img_data < thres_down)
            
            # print(f"target_data > {thres_up}: {torch.any((target_data > thres_up))}, target_data < {thres_down}: {torch.any((target_data < thres_down))}")
            
            # * 2. 获得原来的total_samples、filter_samples；dataset_count、dataset_filter
            # * 前者加1，后者根据是否被filter掉了来记录
            # * dataset同理记录一下
            total_samples = get_value("total_samples")
            filter_samples = get_value("filter_samples")
            dataset_count: dict = get_value("dataset_count")
            dataset_filter: dict = get_value("dataset_filter")
            
            total_samples += 1
            # !可以优化成一行：
            dataset_count[dataset_name] = dataset_count.get(dataset_name, 0) + 1
            
            if target_out or target_img_out:
            # if target_out:
                filter_samples += 1
                dataset_filter[dataset_name] = dataset_filter.get(dataset_name, 0) + 1
        

        # ! 20250418 adds: 判断是否需要做filter
        if self.DO_FILTER:
            cnt = 0
            while cnt < self.max_detect_iter:
                
                # * 1. 计算当前数据是否超出了threshold边界？用target_out和target_img_out记录。
                target_data = torch.tensor(result["target"])
                # target_img_data = torch.tensor(result["target_img"])
                thres_down, thres_up = -1.8044, 2.2489
                # thres_down, thres_up = -5, 5
                
                target_out = torch.any((target_data > thres_up)) or torch.any((target_data < thres_down))
                # target_img_out = torch.any((target_img_data > thres_up)) or torch.any((target_img_data < thres_down))
                
                # if not (target_out or target_img_out):
                if not target_out:
                    break
                else:
                    # ! 250418 adds:
                    cnt += 1
                    
                    # TODO: 这里对于MultiSampleTimeSeriesDataset的时候，他会根据当前idx的大小去采样。所以这里idx的大小必须是严格小于self.num_ts的！！
                    # idx = np.random.choice(self.__len__())
                    idx = np.random.choice(self.num_ts)
                    # idx = np.random.choice(len(self.probabilities), p=self.probabilities)
                    data = self._get_data(idx)
                    total_samples += 1
                    filter_samples += 1
                    
                    
                    if isinstance(data, tuple):
                        data, dataset_name = data
                    # deprecated: # TODO: 这里的data_copy是为了防止原数据data在transfomations中被修改了，所以备份一份！！！
                    flatten_data = self._flatten_data(data)
                    result = self.transform(flatten_data)
                    
                    continue
                
            # print("cnt: ", cnt)
        
        
        if self.IS_RECORD_DATA:
            # * 3. 别忘记set_value！！
            set_value("total_samples", total_samples)
            set_value("filter_samples", filter_samples)
            set_value("dataset_count", dataset_count)
            set_value("dataset_filter", dataset_filter)
            
            # * 4. 每隔一段时间打印一下
            if total_samples % 1000 == 0:
            # if total_samples % 300 == 0:
                # * 4.1 计算各个数据集被筛选比例
                filtered_ratios = {}
                for key in dataset_count.keys():
                    filtered_ratios[key] = dataset_filter.get(key, 0) / dataset_count[key]
                
                # 按值从大到小排序
                sorted_ratios = sorted(filtered_ratios.items(), key=lambda item: item[1], reverse=True)
                sorted_ratios = dict(sorted_ratios)
                
                # 并保存在文件中
                file_path = "sorted_ratios.txt"
                # formatted_dict = pprint.pformat(sorted_ratios)  # 格式化字典为字符串
                # with open(file_path, "w", encoding="utf-8") as file:
                #     file.write(formatted_dict)
                
                def read_file(file, filter_ratio_dict, filter_num_dict, total_num_dict):
                    for line in file:
                        line = line.strip()
                        if not line: continue
                        try:
                            key, value = line.split(": ", 1)
                            value, filter_num, total_num = value.split(", ")
                            value = float(value)
                            filter_num = int(filter_num); total_num = int(total_num)
                            
                            filter_ratio_dict[key] = value
                            filter_num_dict[key] = filter_num; total_num_dict[key] = total_num
                        except Exception as e:
                            print("Error parsing line:", line)
                            print("Error:", e)
                            continue
                
                filter_ratio_dict = {}
                filter_num_dict = {}; total_num_dict = {}
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8')as file:
                        # json_data = json.load(file)
                        read_file(file, filter_ratio_dict, filter_num_dict, total_num_dict)
                    
                    for key in sorted_ratios.keys():
                        filter_num_dict[key] = filter_num_dict.get(key, 0) + dataset_filter.get(key, 0)
                        total_num_dict[key] = total_num_dict.get(key, 0) + dataset_count[key]
                        filter_ratio_dict[key] = filter_num_dict[key] / total_num_dict[key]
                    
                    sorted_filter_ratio = sorted(filter_ratio_dict.items(), key=lambda item: item[1], reverse=True)
                    sorted_filter_ratio = dict(sorted_filter_ratio)
                else:
                    sorted_filter_ratio = sorted_ratios
                
                pprint.pprint(sorted_filter_ratio)
                
                with open(file_path, "w", encoding="utf-8") as file:
                    # json.dump(sorted_json_data, file, indent=4)
                    for key in sorted_filter_ratio.keys():
                        value, filter_num, total_num = sorted_filter_ratio[key], filter_num_dict.get(key, 0), total_num_dict.get(key, 0)
                        file.write(f"{key}: {value}, {filter_num}, {total_num}\n")
                
            
                # * 4.2 打印样本数
                print(f"total_samples: {total_samples}, filter_samples: {filter_samples}, filter_ratio = {filter_samples / total_samples}")
                # print(f"total_samples: {total_samples}, filter_samples: {filter_samples}, saving_ratio = {1.0 - filter_samples / total_samples}")  
        
        
        
        # if is_tuple:
        #     result["dataset_name"] = dataset_name

        return result
        # return result, dataset_name

    @property
    def num_ts(self) -> int:
        """
        Get the number of time series in the dataset
        """
        return len(self.indexer)

    def __len__(self) -> int:
        """
        Length is the number of time series multiplied by dataset_weight
        """
        return int(np.ceil(self.num_ts * self.dataset_weight))

    def _get_data(self, idx: int) -> dict[str, Data | BatchedData]:
        """
        Obtains time series from Indexer object
        """
        return self.indexer[idx % self.num_ts]

    @staticmethod
    def _flatten_data(data: dict[str, Data]) -> dict[str, FlattenedData]:
        """
        Convert time series type data into a list of univariate time series
        """
        return {
            k: (
                [v]
                if isinstance(v, UnivarTimeSeries)
                else list(v) if isinstance(v, MultivarTimeSeries) else v
            )
            for k, v in data.items()
        }


class MultiSampleTimeSeriesDataset(TimeSeriesDataset):
    """
    Samples multiple time series and stacks them into a single time series.
    Underlying dataset should have aligned time series, meaning same start and end dates.
    """

    def __init__(
        self,
        indexer: Indexer[dict[str, Any]],
        transform: Transformation,
        max_ts: int,
        combine_fields: tuple[str, ...],
        sample_time_series: SampleTimeSeriesType = SampleTimeSeriesType.NONE,
        dataset_weight: float = 1.0,
        sampler: Sampler = get_sampler("beta_binomial", a=2, b=5),
    ):
        """
        :param indexer: Underlying Indexer object
        :param transform: Transformation to apply to time series
        :param max_ts: maximum number of time series that can be stacked together
        :param combine_fields: fields which should be stacked
        :param sample_time_series: defines how a time series is obtained from the dataset
        :param dataset_weight: multiplicative factor to apply to dataset size
        :param sampler: how to sample the other time series
        """
        super().__init__(indexer, transform, sample_time_series, dataset_weight)
        self.max_ts = max_ts
        self.combine_fields = combine_fields
        self.sampler = sampler

    def _get_data(self, idx: int) -> dict[str, BatchedData]:
        n_series = self.sampler(min(self.num_ts, self.max_ts))
        choices = np.concatenate([np.arange(idx), np.arange(idx + 1, self.num_ts)])
        others = np.random.choice(choices, n_series - 1, replace=False)
        samples = self.indexer[np.concatenate([[idx], others])]
        return samples

    def _flatten_data(
        self, samples: dict[str, BatchedData]
    ) -> dict[str, FlattenedData]:
        for field in samples.keys():
            if field in self.combine_fields:
                item = samples[field]
                if isinstance(item, list) and isinstance(item[0], MultivarTimeSeries):
                    samples[field] = [
                        univar for sample in samples[field] for univar in sample
                    ]
            elif isinstance(samples[field], BatchedDateTime):
                samples[field] = np.asarray(samples[field][0])
            elif isinstance(samples[field], BatchedString):
                samples[field] = samples[field][0]
            else:
                raise AssertionError(
                    f"Field {field} not accounted for in {self.indexer} MultiSampleTimeSeriesDataset"
                )
        return samples


class EvalDataset(TimeSeriesDataset):
    """
    Dataset class for validation.
    Should be used in conjunction with Eval transformations.
    """

    def __init__(
        self,
        windows: int,
        indexer: Indexer[dict[str, Any]],
        transform: Transformation,
    ):
        """
        :param windows: number of windows to perform evaluation on
        """
        super().__init__(
            indexer,
            transform,
            SampleTimeSeriesType.NONE,
            dataset_weight=windows,
        )

    def _get_data(self, idx: int) -> dict[str, Data]:
        window, idx = divmod(idx, self.num_ts)
        item = self.indexer[idx]
        item["window"] = window
        return item


class HybridTSImageDataset(Dataset):

    def __init__(self, ts_dataset, image_sampling_prob, num_patches, num_patches_input, image_size):
        self.ts_dataset = ts_dataset
        self.image_sampling_prob = image_sampling_prob
        self.image_names = []
        image_dir = env.IMAGENET_1K_PATH
        self.visible_size = (num_patches_input * image_size // num_patches)
        # Collect all image paths in the directory and its subdirectories
        for label_name in tqdm(os.listdir(image_dir), desc="Loading image dataset from disk"):
            for file_name in os.listdir(os.path.join(image_dir, label_name)):
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    self.image_names.append(os.path.join(image_dir, label_name, file_name))
        
        self.transform_train = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )
        self.generator = np.random.default_rng(0)
    
    def __getitem__(self, idx: int):
        if self.generator.random() < self.image_sampling_prob:
            # load image
            img_name = self.image_names[self.generator.choice(len(self.image_names))]
            image = self.transform_train(Image.open(img_name).convert('RGB')) # [3, H, W]
            
            image_right = torch.mean(image, dim=0, keepdim=True)[:, :, self.visible_size:] 
            image_mask = torch.ones_like(image_right)
            image_right = torch.cat([image_right, image_mask], dim=0)

            image[:, :, self.visible_size:] = 0

            # ! 混入的真实图像数据的context_len会被设置为0，
            # ! 可以由此来区分图像or时间序列数据！
            return {
                'target': image.numpy(),
                'target_img': image_right.numpy(),
                'y': np.array([[0.0]]),
                'x': np.array([[0.0]]),
                'pad_left': 0,
                'context_len': 0,
                'pred_len': 0,
                'periodicity': 0,
                'scale_x': 0.0,
            }
        else:
            # dict_keys(['target', 'target_img', 'y', 'x', 'pad_left', 'context_len', 'pred_len', 'periodicity', 'scale_x'])
            # 
            return self.ts_dataset.__getitem__(idx)
    
    def __len__(self) -> int:
        """
        Length is the number of time series multiplied by dataset_weight
        """
        return self.ts_dataset.__len__()
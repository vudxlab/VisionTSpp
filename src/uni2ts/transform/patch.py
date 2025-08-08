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

import abc
from dataclasses import dataclass
from functools import partial
from typing import Any, Optional

import numpy as np
import pandas as pd
from einops import rearrange, repeat
from gluonts.time_feature import norm_freq_str
from jaxtyping import Num
import random
import torch
from PIL import Image
import torch.nn.functional as F

from uni2ts.common.typing import UnivarTimeSeries

from ._base import Transformation
from ._mixin import MapFuncMixin, ApplyFuncMixin

from uni2ts.model.visionts import freq_to_seasonality_list, safe_resize

class PatchSizeConstraints(abc.ABC):
    @abc.abstractmethod
    def _get_boundaries(self, n: int, offset_name: str) -> tuple[int, int]: ...

    def __call__(self, freq: str) -> range:
        offset = pd.tseries.frequencies.to_offset(freq)
        start, stop = self._get_boundaries(offset.n, norm_freq_str(offset.name))
        return range(start, stop + 1)


@dataclass
class FixedPatchSizeConstraints(PatchSizeConstraints):
    start: int
    stop: Optional[int] = None

    def __post_init__(self):
        if self.stop is None:
            self.stop = self.start
        assert self.start <= self.stop

    def _get_boundaries(self, n: int, offset_name: str) -> tuple[int, int]:
        # ! 250424 adds: 这里则就是固定stop和start一样？，并且patch大小都设置为1？
        return self.start, self.stop


class DefaultPatchSizeConstraints(PatchSizeConstraints):
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
    DEFAULT_RANGES = {
        "S": (64, 128),  # 512s = 8.53min, 4096s = 68.26min
        "T": (32, 128),  # 64min = 1.07h, 512min = 8.53h
        "H": (32, 64),  # 128h = 5.33days
        "D": (16, 32),
        "B": (16, 32),
        "W": (16, 32),
        "M": (8, 32),
        "Q": (1, 8),
        "Y": (1, 8),
        "A": (1, 8),
    }

    def _get_boundaries(self, n: int, offset_name: str) -> tuple[int, int]:
        # ! 250424 adds: 原本是对不同时间分辨率的数据，设置不同的patch size范围。
        start, stop = self.DEFAULT_RANGES[offset_name]
        return start, stop


@dataclass
class GetPatchSize(Transformation):
    min_time_patches: int
    target_field: str = "target"
    patch_sizes: tuple[int, ...] | list[int] | range = (8, 16, 32, 64, 128)
    patch_size_constraints: PatchSizeConstraints = DefaultPatchSizeConstraints()
    offset: bool = True

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        freq = data_entry["freq"]
        constraints = self.patch_size_constraints(freq)
        # largest patch size based on min_time_patches
        target: list[UnivarTimeSeries] = data_entry[self.target_field]
        length = target[0].shape[0]
        patch_size_ceil = length // self.min_time_patches

        if isinstance(self.patch_sizes, (tuple, list)):
            patch_size_candidates = [
                patch_size
                for patch_size in self.patch_sizes
                if (patch_size in constraints) and (patch_size <= patch_size_ceil)
            ]
        elif isinstance(self.patch_sizes, range):
            patch_size_candidates = range(
                max(self.patch_sizes.start, constraints.start),
                min(self.patch_sizes.stop, constraints.stop, patch_size_ceil),
            )
        else:
            raise NotImplementedError

        if len(patch_size_candidates) <= 0:
            ts_shape = (len(target),) + target[0].shape
            raise AssertionError(
                "no valid patch size candidates for "
                f"time series shape: {ts_shape}, "
                f"freq: {freq}, "
                f"patch_sizes: {self.patch_sizes}, "
                f"constraints: {constraints}, "
                f"min_time_patches: {self.min_time_patches}, "
                f"patch_size_ceil: {patch_size_ceil}"
            )

        data_entry["patch_size"] = np.random.choice(patch_size_candidates)
        return data_entry


@dataclass
class Patchify(MapFuncMixin, Transformation):
    max_patch_size: int
    fields: tuple[str, ...] = ("target",)
    optional_fields: tuple[str, ...] = ("past_feat_dynamic_real",)
    pad_value: int | float = 0

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        patch_size = data_entry["patch_size"]
        self.map_func(
            partial(self._patchify, patch_size=patch_size),  # noqa
            data_entry,
            self.fields,
            optional_fields=self.optional_fields,
        )
        return data_entry

    def _patchify(self, data_entry: dict[str, Any], field: str, patch_size: int):
        arr = data_entry[field]
        if isinstance(arr, list):
            return [self._patchify_arr(a, patch_size) for a in arr]
        if isinstance(arr, dict):
            for k, v in arr.items():
                if k in self.fields or k in self.optional_fields:
                    arr[k] = self._patchify_arr(v, patch_size)
            return arr
        return self._patchify_arr(arr, patch_size)

    def _patchify_arr(
        self, arr: Num[np.ndarray, "var time*patch"], patch_size: int
    ) -> Num[np.ndarray, "var time max_patch"]:
        assert arr.shape[-1] % patch_size == 0
        arr = rearrange(arr, "... (time patch) -> ... time patch", patch=patch_size)
        pad_width = [(0, 0) for _ in range(arr.ndim)]
        pad_width[-1] = (0, self.max_patch_size - patch_size)
        arr = np.pad(arr, pad_width, mode="constant", constant_values=self.pad_value)
        return arr

def FFT_for_Period(x, k=5):
    # [T, ]
    xf = np.fft.rfft(x)
    # find period by amplitudes
    frequency_list = abs(xf)
    frequency_list[0] = 0
    top_list = np.argsort(frequency_list)[-k:]
    period = len(x) // top_list
    period_prob = abs(xf).mean(-1)[:, top_list]
    period_prob = period_prob / np.sum(period_prob)
    return period, period_prob


@dataclass
class ImagifyTS(ApplyFuncMixin, Transformation):
    image_size: int
    patch_size: int
    num_patch_input: int
    norm_const: float
    min_pred_ratio: float
    max_pred_ratio: float
    max_pre_mask_ratio: float
    pre_mask_prob: float
    fields: tuple[str, ...] = ("target",)

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        freq = data_entry["freq"]
        self.apply_func(
            partial(self._imagify, freq=freq),  # noqa
            data_entry,
            self.fields,
            optional_fields=[],
        )
        return data_entry

    def _imagify(self, data_entry: dict[str, Any], field: str, freq: str):
        # periods = freq_to_seasonality_list(freq)
        # data = data_entry[field][0]
        # fft = np.abs(np.fft.rfft(data))
        # eps = 1e-6
        # fft = np.clip(fft / np.sum(fft), eps, 1 - eps)
        # no_period_prob = np.sum(-fft * np.log(fft) / np.log(len(data)))
        # energy = [(p, fft[len(data) // p]) for p in periods]
        # print(energy, no_period_prob)
        
        # ! data_entry[field].shape = [nvars, total_len]

        # setup
        periodicity_list = [x for x in freq_to_seasonality_list(freq) if x * 2 < len(data_entry[field][0])]
        periodicity = random.choice(periodicity_list)  # 随机从所有freq里选一个？
        
        data = data_entry[field][0]
        pred_len = int(len(data) * (random.random() * (self.max_pred_ratio - self.min_pred_ratio) + self.min_pred_ratio))  # max=0.5, min=0.15
        context_len = len(data) - pred_len

        if context_len > periodicity * self.num_patch_input * self.patch_size:
            data = data[:periodicity * self.num_patch_input * self.patch_size]  # num_patch_input=7, patch_size=16, 如果原来样本的context超过112个周期的长度，则只截取出最后112个周期的数据了！！
            pred_len = int(len(data) * (random.random() * (self.max_pred_ratio - self.min_pred_ratio) + self.min_pred_ratio))  # 
            context_len = len(data) - pred_len  # 重新计算context和pred len。

        x = torch.Tensor(data[:context_len]).reshape((1, -1, 1))
        y = torch.Tensor(data[context_len:]).reshape((1, -1, 1))
        num_patch = self.image_size // self.patch_size  # 224 // 16 = 14
        num_patch_output = num_patch - self.num_patch_input  # 14 -7 = 7
        adjust_input_ratio = self.num_patch_input / num_patch  # 7 / 14 = 0.5
        input_resize = safe_resize((self.image_size, int(self.image_size * adjust_input_ratio)), interpolation=Image.BILINEAR)
        pad_left = 0
        pad_right = 0
        if context_len % periodicity != 0:
            pad_left = periodicity - context_len % periodicity  # 如果不是周期整数倍，需要对左右边做padding！
        if pred_len % periodicity != 0:
            pad_right = periodicity - pred_len % periodicity

        # 因为context_len的周期数要对应到112个像素点里，该值表示横向上“一个像素点里需要放几个周期的时序数据”？例如假设context_len一共8个周期，则scale_x = 8/112 = 1/14 = 0.0714
        scale_x = ((pad_left + context_len) // periodicity) / (int(self.image_size * adjust_input_ratio))
        # output_resize = safe_resize((periodicity, int(round(image_size * scale_x))), interpolation=interpolation)

        # 1. Normalization
        means = x.mean(1, keepdim=True).detach() # x: [bs x 1 x nvars], means: [1 x 1 x 1]
        x_enc = x - means
        stdev = torch.sqrt(
            torch.var(x_enc.to(torch.float64), dim=1, keepdim=True, unbiased=False) + 1e-5) # [bs x 1 x nvars]
        # ! norm_const: 一般设置为0.4，用于约束增大标准差stdev，使得norm之后的值的范围更小！
        stdev /= self.norm_const
        x_enc /= stdev
        # Channel Independent
        x_enc = rearrange(x_enc, 'b s n -> b n s') # x_enc: [bs x nvars x seq_len]

        # TODO: 这句code的目的是？
        lookback_masked = False
        if random.random() < self.pre_mask_prob:  # self.pre_mask_prob = 0.05
            # self.max_pre_mask_ratio = 0.5 -> 有0.05的概率输入一个左半边全为0的图片？
            # ! 250410 adds: 这里额外增加一个random.random()，减少mask过长的input？
            mask_len = int(len(data) * self.max_pre_mask_ratio) * random.random()
            x_enc[:, :, :int(len(data) * self.max_pre_mask_ratio)] = 0
            lookback_masked = True

        # 2. Segmentation
        x_pad = F.pad(x_enc, (pad_left, 0), mode='constant') # [batch, nvars, seq_len]
        x_2d = rearrange(x_pad, 'b n (p f) -> (b n) 1 f p', f=periodicity)  # f指的就是周期！！

        # 3. Render & Alignment
        x_resize = input_resize(x_2d)  # 把[周期长度，周期数]的数据resize到[224, 112]，正好为图片的左半边。（ps：右半边为mask！！）
        masked = torch.zeros((x_2d.shape[0], 1, self.image_size, num_patch_output * self.patch_size), device=x_2d.device, dtype=x_2d.dtype)  # [bs, 1, 224, 112]，右半边的mask
        x_concat_with_masked = torch.cat([
            x_resize, 
            masked
        ], dim=-1)  # concat起来！
        image_input = repeat(x_concat_with_masked, 'b 1 h w -> b c h w', c=3)  # copy 3个channels

        # output
        # 这里对x和y也做了norm！！
        y = (y - means) / stdev # [1, Pred_len, 1]
        x = (x - means) / stdev # [1, Seq_len, 1]
        
        # Pixel loss
        y_true = y[:, :, 0] # [1, Pred_len]
        y_mask = torch.ones_like(y_true) # [1, Pred_len]
        if pad_right != 0:
            y_true = F.pad(y_true, (0, pad_right), mode='replicate') # [1, Pred_len]
            y_mask = F.pad(y_mask, (0, pad_right), mode='constant')
        y_true_with_mask = torch.cat([y_true, y_mask], dim=0) # [2, Pred_len]
        y_true_2d = rearrange(y_true_with_mask, 'b (p f) -> b 1 f p', f=periodicity)  # y_true_2d: [b, 1, 周期长度, 周期数] 第一维为真实的y值，第二维为y_mask，1表示该位置是有值的（对应真实标签值），0表示无值（即就是空着的）
        target_width = int(round(y_true_2d.shape[3] / float(scale_x)))  # 预测窗口的pred_len对应的像素点个数，如19 / 0.83 = 23
        padding_width = num_patch_output * self.patch_size - target_width  # 因为mask部分横向一共有112个像素点，剩余的112-23=89个像素点都当作padding处理了！
        y_true_2d = safe_resize((self.image_size, target_width), interpolation=Image.BILINEAR)(y_true_2d) # 插值到：[2, 1, Image_Size=224, Target_Width=112]
        y_true_2d = F.pad(y_true_2d, (0, padding_width), mode='constant') # y_true_2d右边没有值的部分用0填充，shape为[2, 1, Image_Size=224, Mask_Size=112]
        y_true_2d = rearrange(y_true_2d, 'b 1 h w -> b h w') # [2, Image_Size=224, Mask_Size=112]

        data_entry[field] = image_input.float().numpy()[0]
        data_entry[field + "_img"] = y_true_2d.float().numpy()
        data_entry['pad_left'] = int(pad_left)
        data_entry['context_len'] = int(context_len)
        data_entry['pred_len'] = int(pred_len)
        data_entry['periodicity'] = int(periodicity)
        data_entry['y'] = y.numpy()[0]
        data_entry['x'] = x.numpy()[0]
        data_entry['scale_x'] = float(scale_x)
        # data_entry['mask'] = float(scale_x)
        
        # ! 250410 adds: 
        # assert means.shape == (1, 1, 1)
        # assert stdev.shape == (1, 1, 1)
        data_entry['means'] = float(means.cpu().numpy()[0][0])
        data_entry['stdev'] = float(stdev.cpu().numpy()[0][0])
        data_entry['norm_const'] = float(self.norm_const)
        data_entry['lookback_masked'] = lookback_masked
        # image_size, patch_size, num_patch_input, 
        data_entry['image_size'] = int(self.image_size)
        data_entry['patch_size'] = int(self.patch_size)
        data_entry['num_patch_input'] = int(self.num_patch_input)
        
        data_entry['nvars'] = 1
        
        return
        
    
        # fft
        arr = data_entry[field]
        if isinstance(arr, list):
            return [self._patchify_arr(a, patch_size) for a in arr]
        if isinstance(arr, dict):
            for k, v in arr.items():
                if k in self.fields or k in self.optional_fields:
                    arr[k] = self._patchify_arr(v, patch_size)
            return arr
        return self._patchify_arr(arr, patch_size)




@dataclass
class ImagifyTS_Multivariate(ApplyFuncMixin, Transformation):
    image_size: int
    patch_size: int
    num_patch_input: int
    norm_const: float
    min_pred_ratio: float
    max_pred_ratio: float
    max_pre_mask_ratio: float
    pre_mask_prob: float
    fields: tuple[str, ...] = ("target",)

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        freq = data_entry["freq"]
        self.apply_func(
            partial(self._imagify, freq=freq),  # noqa
            data_entry,
            self.fields,
            optional_fields=[],
        )
        return data_entry

    def _imagify(self, data_entry: dict[str, Any], field: str, freq: str):
        # setup
        periodicity_list = [x for x in freq_to_seasonality_list(freq) if x * 2 < len(data_entry[field][0])]  # 这样保证数据里至少有两个完整周期？
        periodicity = random.choice(periodicity_list)  # 随机从所有freq里选一个？
        
        data = data_entry[field]  # shape: [nvars, total_len]
        nvars, total_len = data.shape
        pred_len = int(len(data[0]) * (random.random() * (self.max_pred_ratio - self.min_pred_ratio) + self.min_pred_ratio))  # max=0.5, min=0.15
        context_len = len(data[0]) - pred_len

        if context_len > periodicity * self.num_patch_input * self.patch_size:  # period * 7 * 16
            data = data[:, :periodicity * self.num_patch_input * self.patch_size]  # num_patch_input=7, patch_size=16, 如果原来样本的context超过112个周期的长度，则只截取出最后112个周期的数据了！！
            pred_len = int(len(data[0]) * (random.random() * (self.max_pred_ratio - self.min_pred_ratio) + self.min_pred_ratio))  # 
            context_len = len(data[0]) - pred_len  # 重新计算context和pred len。

        # x = torch.Tensor(data[:, :context_len]).reshape((1, -1, nvars))  # [bs, seq_len, nvars]
        # y = torch.Tensor(data[:, context_len:]).reshape((1, -1, nvars))  # [bs, pred_len, nvars]
        x = torch.Tensor(rearrange(data[:, :context_len], 'n s -> 1 s n'))  # x: [1, seq_len, nvars]
        y = torch.Tensor(rearrange(data[:, context_len:], 'n s -> 1 s n'))  # x: [1, seq_len, nvars]
        
        
        num_patch = self.image_size // self.patch_size  # 224 // 16 = 14
        num_patch_output = num_patch - self.num_patch_input  # 14 - 7 = 7
        adjust_input_ratio = self.num_patch_input / num_patch  # 7 / 14 = 0.5
        
        # safe_resize返回的是：Resize(size = [224, 112], interpolation = "Bilinear", antialias=False)
        # ! 250426 adds: 这里的image_size_per_var是224/nvars，表示每个变量对应224/nvars个像素点。
        # ! 考虑无法整除的情况，可能会在后面做pad_down！
        image_size_per_var = int(self.image_size / nvars)  # 224 // nvars
        input_resize = safe_resize((image_size_per_var, int(self.image_size * adjust_input_ratio)), interpolation=Image.BILINEAR)
        
        pad_left = 0
        pad_right = 0
        
        if context_len % periodicity != 0:
            pad_left = periodicity - context_len % periodicity  # 如果不是周期整数倍，需要对左右边做padding！
        if pred_len % periodicity != 0:
            pad_right = periodicity - pred_len % periodicity

        # 因为context_len的周期数要对应到112个像素点里，因此scale_x表示横向上“一个像素点里需要放几个周期的时序数据”？
        # 例如假设context_len一共8个周期，则scale_x = 8/112 = 1/14 = 0.0714
        scale_x = ((pad_left + context_len) // periodicity) / (int(self.image_size * adjust_input_ratio))
        
        # output_resize = safe_resize((periodicity, int(round(image_size * scale_x))), interpolation=interpolation)

        # 1. Normalization
        means = x.mean(1, keepdim=True).detach()  # x: [bs x seq_len x nvars], means: [bs x 1 x nvars]
        x_enc = x - means
        stdev = torch.sqrt(
            torch.var(x_enc.to(torch.float64), dim=1, keepdim=True, unbiased=False) + 1e-5)  # [bs x 1 x nvars]
        # ! norm_const: 一般设置为0.4，用于约束增大标准差stdev，使得norm之后的值的范围更小！
        stdev /= self.norm_const
        x_enc /= stdev
        # Channel Independent
        x_enc = rearrange(x_enc, 'b s n -> b n s')  # x_enc: [bs x nvars x seq_len]

        # TODO: 这句code的目的是？
        # ! 250425 adds: 这里是为了以0.05的概率令seq_len的左侧有一部分为0，
        # ! 从而让pred_len比seq_len长的情况也能被训练到！
        lookback_masked = False
        if random.random() < self.pre_mask_prob:  # self.pre_mask_prob = 0.05
            # self.max_pre_mask_ratio = 0.5 -> 有0.05的概率输入一个左半边全为0的图片？
            # ! 250410 adds: 这里额外增加一个random.random()，减少mask过长的input？
            mask_len = int(len(data[0]) * self.max_pre_mask_ratio * random.random())
            # x_enc[:, :, :int(len(data[0]) * self.max_pre_mask_ratio)] = 0
            x_enc[:, :, :mask_len] = 0
            lookback_masked = True

        # 2. Segmentation
        # PS: 当pad有两个参数，代表对最后一个维度扩充，pad = (左边填充数=pad_left，右边填充数=0)
        x_pad = F.pad(x_enc, (pad_left, 0), mode='constant') # [bs, nvars, seq_len]
        
        # ! f指的就是周期！然后p是周期个数？
        # ! -> 注意这里前面是(p f)，后面是f p，是因为前面是横向从前到后顺序的数据，而转换好的图像为纵向从上到下顺序，所以要调换一下！！
        # x_2d = rearrange(x_pad, 'b n (p f) -> (b n) 1 f p', f=periodicity)
        # ! 250424 adds: x_2d.shape = [bs, 1, nvars * periodicity, 周期数p]
        # x_2d = rearrange(x_pad, 'b n (p f) -> b 1 (n f) p', f=periodicity)
        x_2d = rearrange(x_pad, 'b n (p f) -> b n f p', f=periodicity)  # x_2d.shape: [bs, nvars, 周期长度f, 周期数p]

        # 3. Render & Alignment
        # ! 这里把[周期长度f，周期数p]的数据resize到[224, 112]，正好为图片的左半边。（ps：右半边为mask！！）
        x_resize = input_resize(x_2d)  # x_resize.shape: [b, nvars, 224//nvars, p->112]
        x_resize = rearrange(x_resize, 'b n h w -> b 1 (n h) w')  # x_resize.shape: [bs, 1, nvars*224//nvars, 112]
        # 由于nvars不一定完全被224整除，所以可能存在pad_down
        pad_down = self.image_size - x_resize.shape[2]  # 224 - nvars*224//nvars
        
        if pad_down > 0:
            # x_resize = F.pad(x_resize, (0, 0, 0, pad_down), mode='constant')
            x_resize = torch.concat([
                    x_resize, 
                    torch.zeros((x_resize.shape[0], x_resize.shape[1], pad_down, x_resize.shape[3]), 
                        device=x_resize.device, dtype=x_resize.dtype)
                ], 
                dim=2)  # [bs, 1, 224, 112]
        assert x_resize.shape[2] == self.image_size, f"image size mismatch: {x_resize.shape[2]} vs {self.image_size}"
        
        
        # masked: [bs, 1, 224, 112]，右半边的mask
        masked = torch.zeros((x_2d.shape[0], 1, self.image_size, num_patch_output * self.patch_size), device=x_2d.device, dtype=x_2d.dtype)
        x_concat_with_masked = torch.cat([
            x_resize, 
            masked
        ], dim=-1)  # concat起来！最后一维：112+112=224，也即x_concat_with_masked.shape = [bs, 1, 224, 224]
        
        # TODO: 250426 adds: 这里可以根据颜色区分nvars！！
        
        # # ! solution 1: 不加颜色，直接repeat三份
        # image_input = repeat(x_concat_with_masked, 'b 1 h w -> b c h w', c=3)  # copy 3个channels
        
        # ! solution 2: 随机对每个var使用不同颜色！
        # color_dict = {0: [0,0,0], 1: [0,0,1]}
        image_input = torch.zeros((x_concat_with_masked.shape[0], 3, x_concat_with_masked.shape[2], x_concat_with_masked.shape[3]), 
                                  device=x_concat_with_masked.device, 
                                  dtype=x_concat_with_masked.dtype)  # [bs, 3, 224, 224]
        color_list = []
        for i in range(nvars):
            if i == 0: 
                color = random.randint(0, 2)  # random.randint是包含左右闭区间的！！
            else:
                tmp_color = random.randint(0, 2)
                while tmp_color == color:
                    tmp_color = random.randint(0, 2)
                color = tmp_color
            color_list.append(color)
            # 这里把对应的通道的颜色给换上去？
            image_input[:, color, i*image_size_per_var:(i+1)*image_size_per_var, :] = \
                x_concat_with_masked[:, 0, i*image_size_per_var:(i+1)*image_size_per_var, :]
        

        # output
        # 这里对x和y也做了norm！！
        y = (y - means) / stdev # [bs, pred_len, nvars]
        x = (x - means) / stdev # [bs, seq_len, nvars]
        
        
        # ! 250424 adds: 这里不需要算pixel loss！
        # # Pixel loss
        # y_true = y[:, :, :] # [bs, Pred_len, nvars]
        # y_mask = torch.ones_like(y_true) # [1, Pred_len, nvars]
        
        # y_true = rearrange(y_true, 'b pl n -> b n pl')  # [bs x nvars x seq_len]
        # y_mask = rearrange(y_mask, 'b pl n -> b n pl')  # [bs x nvars x seq_len]
        
        # if pad_right != 0:
        #     # 对最后的pl维度做padding，左边不补，右边补pad_right长度
        #     y_true = F.pad(y_true, (0, pad_right), mode='replicate') # [1, nvars, Pred_len]
        #     y_mask = F.pad(y_mask, (0, pad_right), mode='constant')
        
        # y_true_with_mask = torch.cat([y_true, y_mask], dim=0) # [2, nvars, Pred_len]
        # # ! y_true_2d: [2, 1, 周期长度, 周期数] 第一维为2表示：第一个为真实的y值，第二个为y_mask，1表示该位置是有值的（对应真实标签值），0表示无值（即就是空着的mask）
        # # y_true_2d = rearrange(y_true_with_mask, 'b (p f) -> b 1 f p', f=periodicity)
        # # ! 250424 adds: y_true_2d.shape = [2, 1, nvars * periodicity, 周期数p]
        # y_true_2d = rearrange(y_true_with_mask, 'b n (p f) -> b 1 (n f) p', f=periodicity)
        
        # # 预测窗口的pred_len对应的周期数p对应的像素点个数，如19 / 0.83 = 23
        # target_width = int(round(y_true_2d.shape[3] / float(scale_x)))
        # # 因为mask部分横向一共有112个像素点，剩余的112-23=89个像素点都当作padding处理了！
        # padding_width = num_patch_output * self.patch_size - target_width

        # # 插值到：[2, 1, Image_Size=224, Target_Width=23]
        # y_true_2d = safe_resize((self.image_size, target_width), interpolation=Image.BILINEAR)(y_true_2d)
        # # y_true_2d右边没有值的部分（还有112-23=89个像素）用0填充，shape为[2, 1, Image_Size=224, Total_Size=112]
        # y_true_2d = F.pad(y_true_2d, (0, padding_width), mode='constant')
        # # y_true_2d.shape: [2, Image_Size=224, Mask_Size=112]
        # y_true_2d = rearrange(y_true_2d, 'b 1 h w -> b h w')

        
        # ! 250425 asdadds: 这里[0]应该还是要保留的，因为0表示batch维度，但是这里应该不需要！
        data_entry[field] = image_input.float().numpy()[0]  # [3, 224, 224]
        # data_entry[field + "_img"] = y_true_2d.float().numpy()  # [1, 224, 224]
        # data_entry[field + "_img"] = None  # [1, 224, 224]
        
        data_entry['pad_left'] = int(pad_left)
        data_entry['context_len'] = int(context_len)
        data_entry['pred_len'] = int(pred_len)
        data_entry['periodicity'] = int(periodicity)
        
        # ! 250425 adds: 这里[0]应该还是要保留的，因为0表示batch维度，但是这里应该不需要！
        data_entry['y'] = y.numpy()[0]  # [seq_len, nvars]
        data_entry['x'] = x.numpy()[0]  # [pred_len, nvars]
        data_entry['scale_x'] = float(scale_x)
        # data_entry['mask'] = float(scale_x)
        
        # ! 250410 adds: 
        # assert means.shape == (1, 1, 1)
        # assert stdev.shape == (1, 1, 1)
        # data_entry['means'] = float(means.cpu().numpy()[0][0])
        # data_entry['stdev'] = float(stdev.cpu().numpy()[0][0])
        
        data_entry['means'] = means.cpu().numpy()[0][0]  # [nvars, ]
        data_entry['stdev'] = stdev.cpu().numpy()[0][0]  # [nvars, ]
        data_entry['norm_const'] = float(self.norm_const)
        data_entry['lookback_masked'] = lookback_masked  # True/False
        # image_size, patch_size, num_patch_input, 
        data_entry['image_size'] = int(self.image_size)
        data_entry['patch_size'] = int(self.patch_size)
        data_entry['num_patch_input'] = int(self.num_patch_input)
        data_entry['nvars'] = int(nvars)
        data_entry['pad_down'] = int(pad_down)
        data_entry['color_list'] = np.array(color_list) # [nvars, ]
        
        return


@dataclass
class Imagify(ApplyFuncMixin, Transformation):
    image_size: int
    patch_size: int
    num_patch_input: int
    norm_const: float
    min_pred_ratio: float
    max_pred_ratio: float
    max_pre_mask_ratio: float
    pre_mask_prob: float
    fields: tuple[str, ...] = ("target",)

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        freq = data_entry["freq"]
        self.apply_func(
            partial(self._imagify, freq=freq),  # noqa
            data_entry,
            self.fields,
            optional_fields=[],
        )
        return data_entry

    def _imagify(self, data_entry: dict[str, Any], field: str, freq: str):
        # periods = freq_to_seasonality_list(freq)
        # data = data_entry[field][0]
        # fft = np.abs(np.fft.rfft(data))
        # eps = 1e-6
        # fft = np.clip(fft / np.sum(fft), eps, 1 - eps)
        # no_period_prob = np.sum(-fft * np.log(fft) / np.log(len(data)))
        # energy = [(p, fft[len(data) // p]) for p in periods]
        # print(energy, no_period_prob)

        # setup 
        periodicity_list = [x for x in freq_to_seasonality_list(freq) if x * 2 < len(data_entry[field][0])]
        periodicity = random.choice(periodicity_list)
        data = data_entry[field][0]
        pred_len = int(len(data) * (random.random() * (self.max_pred_ratio - self.min_pred_ratio) + self.min_pred_ratio))
        context_len = len(data) - pred_len

        if context_len > periodicity * self.num_patch_input * self.patch_size:
            data = data[:periodicity * self.num_patch_input * self.patch_size]
            pred_len = int(len(data) * (random.random() * (self.max_pred_ratio - self.min_pred_ratio) + self.min_pred_ratio))
            context_len = len(data) - pred_len

        x = torch.Tensor(data[:context_len]).reshape((1, -1, 1))
        y = torch.Tensor(data[context_len:]).reshape((1, -1, 1))
        num_patch = self.image_size // self.patch_size
        num_patch_output = num_patch - self.num_patch_input
        adjust_input_ratio = self.num_patch_input / num_patch
        input_resize = safe_resize((self.image_size, int(self.image_size * adjust_input_ratio)), interpolation=Image.BILINEAR)
        pad_left = 0
        pad_right = 0
        if context_len % periodicity != 0:
            pad_left = periodicity - context_len % periodicity
        if pred_len % periodicity != 0:
            pad_right = periodicity - pred_len % periodicity

        scale_x = ((pad_left + context_len) // periodicity) / (int(self.image_size * adjust_input_ratio))
        # output_resize = safe_resize((periodicity, int(round(image_size * scale_x))), interpolation=interpolation)

        # 1. Normalization
        means = x.mean(1, keepdim=True).detach() # [bs x 1 x nvars]
        x_enc = x - means
        stdev = torch.sqrt(
            torch.var(x_enc.to(torch.float64), dim=1, keepdim=True, unbiased=False) + 1e-5) # [bs x 1 x nvars]
        stdev /= self.norm_const
        x_enc /= stdev
        # Channel Independent
        x_enc = rearrange(x_enc, 'b s n -> b n s') # [bs x nvars x seq_len]

        
        if random.random() < self.pre_mask_prob:
            x_enc[:, :, :int(len(data) * self.max_pre_mask_ratio)] = 0

        # 2. Segmentation
        x_pad = F.pad(x_enc, (pad_left, 0), mode='replicate') # [b n s]
        x_2d = rearrange(x_pad, 'b n (p f) -> (b n) 1 f p', f=periodicity)

        # 3. Render & Alignment
        x_resize = input_resize(x_2d)
        masked = torch.zeros((x_2d.shape[0], 1, self.image_size, num_patch_output * self.patch_size), device=x_2d.device, dtype=x_2d.dtype)
        x_concat_with_masked = torch.cat([
            x_resize, 
            masked
        ], dim=-1)
        image_input = repeat(x_concat_with_masked, 'b 1 h w -> b c h w', c=3)

        # output
        y = (y - means) / stdev
        x = (x - means) / stdev

        data_entry[field] = image_input.numpy()[0]
        data_entry['pad_left'] = int(pad_left)
        data_entry['context_len'] = int(context_len)
        data_entry['pred_len'] = int(pred_len)
        data_entry['periodicity'] = int(periodicity)
        data_entry['y'] = y.numpy()[0]
        data_entry['x'] = x.numpy()[0]
        data_entry['scale_x'] = float(scale_x)
        data_entry['mask'] = float(scale_x)
        return
        # fft
        arr = data_entry[field]
        if isinstance(arr, list):
            return [self._patchify_arr(a, patch_size) for a in arr]
        if isinstance(arr, dict):
            for k, v in arr.items():
                if k in self.fields or k in self.optional_fields:
                    arr[k] = self._patchify_arr(v, patch_size)
            return arr
        return self._patchify_arr(arr, patch_size)
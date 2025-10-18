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
        # ! 250424 bổ sung: cấu hình này cố định stop và start giống nhau, tức là kích thước patch duy nhất
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
        # ! 250424 bổ sung: đặt khoảng kích thước patch khác nhau tùy theo độ phân giải thời gian
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
        self, arr: np.ndarray, patch_size: int
    ) -> np.ndarray:
        assert arr.shape[-1] % patch_size == 0
        arr = rearrange(arr, "... (time patch) -> ... time patch", patch=patch_size)
        pad_width = [(0, 0) for _ in range(arr.ndim)]
        pad_width[-1] = (0, self.max_patch_size - patch_size)
        arr = np.pad(arr, pad_width, mode="constant", constant_values=self.pad_value)
        return arr

def FFT_for_Period(x, k=5):
    # Dạng [T, ]
    xf = np.fft.rfft(x)
    # tìm chu kỳ dựa trên biên độ phổ
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
    task: str = "forecast"
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
        
        # ! data_entry[field] có dạng [nvars, total_len]

        # Khởi tạo
        periodicity_list = [x for x in freq_to_seasonality_list(freq) if x * 2 < len(data_entry[field][0])]
        periodicity = random.choice(periodicity_list)  # Chọn ngẫu nhiên một chu kỳ từ danh sách tần suất

        full_series = data_entry[field][0]
        total_len = len(full_series)
        if total_len <= 1:
            raise ValueError("Time series is too short to create an imputation task.")

        ratio = random.random() * (self.max_pred_ratio - self.min_pred_ratio) + self.min_pred_ratio
        pred_len = max(1, int(total_len * ratio))
        pred_len = min(pred_len, total_len - 1)

        max_kernel = periodicity * self.num_patch_input * self.patch_size

        if getattr(self, "task", "forecast") == "impute":
            max_start = total_len - pred_len
            min_start = min(max_start - 1, max(periodicity, self.patch_size))
            if min_start < 1:
                min_start = 1
            if max_start <= min_start:
                missing_start = max_start
            else:
                missing_start = random.randint(min_start, max_start)

            context_end = missing_start
            context_start = max(0, context_end - max_kernel)
            context_len = context_end - context_start
            if context_len <= 0:
                raise ValueError("Failed to sample a valid context window for imputation.")

            data = full_series[context_start : missing_start + pred_len]
            data_entry["impute_context_start"] = context_start
            data_entry["impute_missing_start"] = missing_start
            data_entry["impute_missing_length"] = pred_len
        else:
            data = full_series
            context_len = total_len - pred_len
            if context_len > max_kernel:
                context_start = context_len - max_kernel
                context_len = max_kernel
                data = full_series[context_start : context_start + context_len + pred_len]
            else:
                data = full_series[: context_len + pred_len]

        x = torch.Tensor(data[:context_len]).reshape((1, -1, 1))
        y = torch.Tensor(data[context_len:]).reshape((1, -1, 1))
        num_patch = self.image_size // self.patch_size  # 224 // 16 = 14
        num_patch_output = num_patch - self.num_patch_input  # 14 -7 = 7
        adjust_input_ratio = self.num_patch_input / num_patch  # 7 / 14 = 0.5
        input_resize = safe_resize((self.image_size, int(self.image_size * adjust_input_ratio)), interpolation=Image.BILINEAR)
        pad_left = 0
        pad_right = 0
        if context_len % periodicity != 0:
            pad_left = periodicity - context_len % periodicity  # Nếu không chia hết theo chu kỳ thì cần đệm thêm ở hai đầu
        if pred_len % periodicity != 0:
            pad_right = periodicity - pred_len % periodicity

        # Vì số chu kỳ trong context_len phải được ánh xạ vào 112 điểm ảnh, giá trị này biểu diễn số chu kỳ nằm trong một điểm ảnh theo chiều ngang.
        # Ví dụ context_len có 8 chu kỳ thì scale_x = 8/112 = 1/14 = 0,0714.
        scale_x = ((pad_left + context_len) // periodicity) / (int(self.image_size * adjust_input_ratio))
        # output_resize = safe_resize((periodicity, int(round(image_size * scale_x))), interpolation=interpolation)

        # 1. Chuẩn hóa
        means = x.mean(1, keepdim=True).detach()  # x: [bs x 1 x nvars], means: [1 x 1 x 1]
        x_enc = x - means
        stdev = torch.sqrt(
            torch.var(x_enc.to(torch.float64), dim=1, keepdim=True, unbiased=False) + 1e-5
        )  # [bs x 1 x nvars]
        # ! norm_const: thường đặt 0.4 để hạn chế việc phóng đại độ lệch chuẩn, giúp giá trị sau chuẩn hóa nằm trong phạm vi hẹp hơn
        stdev /= self.norm_const
        x_enc /= stdev
        # Kênh độc lập
        x_enc = rearrange(x_enc, 'b s n -> b n s')  # x_enc: [bs x nvars x seq_len]

        # TODO: mục đích cụ thể của đoạn mã này là gì?
        lookback_masked = False
        if random.random() < self.pre_mask_prob:  # self.pre_mask_prob = 0.05
            # self.max_pre_mask_ratio = 0.5 -> có 5% xác suất phần phía trái của ảnh bị che toàn bộ bằng 0
            # ! 250410 bổ sung: nhân thêm random.random() để giảm nguy cơ mặt nạ quá dài
            mask_len = int(len(data) * self.max_pre_mask_ratio) * random.random()
            x_enc[:, :, :int(len(data) * self.max_pre_mask_ratio)] = 0
            lookback_masked = True

        # 2. Phân đoạn
        x_pad = F.pad(x_enc, (pad_left, 0), mode='constant')  # [batch, nvars, seq_len]
        x_2d = rearrange(x_pad, 'b n (p f) -> (b n) 1 f p', f=periodicity)  # f chính là chu kỳ

        # 3. Dựng ảnh & Căn chỉnh
        x_resize = input_resize(x_2d)  # Co dãn dữ liệu [độ dài chu kỳ, số chu kỳ] về [224, 112] cho nửa trái của ảnh
        masked = torch.zeros(
            (x_2d.shape[0], 1, self.image_size, num_patch_output * self.patch_size),
            device=x_2d.device,
            dtype=x_2d.dtype,
        )  # [bs, 1, 224, 112], nửa phải là vùng mặt nạ
        x_concat_with_masked = torch.cat([
            x_resize, 
            masked
        ], dim=-1)  # Ghép hai nửa ảnh lại với nhau
        image_input = repeat(x_concat_with_masked, 'b 1 h w -> b c h w', c=3)  # Sao chép thành 3 kênh màu

        # Đầu ra
        # x và y cũng được chuẩn hóa
        y = (y - means) / stdev  # [1, Pred_len, 1]
        x = (x - means) / stdev # [1, Seq_len, 1]
        
        # Pixel loss
        y_true = y[:, :, 0] # [1, Pred_len]
        y_mask = torch.ones_like(y_true) # [1, Pred_len]
        if pad_right != 0:
            y_true = F.pad(y_true, (0, pad_right), mode='replicate') # [1, Pred_len]
            y_mask = F.pad(y_mask, (0, pad_right), mode='constant')
        y_true_with_mask = torch.cat([y_true, y_mask], dim=0) # [2, Pred_len]
        y_true_2d = rearrange(y_true_with_mask, 'b (p f) -> b 1 f p', f=periodicity)  # y_true_2d: [b, 1, độ dài chu kỳ, số chu kỳ], hàng đầu là giá trị thật, hàng thứ hai là mặt nạ (1: có nhãn, 0: trống)
        target_width = int(round(y_true_2d.shape[3] / float(scale_x)))  # Số điểm ảnh tương ứng với pred_len trong cửa sổ dự báo, ví dụ 19 / 0.83 ≈ 23
        padding_width = num_patch_output * self.patch_size - target_width  # Phần mặt nạ có 112 điểm ảnh ngang, còn lại 112-23=89 điểm ảnh sẽ được đệm
        y_true_2d = safe_resize((self.image_size, target_width), interpolation=Image.BILINEAR)(y_true_2d)  # Nội suy thành [2, 1, 224, target_width]
        y_true_2d = F.pad(y_true_2d, (0, padding_width), mode='constant')  # Điền 0 cho phần bên phải không có giá trị, thu được [2, 1, 224, 112]
        y_true_2d = rearrange(y_true_2d, 'b 1 h w -> b h w')  # [2, 224, 112]

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
        
        # ! 250410 bổ sung:
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
        # Thiết lập ban đầu
        periodicity_list = [x for x in freq_to_seasonality_list(freq) if x * 2 < len(data_entry[field][0])]  # Đảm bảo dữ liệu có ít nhất hai chu kỳ đầy đủ
        periodicity = random.choice(periodicity_list)  # Chọn ngẫu nhiên một chu kỳ từ danh sách tần suất
        
        data = data_entry[field]  # shape: [nvars, total_len]
        nvars, total_len = data.shape
        pred_len = int(len(data[0]) * (random.random() * (self.max_pred_ratio - self.min_pred_ratio) + self.min_pred_ratio))  # max=0.5, min=0.15
        context_len = len(data[0]) - pred_len

        if context_len > periodicity * self.num_patch_input * self.patch_size:  # period * 7 * 16
            data = data[:, :periodicity * self.num_patch_input * self.patch_size]  # Nếu context vượt quá 112 đoạn chu kỳ thì chỉ giữ lại phần cuối cùng
            pred_len = int(len(data[0]) * (random.random() * (self.max_pred_ratio - self.min_pred_ratio) + self.min_pred_ratio))
            context_len = len(data[0]) - pred_len  # Tính lại độ dài context và pred_len

        # x = torch.Tensor(data[:, :context_len]).reshape((1, -1, nvars))  # [bs, seq_len, nvars]
        # y = torch.Tensor(data[:, context_len:]).reshape((1, -1, nvars))  # [bs, pred_len, nvars]
        x = torch.Tensor(rearrange(data[:, :context_len], 'n s -> 1 s n'))  # x: [1, seq_len, nvars]
        y = torch.Tensor(rearrange(data[:, context_len:], 'n s -> 1 s n'))  # x: [1, seq_len, nvars]
        
        
        num_patch = self.image_size // self.patch_size  # 224 // 16 = 14
        num_patch_output = num_patch - self.num_patch_input  # 14 - 7 = 7
        adjust_input_ratio = self.num_patch_input / num_patch  # 7 / 14 = 0.5
        
        # safe_resize trả về: Resize(size = [224, 112], interpolation = "Bilinear", antialias=False)
        # ! 250426 bổ sung: image_size_per_var = 224/nvars, nghĩa là mỗi biến chiếm 224/nvars điểm ảnh theo chiều dọc.
        # ! Nếu không chia hết, phần thiếu sẽ được bù vào cuối ảnh (pad_down).
        image_size_per_var = int(self.image_size / nvars)  # 224 // nvars
        input_resize = safe_resize((image_size_per_var, int(self.image_size * adjust_input_ratio)), interpolation=Image.BILINEAR)
        
        pad_left = 0
        pad_right = 0
        
        if context_len % periodicity != 0:
            pad_left = periodicity - context_len % periodicity  # Nếu không chia hết cho chu kỳ thì đệm thêm ở phía trái
        if pred_len % periodicity != 0:
            pad_right = periodicity - pred_len % periodicity

        # Vì số chu kỳ trong context_len phải ánh xạ vào 112 điểm ảnh, scale_x biểu diễn số chu kỳ cho mỗi điểm ảnh theo chiều ngang.
        # Ví dụ context_len gồm 8 chu kỳ thì scale_x = 8/112 = 1/14 ≈ 0,0714.
        scale_x = ((pad_left + context_len) // periodicity) / (int(self.image_size * adjust_input_ratio))

        # output_resize = safe_resize((periodicity, int(round(image_size * scale_x))), interpolation=interpolation)

        # 1. Chuẩn hóa
        means = x.mean(1, keepdim=True).detach()  # x: [bs x seq_len x nvars], means: [bs x 1 x nvars]
        x_enc = x - means
        stdev = torch.sqrt(
            torch.var(x_enc.to(torch.float64), dim=1, keepdim=True, unbiased=False) + 1e-5)  # [bs x 1 x nvars]
        # ! norm_const thường đặt 0.4 để giới hạn độ lệch chuẩn nhằm thu hẹp biên độ sau chuẩn hóa
        stdev /= self.norm_const
        x_enc /= stdev
        # Kênh độc lập
        x_enc = rearrange(x_enc, 'b s n -> b n s')  # x_enc: [bs x nvars x seq_len]

        # TODO: mục đích cụ thể của đoạn mã này là gì?
        # ! 250425 bổ sung: mục tiêu là tạo ra 5% xác suất chuỗi bị che bên trái để mô phỏng trường hợp pred_len dài hơn seq_len.
        lookback_masked = False
        if random.random() < self.pre_mask_prob:  # self.pre_mask_prob = 0.05
            # self.max_pre_mask_ratio = 0.5 -> có 5% xác suất nửa bên trái ảnh được đặt toàn số 0
            # ! 250410 bổ sung: nhân thêm random.random() để giảm khả năng phần mặt nạ quá dài
            mask_len = int(len(data[0]) * self.max_pre_mask_ratio * random.random())
            # x_enc[:, :, :int(len(data[0]) * self.max_pre_mask_ratio)] = 0
            x_enc[:, :, :mask_len] = 0
            lookback_masked = True

        # 2. Phân đoạn
        # Ghi chú: khi pad có hai tham số, nghĩa là bổ sung ở chiều cuối cùng: pad = (đệm bên trái = pad_left, đệm bên phải = 0)
        x_pad = F.pad(x_enc, (pad_left, 0), mode='constant')  # [bs, nvars, seq_len]

        # ! f chính là chu kỳ, p là số chu kỳ
        # ! Lưu ý: biểu thức trước là (p f) còn sau là f p vì ban đầu dữ liệu theo thứ tự ngang, khi đổi sang ảnh thì phải đảo lại theo chiều dọc.
        # ! 250424 bổ sung: x_2d.shape = [bs, nvars, periodicity, số chu kỳ p]
        x_2d = rearrange(x_pad, 'b n (p f) -> b n f p', f=periodicity)  # x_2d.shape: [bs, nvars, độ dài chu kỳ f, số chu kỳ p]

        # 3. Dựng ảnh & Căn chỉnh
        # ! Dữ liệu [độ dài chu kỳ f, số chu kỳ p] được co dãn về [224, 112] để tạo nửa trái của ảnh (nửa phải sẽ là mặt nạ)
        x_resize = input_resize(x_2d)  # x_resize.shape: [bs, nvars, 224//nvars, ~112]
        x_resize = rearrange(x_resize, 'b n h w -> b 1 (n h) w')  # x_resize.shape: [bs, 1, nvars*224//nvars, 112]
        # Vì nvars có thể không chia hết cho 224 nên cuối ảnh có thể cần đệm thêm (pad_down)
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


        # masked: [bs, 1, 224, 112], nửa bên phải là vùng mặt nạ
        masked = torch.zeros((x_2d.shape[0], 1, self.image_size, num_patch_output * self.patch_size), device=x_2d.device, dtype=x_2d.dtype)
        x_concat_with_masked = torch.cat([
            x_resize, 
            masked
        ], dim=-1)  # Ghép thành [bs, 1, 224, 224] (112 điểm ảnh gốc + 112 điểm ảnh mặt nạ)
        
        # TODO: 250426 bổ sung: có thể dùng màu để phân biệt từng biến
        
        # # ! Phương án 1: không tô màu, chỉ lặp lại 3 lần
        # image_input = repeat(x_concat_with_masked, 'b 1 h w -> b c h w', c=3)  # Sao chép thành 3 kênh
        
        # ! Phương án 2: gán màu ngẫu nhiên cho từng biến
        # color_dict = {0: [0,0,0], 1: [0,0,1]}
        image_input = torch.zeros((x_concat_with_masked.shape[0], 3, x_concat_with_masked.shape[2], x_concat_with_masked.shape[3]), 
                                  device=x_concat_with_masked.device, 
                                  dtype=x_concat_with_masked.dtype)  # [bs, 3, 224, 224]
        color_list = []
        for i in range(nvars):
            if i == 0: 
                color = random.randint(0, 2)  # random.randint bao gồm cả hai đầu mút
            else:
                tmp_color = random.randint(0, 2)
                while tmp_color == color:
                    tmp_color = random.randint(0, 2)
                color = tmp_color
            color_list.append(color)
            # Gán dữ liệu vào kênh màu đã chọn
            image_input[:, color, i*image_size_per_var:(i+1)*image_size_per_var, :] = \
                x_concat_with_masked[:, 0, i*image_size_per_var:(i+1)*image_size_per_var, :]
        

        # Đầu ra
        # x và y cũng được chuẩn hóa
        y = (y - means) / stdev  # [bs, pred_len, nvars]
        x = (x - means) / stdev  # [bs, seq_len, nvars]
        
        
        # ! 250424 bổ sung: không cần tính pixel loss ở đây
        # # Pixel loss
        # y_true = y[:, :, :] # [bs, Pred_len, nvars]
        # y_mask = torch.ones_like(y_true) # [1, Pred_len, nvars]
        
        # y_true = rearrange(y_true, 'b pl n -> b n pl')  # [bs x nvars x seq_len]
        # y_mask = rearrange(y_mask, 'b pl n -> b n pl')  # [bs x nvars x seq_len]
        
        # if pad_right != 0:
        #     # Đệm ở chiều pred_len: giữ bên trái, bổ sung pad_right bên phải
        #     y_true = F.pad(y_true, (0, pad_right), mode='replicate')  # [1, nvars, Pred_len]
        #     y_mask = F.pad(y_mask, (0, pad_right), mode='constant')
        
        # y_true_with_mask = torch.cat([y_true, y_mask], dim=0)  # [2, nvars, Pred_len]
        # # ! y_true_2d: [2, 1, độ dài chu kỳ, số chu kỳ]; hàng đầu là y thật, hàng thứ hai là y_mask (1: có nhãn, 0: không có)
        # # y_true_2d = rearrange(y_true_with_mask, 'b (p f) -> b 1 f p', f=periodicity)
        # # ! 250424 bổ sung: y_true_2d.shape = [2, 1, nvars * periodicity, số chu kỳ p]
        # y_true_2d = rearrange(y_true_with_mask, 'b n (p f) -> b 1 (n f) p', f=periodicity)
        
        # # Số điểm ảnh tương ứng với pred_len trong cửa sổ dự báo, ví dụ 19 / 0.83 ≈ 23
        # target_width = int(round(y_true_2d.shape[3] / float(scale_x)))
        # # Do phần mặt nạ có 112 điểm ảnh ngang nên phần còn lại 112-23=89 được dùng để đệm
        # padding_width = num_patch_output * self.patch_size - target_width

        # # Nội suy đến kích thước [2, 1, 224, target_width]
        # y_true_2d = safe_resize((self.image_size, target_width), interpolation=Image.BILINEAR)(y_true_2d)
        # # Điền 0 cho phần bên phải không có giá trị (112-23=89), thu được [2, 1, 224, 112]
        # y_true_2d = F.pad(y_true_2d, (0, padding_width), mode='constant')
        # # y_true_2d.shape: [2, 224, 112]
        # y_true_2d = rearrange(y_true_2d, 'b 1 h w -> b h w')

        
        # ! 250425 ghi chú: chỉ số [0] vốn nhằm giữ chiều batch, nhưng ở đây có thể bỏ được
        data_entry[field] = image_input.float().numpy()[0]  # [3, 224, 224]
        # data_entry[field + "_img"] = y_true_2d.float().numpy()  # [1, 224, 224]
        # data_entry[field + "_img"] = None  # [1, 224, 224]
        
        data_entry['pad_left'] = int(pad_left)
        data_entry['context_len'] = int(context_len)
        data_entry['pred_len'] = int(pred_len)
        data_entry['periodicity'] = int(periodicity)
        
        # ! 250425 ghi chú: tương tự, chiều batch được loại bỏ khi chuyển về NumPy
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

        # 1. Chuẩn hóa
        means = x.mean(1, keepdim=True).detach()  # [bs x 1 x nvars]
        x_enc = x - means
        stdev = torch.sqrt(
            torch.var(x_enc.to(torch.float64), dim=1, keepdim=True, unbiased=False) + 1e-5
        )  # [bs x 1 x nvars]
        stdev /= self.norm_const
        x_enc /= stdev
        # Kênh độc lập
        x_enc = rearrange(x_enc, 'b s n -> b n s')  # [bs x nvars x seq_len]


        if random.random() < self.pre_mask_prob:
            x_enc[:, :, :int(len(data) * self.max_pre_mask_ratio)] = 0

        # 2. Phân đoạn
        x_pad = F.pad(x_enc, (pad_left, 0), mode='replicate')  # [b n s]
        x_2d = rearrange(x_pad, 'b n (p f) -> (b n) 1 f p', f=periodicity)

        # 3. Dựng ảnh & Căn chỉnh
        x_resize = input_resize(x_2d)
        masked = torch.zeros((x_2d.shape[0], 1, self.image_size, num_patch_output * self.patch_size), device=x_2d.device, dtype=x_2d.dtype)
        x_concat_with_masked = torch.cat([
            x_resize, 
            masked
        ], dim=-1)
        image_input = repeat(x_concat_with_masked, 'b 1 h w -> b c h w', c=3)

        # Đầu ra
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

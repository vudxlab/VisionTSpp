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

from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np

from uni2ts.common.sampler import Sampler, get_sampler
from uni2ts.common.typing import UnivarTimeSeries

from ._base import Transformation
from ._mixin import CheckArrNDimMixin, CollectFuncMixin, MapFuncMixin


@dataclass
class SampleDimension(
    CheckArrNDimMixin, CollectFuncMixin, MapFuncMixin, Transformation
):
    max_dim: int
    fields: tuple[str, ...]
    optional_fields: tuple[str, ...] = tuple()
    sampler: Sampler = get_sampler("uniform")

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        total_field_dim = sum(
            self.collect_func_list(
                self._get_dim,
                data_entry,
                self.fields,
                optional_fields=self.optional_fields,
            )
        )
        self.map_func(
            partial(self._process, total_field_dim=total_field_dim),  # noqa
            data_entry,
            self.fields,
            optional_fields=self.optional_fields,
        )
        return data_entry

    def _get_dim(self, data_entry: dict[str, Any], field: str) -> int:
        self.check_ndim(field, data_entry[field], 2)
        return len(data_entry[field])

    def _process(
        self, data_entry: dict[str, Any], field: str, total_field_dim: int
    ) -> list[UnivarTimeSeries]:
        arr: list[UnivarTimeSeries] = data_entry[field]  # arr.shape: (dataset_num, seq_len)?
        rand_idx = np.random.permutation(len(arr))
        if self.max_dim == 1:  # max_dim = 1：只采样一个channel，表示就是单变量预测
            rand_idx = rand_idx[0]
            return [arr[rand_idx]]
        else:  # 否则就是采样多个变量，上限为max_dim！
            field_max_dim = (self.max_dim * len(arr)) // total_field_dim
            n = self.sampler(min(len(arr), field_max_dim))  # 这里是uniform_sampler，也即从[0, dim]的随机整数 -> 所以这里的n会比len(arr)还要再小一些，表示只采样一部分channels
            return [arr[idx] for idx in rand_idx[:n]]


@dataclass
class Subsample(Transformation):  # just take every n-th element
    fields: tuple[str, ...] = ("target", "past_feat_dynamic_real")

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        pass


class GaussianFilterSubsample(
    Subsample
):  # blur using gaussian filter before subsampling
    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        # gaussian filter
        return super()(data_entry)


class Downsample(Transformation):  # aggregate
    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        pass


class Upsample(Transformation):
    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        pass


@dataclass
class SampleOneDimension(
    CheckArrNDimMixin, CollectFuncMixin, MapFuncMixin, Transformation
):
    fields: tuple[str, ...]
    optional_fields: tuple[str, ...] = tuple()
    sampler: Sampler = get_sampler("uniform")

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        self.map_func(
            partial(self._process),  # noqa
            data_entry,
            self.fields,
            optional_fields=self.optional_fields,
        )
        return data_entry

    def _process(
        self, data_entry: dict[str, Any], field: str
    ) -> list[UnivarTimeSeries]:
        arr: list[UnivarTimeSeries] = data_entry[field]
        rand_idx = np.random.permutation(len(arr))[0]
        return [arr[rand_idx]]
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

from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any, Optional

import lightning as L
import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from torch import nn
from torch.distributions import Distribution
import torch.nn.functional as F
import os

from uni2ts.loss.packed import (
    PackedDistributionLoss,
    PackedLoss,
    PackedNLLLoss,
    PackedPointLoss,
)
from uni2ts.module.norm import RMSNorm
from uni2ts.module.position import (
    BinaryAttentionBias,
    LearnedEmbedding,
    LearnedProjection,
)
from uni2ts.module.ts_embed import MultiInSizeLinear, MultiOutSizeLinear
from uni2ts.optim import SchedulerType, get_scheduler
from uni2ts.transform import (
    AddObservedMask,
    AddTimeIndex,
    AddVariateIndex,
    DefaultPatchSizeConstraints,
    FixedPatchSizeConstraints,
    DummyValueImputation,
    ExtendMask,
    FlatPackCollection,
    FlatPackFields,
    GetPatchSize,
    ImputeTimeSeries,
    LastValueImputation,
    Imagify,
    ImagifyTS,
    ImagifyTS_Multivariate,
    MaskedPrediction,
    PackFields,
    PatchCrop,
    Patchify,
    SampleDimension,
    SelectFields,
    SequencifyField,
    Transformation,
    SampleOneDimension,
)
from uni2ts.common.env import env

from .module import MoiraiModule

from uni2ts.model.visionts import models_mae, safe_resize
from PIL import Image
from einops import repeat, rearrange
import matplotlib.pyplot as plt

class MoiraiPretrain(L.LightningModule):
    seq_fields: tuple[str, ...] = (
        "x",
        "y",
        # "observed_mask",
        # "time_id",
        # "variate_id",
        # "prediction_mask",
        # "patch_size",
    )
    # ! 250425 adds:
    seq_fields_mean_stdev: tuple[str, ...] = (
        "means",
        "stdev",
        "color_list",
    )
    pad_func_map: dict[str, Callable[[Sequence[int], np.dtype], np.ndarray]] = {
        "y": np.zeros,
    }

    def __init__(
        self,
        min_patches: int,
        min_mask_ratio: float,
        max_mask_ratio: float,
        num_patch_input: int,
        norm_const: int,
        max_dim: int,
        num_training_steps: int,
        num_warmup_steps: int,
        module_kwargs: Optional[dict[str, Any]] = None,
        num_samples: int = 100,
        beta1: float = 0.9,
        beta2: float = 0.98,
        loss_func: PackedDistributionLoss = PackedNLLLoss(),
        val_metric: Optional[PackedLoss | list[PackedLoss]] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        log_on_step: bool = False,
        # log_image_step: int = 1000,
        log_image_step: int = 200,
        load_ckpt: bool = True,
        max_pre_mask_ratio: float = 0.5,
        pre_mask_prob: float = 0.1,
        output_dist: Optional[str] = None,
        pixel_loss_weight: float = 0.0,
        pixel_loss_type: str = 'gaussian',
        loss_topk = 1.0,
        nonlinear_dist: bool = False,
        model_size: str = "base",
        clip_input: int = 0,
        complete_no_clip: bool = False,
    ):
        assert (
            num_warmup_steps <= num_training_steps
        ), f"num_warmup_steps ({num_warmup_steps}) should be <= num_training_steps ({num_training_steps})."
        super().__init__()
        self.save_hyperparameters(ignore=["module"])
        # self.module = MoiraiModule(**module_kwargs) if module is None else module
        self.module_kwargs = module_kwargs
        
        # ! MAE模型的size必须在这其中，包括base、large和huge
        size_list = ["base", "large", "huge"]
        assert model_size in size_list
        
        if model_size == "base":
            print("Use MAE base model!")
            if output_dist is None:
                self.module = models_mae.mae_vit_base_patch16()
            elif output_dist == 'gaussian':
                print(f"Use Gaussian, nonlinear_dist={nonlinear_dist}!")
                self.module = models_mae.mae_vit_base_patch16(logvar=True, nonlinear_dist=nonlinear_dist)
            elif output_dist == 'quantile':
                print(f"Use Quantile, nonlinear_dist={nonlinear_dist}!")
                self.module = models_mae.mae_vit_base_patch16(logvar=False, quantile=True, nonlinear_dist=nonlinear_dist)
        elif model_size == "large":
            print("Use MAE large model!")
            print(f"Use Quantile, nonlinear_dist={nonlinear_dist}")
            self.module = models_mae.mae_vit_large_patch16(logvar=False, quantile=True, nonlinear_dist=nonlinear_dist)
        elif model_size == "huge":
            print("Use MAE huge model!")
            print(f"Use Quantile, nonlinear_dist={nonlinear_dist}")
            self.module = models_mae.mae_vit_huge_patch14(logvar=False, quantile=True, nonlinear_dist=nonlinear_dist)

        print(f"type of clip_input? (0: no clip, 1: simple clip, 2: clip_new) : {self.hparams.clip_input}")
        print(f"Is using complete_no_clip? : {self.hparams.complete_no_clip}")
        
        # 打印一些超参数信息：
        print("output_dist:", self.hparams.output_dist)
        print("pixel_loss_weight:", self.hparams.pixel_loss_weight)
        print("pixel_loss_type:", self.hparams.pixel_loss_type)
        print("loss_topk:", self.hparams.loss_topk)
        print("nonlinear_dist:", self.hparams.nonlinear_dist)
        print("clip_input:", self.hparams.clip_input)
        print("complete_no_clip:", self.hparams.complete_no_clip)
        

        if load_ckpt:
            if model_size == "base":
                checkpoint = torch.load(os.path.join(env.VISIONTS_CHECKPOINT_PATH, "mae_visualize_vit_base.pth"), map_location='cpu')
            elif model_size == "large":
                checkpoint = torch.load(os.path.join(env.VISIONTS_CHECKPOINT_PATH, "mae_visualize_vit_large.pth"), map_location='cpu')
            elif model_size == "huge":
                checkpoint = torch.load(os.path.join(env.VISIONTS_CHECKPOINT_PATH, "mae_visualize_vit_huge.pth"), map_location='cpu')
            print(f"load ckpt: {self.module.load_state_dict(checkpoint['model'], strict=False)}")

        # if loss_topk < 1.0:
        #     self.module_reference = models_mae.mae_vit_base_patch16(logvar=True)
        #     self.module_reference.load_state_dict(self.module.state_dict())
        #     for n, p in self.module_reference.named_parameters():
        #         p.requires_grad = False

        self.image_size = self.module.patch_embed.img_size[0]
        self.patch_size = self.module.patch_embed.patch_size[0]
        self.num_patch = self.image_size // self.patch_size
        mask = torch.ones((self.num_patch, self.num_patch))
        mask[:, :self.hparams.num_patch_input] = torch.zeros((self.num_patch, self.hparams.num_patch_input))
        self.mask_ratio = torch.mean(mask).item()
        self.log_image_step = log_image_step
        self.register_buffer("mask", mask.float().reshape((1, -1)))


    def show_image(self, image, title=''):
        # image is [H, W, 3]
        imagenet_mean = np.array([0.485, 0.456, 0.406])
        imagenet_std = np.array([0.229, 0.224, 0.225])
        assert image.shape[2] == 3
        
        plt.imshow(torch.clip((image.cpu() * imagenet_std + imagenet_mean) * 255, 0, 255).int())
        plt.title(title, fontsize=16)
        plt.axis('off')

    def visualization(self, x, y, y_pred, input_image, reconstructed_image, cur_periodicity, cur_context_len, cur_pred_len, cur_nvars=1, y_pred_std=None, y_pred_25=None, y_pred_75=None):
        # x: [seq_len, 1/nvars], y & y_pred: [pred_len, 1/nvars]
        # input_image & reconstructed_image: [224, 224, 3]

        # Visualization
        plt.subplot(2, 2, 1)
        self.show_image(input_image, 'input')

        plt.subplot(2, 2, 2)
        plt.plot(x.cpu())
        plt.plot(torch.arange(len(y)) + len(x), y.cpu(), label='true', alpha=0.5)
        plt.title(f'input time series {cur_nvars} vars')
        plt.legend()
        plt.tight_layout()

        plt.subplot(2, 2, 3)
        self.show_image(reconstructed_image, 'reconstructed')

        plt.subplot(2, 2, 4)
        plt.plot(x.cpu())
        plt.plot(torch.arange(len(y)) + len(x), y.cpu(), label='true', alpha=0.5, color='C0')
        plt.plot(torch.arange(len(y)) + len(x), y_pred.cpu(), label='pred', color='C1')
        
        if y_pred_std is not None:
            plt.fill_between(torch.arange(len(y)) + len(x), (y_pred - y_pred_std).cpu(), (y_pred + y_pred_std).cpu(), color='C1', alpha=0.2)
        if y_pred_25 is not None and y_pred_75 is not None:
            plt.plot(torch.arange(len(y)) + len(x), y_pred_25.cpu(), label='pred_25', color='C2', alpha=0.5)
            plt.plot(torch.arange(len(y)) + len(x), y_pred_75.cpu(), label='pred_75', color='C3', alpha=0.5)

        # plt.title(f'forecasting')
        plt.title(f'period={cur_periodicity} ctx_len={cur_context_len}, pred_len={cur_pred_len}')
        plt.legend()
        plt.tight_layout()

        plt.draw()
        img_array = np.array(plt.gcf().canvas.buffer_rgba())
        plt.close()

        img_array = img_array[:, :, :3]
        img_array = np.transpose(img_array, (2, 0, 1))
        img_tensor = torch.from_numpy(img_array).float() / 255.0
        self.logger.experiment.add_image(f'image', img_tensor, self.global_step)


    # TODO:重点修改。
    def forward_module(self, module, target):
        # module的返回值
        _, vision_out, mask = module(
            target, 
            mask_ratio=self.mask_ratio, noise=repeat(self.mask, '1 l -> n l', n=target.shape[0])
        )
        
        if self.hparams.output_dist == 'gaussian':
            # 输出mean和logvar，其为被展平的图片序列
            vision_out, vision_out_logvar = vision_out
            y_logvar = module.unpatchify(vision_out_logvar, n_channels=1) # [(bs x nvars) x 1 x h x w]
        elif self.hparams.output_dist == 'quantile':
            # 输出mean、mean_25、mean_75三个值，其为被展平的图片序列
            vision_out, vision_out_25, vision_out_75 = vision_out
            # 补一下y_logvar
            y_logvar = None
        else:
            y_logvar = None
        
        # print("self.hparams.output_dist:", self.hparams.output_dist)  # quantile
        # print("vision_out.shape:", vision_out.shape)  # [bs=512, 196, 768]
        # print("vision_out_25.shape:", vision_out_25.shape)  # [bs=512, 196, 768]
        # print("vision_out_75.shape:", vision_out_75.shape)  # [bs=512, 196, 768]
        
        # 对于主数据处理一下：
        image_reconstructed = module.unpatchify(vision_out) # [(bs x nvars) x 3 x H=224 x W=224]
        y_grey = torch.mean(image_reconstructed, 1, keepdim=True) # [B, 1, H, W]
        
        # 计算预测窗口开始的位置offset：
        pred_pixel_offset = self.hparams.num_patch_input * module.patch_embed.patch_size[0]
        # 取出预测窗口部分的数据，但后面可能包含padding部分
        y_pred_img = image_reconstructed[:, :, :, pred_pixel_offset:] # [B, 3, H=224, W_pred=112]
        
        
        # 补一下y_pred_img for quantile
        if self.hparams.output_dist == 'quantile':
            # 对25和75分位的类似处理一下
            image_25_reconstructed = module.unpatchify(vision_out_25) # [(bs x nvars) x 3 x H x w]
            image_75_reconstructed = module.unpatchify(vision_out_75) # [(bs x nvars) x 3 x H x w]
            y_grey_25 = torch.mean(image_25_reconstructed, 1, keepdim=True) # [B, 1, H=224, W=224]，y_grey这里把三个channel做平均从RGB转成灰度图
            y_grey_75 = torch.mean(image_75_reconstructed, 1, keepdim=True) # [B, 1, H, W]
            y_pred_img_25 = image_25_reconstructed[:, :, :, pred_pixel_offset:] # [B, 3, H=224, W_pred=112]
            y_pred_img_75 = image_75_reconstructed[:, :, :, pred_pixel_offset:] # [B, 3, H, W_pred]
            
        
        # mask是MAE的mask，固定为7个patch：7个patch
        if not self.hparams.output_dist == 'quantile':
            return mask, y_logvar, image_reconstructed, y_grey, y_pred_img
        else:
            return mask, y_logvar, image_reconstructed, y_grey, y_pred_img, y_grey_25, y_pred_img_25, y_grey_75, y_pred_img_75
        

    # # rho-1方法，暂时丢弃
    # def create_topk_mask(self, score):
    #     # score: [B, N]
    #     k = int(round(score.shape[1] * self.hparams.loss_topk))
    #     score = score.detach()
    #     score += torch.randn_like(score) / 1000 # add small noise
    #     _, top_k_indices = torch.topk(score, k=k, dim=-1)
    #     mask = torch.zeros_like(score)
    #     mask.scatter_(dim=-1, index=top_k_indices, value=1)
    #     return mask
    

    # 图片转时间序列: 对整张图（包括look-back + pred_len）做resize、然后取出pred_len部分。
    def extract_TS_from_image(self, cur_y_grey, cur_periodicity, cur_scale_x, cur_pad_left, cur_context_len, cur_pred_len, cur_nvars, cur_pad_down):
        # ! 250425 adds: 这里记得要把cur_periodicity改成cur_periodicity * cur_nvars！！
        # 这里原来cur_y_grey.shape = [1, 1, 224, 224]
        # 转换之后y_segmentations变成[1, 1, periodicity, max周期个数]，
        # 这里max周期个数由224*cur_scale_x算出；
        # 而scale_x则是根据输入窗口中的周期数除以左半边像素点数112算出来的！例如10/112=0.0892
        
        # print(f"{cur_y_grey.shape = }")
        # print(f"{cur_pad_down = }")
        # print(f"{cur_nvars = }")
        # print(f"{cur_periodicity = }")
        # print(f"{cur_scale_x = }")
        
        if cur_pad_down > 0: 
            cur_y_grey = cur_y_grey[:, :, :-cur_pad_down, :]  # [1, 1, 224-cur_pad_down, 224]
        cur_y_grey = rearrange(cur_y_grey, 'b 1 (n h) w -> b n h w', n=cur_nvars)  # [1, nvars, 224//nvars, 224]
        output_resize = safe_resize((cur_periodicity, int(round(self.image_size * cur_scale_x))), interpolation=Image.BILINEAR)
        y_segmentations = output_resize(cur_y_grey)  # shape: [b, n, periodicity, num_periods]
        
        # print(f"{cur_periodicity = }, {cur_nvars = }")
        # print(f"{y_segmentations.shape = }")
        # assert cur_periodicity * cur_nvars == y_segmentations.shape[2]
        
        y_flatten = rearrange(
            y_segmentations, 
            'b n f p -> b (p f) n', 
            # f=cur_periodicity,
            # n=cur_nvars,
        ) # flatten -> shape变成[bs, total_len, nvars]
        
        start_idx = cur_pad_left + cur_context_len
        end_idx = cur_pad_left + cur_context_len + cur_pred_len
        y_pred = y_flatten[:, start_idx: end_idx, :]
        
        return y_pred  # 最后的shape为[bs, pred_len, nvars]

    
    def forward(
        self,
        target: Float[torch.Tensor, "*batch 3 image_size=224 image_size=224"],  # 输入，时间序列转成的RGB图片，其中左半边112为有值的输入，右半边则全为0。
        # target_img: Float[torch.Tensor, "*batch 2 image_size=224 left_image_size=112"],  # 预测窗口真实值、第一维也是时间序列转成的图片，另一维则是mask
        x: Float[torch.Tensor, "*batch max_seq_len max_nvars"],  # 输入窗口对应的时间序列形式
        y: Float[torch.Tensor, "*batch max_seq_len max_nvars"],  # 预测窗口对应的时间序列形式
        sample_id: Int[torch.Tensor, "*batch max_seq_len"],
        pad_left: Int[torch.Tensor, "*batch"],
        context_len: Int[torch.Tensor, "*batch"],
        pred_len: Int[torch.Tensor, "*batch"],
        periodicity: Int[torch.Tensor, "*batch"],
        scale_x: Float[torch.Tensor, "*batch"],
        means: Float[torch.Tensor, "*batch max_nvars"],
        stdev: Float[torch.Tensor, "*batch max_nvars"],
        norm_const: Float[torch.Tensor, "*batch"],
        lookback_masked: Bool[torch.Tensor, "*batch"],
        image_size: Int[torch.Tensor, "*batch"],
        patch_size: Int[torch.Tensor, "*batch"],
        num_patch_input: Int[torch.Tensor, "*batch"],
        nvars: Int[torch.Tensor, "*batch"],
        pad_down: Int[torch.Tensor, "*batch"],
        color_list: Int[torch.Tensor, "*batch max_nvars"],
        # dataset_name,
    ) -> Distribution:
        """
        Redirects to the forward function of MoiraiModule.

        :param target: input data
        :param observed_mask: binary mask for missing values, 1 if observed, 0 otherwise
        :param sample_id: indices indicating the sample index (for packing)
        :param time_id: indices indicating the time index
        :param variate_id: indices indicating the variate index
        :param prediction_mask: binary mask for prediction horizon, 1 if part of the horizon, 0 otherwise
        :param patch_size: patch size for each token
        :return: predictive distribution
        """
        
        # # x.shape = [bs, 8192, 1], y.shape = [bs, 8192, 1], target.shape = [bs, 3, 224, 224], target_img.shape = [bs, 2, 224, 112]
        # print(f"{x.shape = }, {y.shape = }, {target.shape = }, {target_img.shape = }")
        # # print(f"{x[0, :10, 0] = }, {y[0, :10, 0] = }, {target[0, 0, :3, :10] = }, {target_img[0, 0, :3, :10] = }")
        # # print(f"{x[0, :10, 0] = }, {y[0, :10, 0] = }, {target[0] = }, {target_img[0] = }")
        # # 判断是否有值大于 5 或小于 -5
        # print(f"x > 5: {torch.any((x > 5))}, x < -5: {torch.any((x < -5))}")
        # print(f"y > 5: {torch.any((y > 5))}, y < -5: {torch.any((y < -5))}")
        # print(f"target > 5: {torch.any((target > 5))}, target < -5: {torch.any((target < -5))}")
        # print(f"target_img > 5: {torch.any((target_img > 5))}, target_img < -5: {torch.any((target_img < -5))}")
        
        # condition = (target > 5) | (target < -5)
        # indices = torch.nonzero(condition).squeeze()
        # print(f"{indices = }")
        # # values = []
        # for index in indices:
        #     print(f"{index = }")
        #     print(f"{target[tuple(index.cpu().numpy())] = }")
        #     # values.append(target[index.cpu()].cpu())
        # # print(f"{values = }")
        
        
        # 250410 adds:
        # 超参数：循环检测几轮？
        # CHECK_ITERS = 3
        CHECK_ITERS = 2
        
        
        norm_const = float(norm_const[0])
        image_size = int(image_size[0])
        patch_size = int(patch_size[0])
        num_patch_input = int(num_patch_input[0])
        
        # print(f"{dataset_name = }")
        
        
        if self.hparams.clip_input == 0:
            if self.hparams.complete_no_clip:
                print(f"target > 5: {torch.any((target > 5))}, target < -5: {torch.any((target < -5))}")
                # pass
            else:
                y = torch.clip(y, -5, 5)
                x = torch.clip(x, -5, 5)
                target = torch.clip(target, -5, 5)
                # if target_img is not None: target_img = torch.clip(target_img, -5, 5)
        elif self.hparams.clip_input == 1:
            thres_down = -1.8044
            thres_up = 2.2489
            print(f"target > {thres_up}: {torch.any((target > thres_up))}, target < {thres_down}: {torch.any((target < thres_down))}")
            
            y = torch.clip(y, thres_down, thres_up)
            x = torch.clip(x, thres_down, thres_up)
            target = torch.clip(target, thres_down, thres_up)
            # if target_img is not None: target_img = torch.clip(target_img, thres_down, thres_up)
                
        # elif self.hparams.clip_input == 2:
        #     # 1. ImageNet的mean和std
        #     image_mean = [0.485,0.456,0.406]
        #     image_std = [0.229,0.224,0.225]
        #     # 计算 [-mean/std, (1-mean)/std]，得到的结果如下
        #     thres_down_list = [-2.1179039301310043, -2.0357142857142856, -1.8044444444444445]
        #     thres_up_list = [2.2489082969432315, 2.428571428571429, 2.6399999999999997]
        #     thres_down = -1.8044
        #     thres_up = 2.2489
            
        #     # print(f"x > {thres_up}: {torch.any((x > thres_up))}, x < {thres_down}: {torch.any((x < thres_down))}")
        #     print(f"before: target > {thres_up}: {torch.any((target > thres_up))}, target < {thres_down}: {torch.any((target < thres_down))}")
        
        #     # # option 2.1 对输入数据做clip，使得其不超出上下界
        #     # y = torch.clip(y, thres_down, thres_up)
        #     # x = torch.clip(x, thres_down, thres_up)
        #     # target = torch.clip(target, thres_down, thres_up)
        #     # target_img = torch.clip(target_img, thres_down, thres_up)
        
        #     # option 2.2 继续做clip和denorm和重新norm
            
        #     # 2.2.1 先对x和y做clip
        #     x_clipped = torch.clip(x, thres_down, thres_up)
        #     y_clipped = torch.clip(y, thres_down, thres_up)
    
        #     # # 2.2.2 denorm回去
        #     # x_denorm = x_clipped * stdev + means
        #     # y_denorm = y_clipped * stdev + means
            
        #     # 2.2.0 最早先判断一下，如果整个target本身就没超上下界，就不用做下面的东西了
        #     is_taregt_out = torch.any((target > thres_up)) | torch.any((target < thres_down))
        #     # print(f"is 'target' out of the clipped range? : {is_taregt_out}")
                      
            
        #     if is_taregt_out:
        #         bs = x.shape[0]
        #         x_new = torch.zeros_like(x).to(x.device)
        #         y_new = torch.zeros_like(y).to(y.device)
        #         # means_new = torch.zeros_like(means).to(means.device)
        #         # stdev_new = torch.zeros_like(stdev).to(stdev.device)
        #         target_new = torch.zeros_like(target).to(target.device)
        #         target_img_new = torch.zeros_like(target_img).to(target_img.device)
                
        #         for i in range(bs):
        #             check_cnt = 0
        #             # 2.2.3 先判断一下，如果数据本身就没超上下界，就不用做下面的东西了
        #             is_taregt_i_out = torch.any((target[i] > thres_up)) | torch.any((target[i] < thres_down))
        #             # print(f"is 'target[{i}]' out of the clipped range? : {is_taregt_i_out}")
                    
        #             lookback_masked_i = lookback_masked[i]
                    
        #             if (not is_taregt_i_out) or lookback_masked_i:
        #                 target_clipped_i = torch.clip(target[i], thres_down, thres_up)
        #                 target_img_clipped_i = torch.clip(target_img[i], thres_down, thres_up)
                        
        #                 x_new[i] = x_clipped[i]
        #                 y_new[i] = y_clipped[i]
        #                 target_new[i] = target_clipped_i
        #                 target_img_new[i] = target_img_clipped_i
                        
        #             else:
        #                 # 取出当前list的数据并做denorm
        #                 cur_x_len = context_len[i]
        #                 cur_x = x_clipped[i:i+1, :cur_x_len, :]
        #                 cur_x_denormed = cur_x * stdev[i:i+1] + means[i:i+1]
                        
        #                 while True:
        #                     # 2.2.3 重新norm一遍
        #                     cur_means = cur_x_denormed.mean(1, keepdim=True).detach()  # cur_x: [bs=1 x seq_len x nvars]
                            
        #                     cur_x_enc = cur_x_denormed - cur_means
        #                     cur_stdev = torch.sqrt(
        #                         torch.var(cur_x_enc.to(torch.float64), dim=1, keepdim=True, unbiased=False) + 1e-5) # [bs x 1 x nvars]
        #                     # ! norm_const: 一般设置为0.4，用于约束增大标准差stdev，使得norm之后的值的范围更小！
        #                     cur_stdev /= norm_const
        #                     cur_x_enc /= cur_stdev  # [bs, seq_len, nvars]
        #                     # # Channel Independent
        #                     # cur_x_enc = rearrange(x_enc, 'b s n -> b n s') # [bs x nvars x seq_len]
                            
        #                     # 判断是否仍然超出边界 or 循环次数未达到，是的话则重新clip后denorm回来
        #                     is_cur_x_enc_out = torch.any((cur_x_enc > thres_up)) | torch.any((cur_x_enc < thres_down))
        #                     check_cnt += 1
        #                     if not is_cur_x_enc_out or check_cnt >= CHECK_ITERS:
        #                         break
                            
        #                     # 重新clip + denorm回来
        #                     cur_x_enc = torch.clip(cur_x_enc, thres_down, thres_up)
        #                     cur_x_denormed = cur_x_enc * cur_stdev + cur_means
                        
        #                 # 走到这里的话，说明cur_x_enc已经可以用了
        #                 # 于是先把y denorm回来，然后用新的mean和stdev把原始的y重新norm
        #                 cur_y_len = pred_len[i]
        #                 cur_y = y_clipped[i:i+1, :cur_y_len, :]
        #                 cur_y_denormed = cur_y * stdev[i:i+1] + means[i:i+1]
                        
        #                 cur_y_enc = (cur_y_denormed - cur_means) / cur_stdev  # [bs x pred_len x nvars]
                        
        #                 # 最后clip一次后准备送入模型
        #                 cur_x_enc = torch.clip(cur_x_enc, thres_down, thres_up)
        #                 cur_y_enc = torch.clip(cur_y_enc, thres_down, thres_up)
                        
        #                 x_new[i, :cur_x_len, :] = cur_x_enc[0]
        #                 y_new[i, :cur_y_len, :] = cur_y_enc[0]
                        
                        
        #                 # 然后仿照imagifyTS步骤，重新做一遍得到target和target_img
        #                 cur_periodicity = periodicity[i]
                        
        #                 num_patch = image_size // patch_size
        #                 num_patch_output = num_patch - num_patch_input
        #                 adjust_input_ratio = num_patch_input / num_patch
        #                 input_resize = safe_resize((image_size, int(image_size * adjust_input_ratio)), interpolation=Image.BILINEAR)
        #                 cur_pad_left = 0
        #                 pad_right = 0
        #                 if cur_x_len % cur_periodicity != 0:
        #                     cur_pad_left = cur_periodicity - cur_x_len % cur_periodicity
        #                 if cur_y_len % cur_periodicity != 0:
        #                     pad_right = cur_periodicity - cur_y_len % cur_periodicity
                        
        #                 cur_scale_x = ((cur_pad_left + cur_x_len) // cur_periodicity) / (int(image_size * adjust_input_ratio))
        #                 # print(f"{cur_scale_x = }, {scale_x[i] = }")
        #                 # print(f"{float(cur_scale_x.cpu().item()) = }, {float(scale_x[i].cpu().item()) = }")
        #                 # assert round(float(cur_scale_x.cpu().item()), 3) == round(float(scale_x[i].cpu().item()), 3)
                        
                        
        #                 cur_x_enc = rearrange(cur_x_enc, 'b s n -> b n s') # [bs x nvars x seq_len]

        #                 # 2. Segmentation
        #                 cur_x_pad = F.pad(cur_x_enc, (cur_pad_left, 0), mode='constant') # [b n s]
        #                 cur_x_2d = rearrange(cur_x_pad, 'b n (p f) -> (b n) 1 f p', f=cur_periodicity)

        #                 # 3. Render & Alignment
        #                 cur_x_resize = input_resize(cur_x_2d)
        #                 cur_masked = torch.zeros((cur_x_2d.shape[0], 1, image_size, num_patch_output * patch_size), device=cur_x_2d.device, dtype=cur_x_2d.dtype)
        #                 cur_x_concat_with_masked = torch.cat([
        #                     cur_x_resize, 
        #                     cur_masked
        #                 ], dim=-1)
        #                 cur_image_input = repeat(cur_x_concat_with_masked, 'b 1 h w -> b c h w', c=3)
                        
        #                 # Pixel loss
        #                 y_true = cur_y_enc[:, :, 0] # [1, Pred_len]
        #                 y_mask = torch.ones_like(y_true) # [1, Pred_len]
        #                 if pad_right != 0:
        #                     y_true = F.pad(y_true, (0, pad_right), mode='replicate') # [1, Pred_len]
        #                     y_mask = F.pad(y_mask, (0, pad_right), mode='constant')
        #                 y_true_with_mask = torch.cat([y_true, y_mask], dim=0) # [2, Pred_len]
        #                 y_true_2d = rearrange(y_true_with_mask, 'b (p f) -> b 1 f p', f=cur_periodicity)
        #                 target_width = int(round(y_true_2d.shape[3] / float(cur_scale_x)))
        #                 padding_width = num_patch_output * patch_size - target_width
        #                 y_true_2d = safe_resize((image_size, target_width), interpolation=Image.BILINEAR)(y_true_2d) # [2, 1, Image_Size, Target_Width]
        #                 y_true_2d = F.pad(y_true_2d, (0, padding_width), mode='constant') # [2, 1, Image_Size, Mask_Size]
        #                 y_true_2d = rearrange(y_true_2d, 'b 1 h w -> b h w') # [2, Image_Size, Mask_Size]
                        
                        
        #                 # 再把target这些加进去
        #                 target_new[i] = cur_image_input[0]
        #                 target_img_new[i] = y_true_2d[0]
                                
        #         # 遍历完batch后，把结果更新到x, y, target, target_img上
        #         x = torch.clip(x_new, thres_down, thres_up)
        #         y = torch.clip(y_new, thres_down, thres_up)
        #         print(f"after: target > {thres_up}: {torch.any((target > thres_up))}, target < {thres_down}: {torch.any((target < thres_down))}")
        #         target = torch.clip(target_new, thres_down, thres_up)
        #         target_img = torch.clip(target_img_new, thres_down, thres_up)
        #     else:
        #         x = x_clipped
        #         y = y_clipped
        #         target = torch.clip(target, thres_down, thres_up)
        #         target_img = torch.clip(target_img, thres_down, thres_up)
        
        
        periodicity = periodicity.cpu().tolist()
        pad_left = pad_left.cpu().tolist()
        context_len = context_len.cpu().tolist()
        pred_len = pred_len.cpu().tolist()
        scale_x = scale_x.cpu().tolist()
        cur_nvars = nvars.cpu().tolist()
        pad_down = pad_down.cpu().tolist()
        color_list = color_list.cpu().tolist()


        # print(f"{x.shape = }, {y.shape = }, {target.shape = }, {target_img.shape = }")
        # print(f"{x.device = }, {y.device = }, {target.device = }, {target_img.device = }")
        

        # y_logvar虽然生成的时候是按照图像大小生成的，但是他只会被展平+resize后和时间序列计算loss来监督训练。
        # mask.shape: [bs, ???]
        # image_reconstructed.shape: [(bs x nvars) x 3 x H=224 x W=224]
        if not self.hparams.output_dist == 'quantile':
            mask, y_logvar, image_reconstructed, y_grey, y_pred_img = self.forward_module(self.module, target)
        else:
            mask, y_logvar, image_reconstructed, y_grey, y_pred_img, y_grey_25, y_pred_img_25, y_grey_75, y_pred_img_75 = self.forward_module(self.module, target)
        
        
        if self.global_step % self.log_image_step == 0:
            mask = mask.detach()
            # unsqueeze后的mask为：(bs, num_patch*num_patch = 14*14, patch_size*patch_size*3 = 16*16*3)
            mask = mask.unsqueeze(-1).repeat(1, 1, self.module.patch_embed.patch_size[0]**2 *3)
            # unpatchify后为：(bs, 3, 224, 224)
            mask = self.module.unpatchify(mask)  # mask=1表示掩码部分, mask=0表示未掩码部分
            # mask = torch.einsum('nchw->nhwc', mask)
            
            #  image_reconstructed_disp就是：将原输入的未mask部分、和预测的mask部分成重建后的图像拼在一起
            image_input = target  # [bs, 3, 224, 224]
            image_reconstructed_disp = image_input * (1 - mask) + image_reconstructed * mask  # [bs, 3, 224, 224]
            green_bg = -torch.ones_like(image_reconstructed_disp) * 2  # gree_bg中的值全为-2！
            
            image_input = image_input * (1 - mask) + green_bg * mask  # mask为1表示被掩码的部分用-2填充，也即非常黑的颜色！
            image_input = rearrange(image_input, '(b n) c h w -> b n h w c', b=x.shape[0])  # [bs, 1, 224, 224, 3]
            image_reconstructed_disp = rearrange(image_reconstructed_disp, '(b n) c h w -> b n h w c', b=x.shape[0])  # [bs, 1, 224, 224, 3]


        loss_list = []
        loss_list_img = []
        
        # 一开始，所有monitor的参数都在这里注册一下！
        if not self.hparams.output_dist == 'quantile':
            monitors = {'mse': [], 'mae': [], 'std': [], 'px_loss': [], 'time_loss': [], 'diff_score_pixel': []}
        else:
            monitors = {'mse': [], 'mae': [], 'std': [], 'px_loss': [], 'time_loss': [], 'diff_score_pixel': [], 
                        'q0.25_loss': [], 'q0.50_loss': [], 'q0.75_loss': [], 
                        'q0.25_loss_img': [], 'q0.50_loss_img': [], 'q0.75_loss_img': [], 
                        f'cur_pixel_loss_{self.hparams.pixel_loss_type}': [], 
                        }

        # # 1. Pixel loss，也即图像loss
        # # target_img_mask这里是指padding的mask
        # # 也即预测窗口中的后半部分是没有值，应当抛弃掉的。
        # if target_img is not None: 
        #     target_img_grey = target_img[:, 0].unsqueeze(1) # [B, 1, H, W_pred]
        #     target_img_mask = target_img[:, 1].unsqueeze(1) # [B, 1, H, W_pred]


        # # 图像loss这里用huber效果更好
        # # 现在想改成quantile试一试？
        # # 这里改成只对图像数据算pixel_loss
        # if self.hparams.pixel_loss_type == 'mse':
        #     pixel_loss = (y_pred_img - target_img_grey) ** 2 # [B, 3, H, W_pred]
        # elif self.hparams.pixel_loss_type == 'huber':
        #     pixel_loss = F.huber_loss(y_pred_img, repeat(target_img_grey, 'b 1 h w -> b c h w', c=3), reduction='none') # [B, 3, H, W_pred]
        
        # # 去掉padding部分
        # pixel_loss = (pixel_loss * target_img_mask).sum() / target_img_mask.sum()

        # monitors['px_loss'].append(pixel_loss)
        
        
        
        # # ! RHO-1部分：
        # if self.hparams.loss_topk < 1.0:
        #     with torch.no_grad():
        #         _, _, _, y_grey_ref, y_pred_img_ref = self.forward_module(self.module, target)
        #         diff = (y_pred_img.mean(1, keepdims=True) - target_img_grey).abs() - (y_pred_img_ref.mean(1, keepdims=True) - target_img_grey).abs()
        #         pixel_score = diff + (1 - target_img_mask) * (-1000)
        #         pixel_score = rearrange(pixel_score, 'b 1 h w -> b (h w)')
        #         pixel_loss_mask = self.create_topk_mask(pixel_score)
        #         pixel_loss_mask = rearrange(pixel_loss_mask, 'b (h w) -> b 1 h w', h=y_pred_img.shape[2])
        #         monitors['diff_score_pixel'].append(pixel_score.mean())
        #     target_img_mask = target_img_mask * pixel_loss_mask


        
        # TODO: parallel
        # 2、时间序列loss，但是无法并行，需要对batch中的每个样本循环去算
        # ? why?是因为每个样本的周期都不一样么？
        # has_drawn_time_series = False
        
        for i in range(len(y_grey)):
            cur_context_len = context_len[i]
            # print("cur_context_len:", cur_context_len)
            # ! 由于混入的真实图像数据的context_len会被设置为0，由此来区分图像or真实数据
            # ! 因此不需要对真实图像数据计算TS的loss！
            # ! 250314修改：现在改为对pixel的计算pixel loss了
            if cur_context_len == 0: # ctx_len==0 means an image data. Ignore it!
                assert False, "Pixel loss is not supported!"
                # if self.hparams.pixel_loss_type == 'mse':
                #     cur_pixel_loss = (y_pred_img - target_img_grey) ** 2  # [B, 3, H, W_pred]
                # elif self.hparams.pixel_loss_type == 'huber':
                #     cur_pixel_loss = F.huber_loss(y_pred_img, repeat(target_img_grey, 'b 1 h w -> b c h w', c=3), reduction='none') # [B, 3, H, W_pred]
                # elif self.hparams.pixel_loss_type == 'quantile':
                #     # y_pred_img_25, y_pred_img, y_pred_img_75分别表示25%, 50%, 75%的预测的图片值
                #     # 定义分位数的q值
                #     quantiles = [0.25, 0.5, 0.75]
                #     predictions_img = [y_pred_img_25[i], y_pred_img[i], y_pred_img_75[i]]
                #     # 定义加权的权重
                #     w_list = [1/3, 1/3, 1/3]
                    
                #     cur_label_img = repeat(target_img_grey[i], '1 h w -> c h w', c=3)
                    
                #     cur_time_loss_img = 0
                #     for q, pred_img, w in zip(quantiles, predictions_img, w_list):
                #         error_img = cur_label_img - pred_img
                #         quantile_loss_img = torch.max(q * error_img, (q - 1) * error_img)  # quantile loss公式
                        
                #         # ! 注意一下：pixel_loss中只对有值的部分计算！
                #         # 所以这里直接在里面算mean了！
                #         cur_pixel_loss = (quantile_loss_img * target_img_mask).sum() / target_img_mask.sum()
                        
                #         # monitors[f"q{q:.2f}_loss_img"].append(quantile_loss_img.mean())
                #         monitors[f"q{q:.2f}_loss_img"].append(cur_pixel_loss)
                #         # 对不同分位数的loss进行加权
                #         cur_time_loss_img += w * cur_pixel_loss
                #     # # 另外delta_y_img和delta_y_square_img还是要记录一下，后面要记录mse_img和mae_img
                #     # delta_y_img = pred_img - cur_label_img
                #     # delta_y_img_square = delta_y_img ** 2
                
                # # 计算所有样本
                # # 前面算了mean的话这里可以不算mean了
                # # cur_time_loss_img = cur_time_loss_img.mean()
                # monitors[f'cur_pixel_loss_{self.hparams.pixel_loss_type}'].append(cur_time_loss_img)

                # loss_list_img.append(cur_time_loss_img)
                
                # # monitors["mse_img"].append(delta_y_img_square.mean())
                # # monitors["mae_img"].append(delta_y_img.abs().mean())
                
            # 对时间序列计算time series loss：
            else:
                cur_periodicity = periodicity[i]
                cur_scale_x = scale_x[i]  # 图像和TS做resize时的比例
                cur_pad_left = pad_left[i]
                cur_pred_len = pred_len[i]
                cur_nvars = nvars[i]
                cur_pad_down = pad_down[i]
                cur_color_list = color_list[i][:cur_nvars]
                
                # ! 250425 adds: x里面的第二维只有前面cur_context_len的部分是有值的，后面都会填充0
                # ! 同理第三维只有前cur_nvars个变量有值，后面都填充为0
                cur_x = torch.unsqueeze(x[i, :cur_context_len, :cur_nvars], 0)  # [1, Seq_len, nvars]

                cur_y_true = torch.unsqueeze(y[i, :cur_pred_len, :cur_nvars], 0) # [1, Pred_len, nvars]
                cur_y_grey = torch.unsqueeze(y_grey[i], 0) # [1, 1, H=224, W=224]
                
                if self.hparams.output_dist == 'gaussian':
                    cur_y_logvar = torch.unsqueeze(y_logvar[i], 0) # [1, 1, H, W]
                    # 沿着batch纬度，将数据和logvar拼成两维
                    cur_y_grey = torch.cat([cur_y_grey, cur_y_logvar], dim=0)
                if self.hparams.output_dist == 'quantile':
                    cur_y_grey_25 = torch.unsqueeze(y_grey_25[i], 0) # [1, 1, H, W]
                    cur_y_grey_75 = torch.unsqueeze(y_grey_75[i], 0) # [1, 1, H, W]
                
                # if self.hparams.loss_topk < 1.0:
                #     cur_y_grey = torch.cat([cur_y_grey, y_grey_ref[i].detach().unsqueeze(0)], dim=0) # [3, 1, H, W]
                
                # 图像变成序列的预测结果
                # y_pred也是二维的，first batch: mean; second batch: std
                # y_pred.shape: [bs, pred_len, nvars]
                y_pred = self.extract_TS_from_image(cur_y_grey, cur_periodicity, cur_scale_x, cur_pad_left, cur_context_len, cur_pred_len, cur_nvars, cur_pad_down)  # ([3, L, nvars] if use RHO-1); first batch: mean; second batch: std
                if self.hparams.output_dist == 'quantile':
                    # 下面的几个均为[bs, pred_len, nvars]
                    y_pred_25 = self.extract_TS_from_image(cur_y_grey_25, cur_periodicity, cur_scale_x, cur_pad_left, cur_context_len, cur_pred_len, cur_nvars, cur_pad_down) # [2, L, 1]
                    y_pred_75 = self.extract_TS_from_image(cur_y_grey_75, cur_periodicity, cur_scale_x, cur_pad_left, cur_context_len, cur_pred_len, cur_nvars, cur_pad_down) # [2, L, 1]
                
                # if self.hparams.loss_topk < 1.0:
                #     y_pred_ref = y_pred[2]
                #     y_pred = y_pred[:2]
                #     with torch.no_grad():
                #         diff = (y_pred[0] - cur_y_true).abs() - (y_pred_ref - cur_y_true).abs() # [1, L, 1]
                #         loss_mask = self.create_topk_mask(diff[:, :, 0]).unsqueeze(-1) # [1, L, 1]
                # else:
                # 不做rho-1了，所以这里mask全为1，相当于没加。
                loss_mask = torch.ones((1, y_pred.shape[1], cur_nvars), device=y_pred.device) # [1, pred_len, nvars]

                
                if self.hparams.output_dist == 'gaussian':
                    delta_y = y_pred[0] - cur_y_true # [1, pred_len, nvars]
                    delta_y_square = delta_y ** 2  # MSE loss
                    y_pred, y_pred_logvar = torch.unsqueeze(y_pred[0], 0), torch.unsqueeze(y_pred[1], 0)
                    y_pred_logvar = torch.clamp(y_pred_logvar, -14)  # PS: clamp用于将输入中每个元素的范围限制到区间[min,max]上
                    # y_pred_logvar虽然在图像上生成，但是只用展平+resize后和时间序列的loss来监督的。
                    # 这里计算likelihood作为loss：
                    cur_time_loss = delta_y_square * torch.exp(-y_pred_logvar) + y_pred_logvar
                    cur_time_loss = cur_time_loss * loss_mask
                    y_pred_std = torch.exp(y_pred_logvar / 2)
                    monitors["std"].append(y_pred_std.mean())
                
                elif self.hparams.output_dist == 'quantile':
                    # y_pred_25, y_pred, y_pred_75分别表示25%, 50%, 75%的预测值
                    # 定义分位数的q值
                    quantiles = [0.25, 0.5, 0.75]
                    predictions = [y_pred_25, y_pred, y_pred_75]
                    # 定义加权的权重
                    w_list = [1/3, 1/3, 1/3]
                    
                    cur_time_loss = 0
                    for q, pred, w in zip(quantiles, predictions, w_list):
                        error = cur_y_true - pred  # error.shape: [1, pred_len, nvars]
                        quantile_loss = torch.max(q * error, (q - 1) * error)  # quantile loss公式
                        monitors[f"q{q:.2f}_loss"].append(quantile_loss.mean())
                        # 对不同分位数的loss进行加权
                        cur_time_loss += w * quantile_loss
                    # 另外delta_y和delta_y_square还是要记录一下，后面要记录mse和mae
                    delta_y = y_pred - cur_y_true  # [1, pred_len, nvars]
                    delta_y_square = delta_y ** 2
                else:
                    delta_y = y_pred - cur_y_true
                    # 核心就是一个MSE loss
                    delta_y_square = delta_y ** 2
                    cur_time_loss = delta_y_square * loss_mask
                
                # 计算所有样本
                cur_time_loss = cur_time_loss.mean()
                monitors['time_loss'].append(cur_time_loss)

                loss_list.append(cur_time_loss)
                monitors["mse"].append(delta_y_square.mean())
                monitors["mae"].append(delta_y.abs().mean())
            
            
            # 这里有可能出现i==0时的不是时间序列，而是图片，所以会出问题
            # 所以改用一个bool变量来控制
            
            if self.global_step % self.log_image_step == 0 and i == 0:
            # if self.global_step % self.log_image_step == 0 and not has_drawn_time_series:
                # 置为True表示现在已经画过了！
                has_drawn_time_series = True
                
                if self.hparams.output_dist == 'gaussian':
                    y_pred_std = y_pred_std[0, :, 0].detach()
                else:
                    y_pred_std = None
                # 250307 adds:
                if self.hparams.output_dist == 'quantile':
                    try:
                        cur_y_pred_25, cur_y_pred_75 = y_pred_25[0, :, 0].detach().float(), y_pred_75[0, :, 0].detach().float()
                    except:
                        # 如果当前batch第一个样本为图像，他是不会生成y_pred_25和y_pred75的
                        # 反正visual用，也不是很重要，直接设置成none就ok了～
                        cur_y_pred_25, cur_y_pred_75 = None, None
                else:
                    cur_y_pred_25, cur_y_pred_75 = None, None
                
                try:
                    # cur_x.shape为[bs, seq_len, nvars]，cur_y_true和y_pred的shape为[bs, pred_len, nvars]
                    # 这里目前都只画出他们的第一维
                    # PS：这里只有i==0的时候才visual！
                    self.visualization(cur_x[i, :, 0].detach().float(), 
                                       cur_y_true[i, :, 0].detach().float(), 
                                       y_pred[i, :, 0].detach().float(), 
                                       input_image=image_input[i, 0].detach().float(), 
                                       reconstructed_image=image_reconstructed_disp[i, 0].detach().float(), 
                                       cur_periodicity=cur_periodicity, 
                                       cur_nvars=cur_nvars,
                                       cur_context_len=cur_context_len,
                                       cur_pred_len=cur_pred_len,
                                       y_pred_std=y_pred_std, 
                                       y_pred_25=cur_y_pred_25, 
                                       y_pred_75=cur_y_pred_75)
                except:
                    print("Exception in visualization, skip it.")

        if len(loss_list) == 0:
            loss = 0.0
        else:
            loss = torch.stack(loss_list).mean()
        # loss = 0
        
        if len(loss_list_img) == 0:
            pixel_loss = 0
        else:
            pixel_loss = torch.stack(loss_list_img).mean()

        # ! 混合两个loss比例：
        # ! 目前pixel_loss_weight = 0.5最好。
        # loss = loss * (1 - self.hparams.pixel_loss_weight) + pixel_loss * self.hparams.pixel_loss_weight
        # ! 250314，不设置pixel_loss_weight了，把两个loss直接加起来好了。
        loss = loss + pixel_loss


        # p_to_loss_mean = {}
        # for p in p_to_loss:
        #     p_to_loss_mean[p] = sum(p_to_loss[p]) / (len(p_to_loss[p]))
        
        return loss, monitors


    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Implements LightningModule training_step. Logs training loss.

        :param batch: batched inputs
        :param batch_idx: index of current batch
        :return: training loss for current batch
        """
        loss, monitors = self(**batch)
        batch_size = (
            batch["sample_id"].max(dim=1).values.sum() if "sample_id" in batch else None
        )
        self.log(
            f"train/loss",
            loss,
            on_step=self.hparams.log_on_step,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
            rank_zero_only=True,
        )
        for monitor in monitors:
            if len(monitors[monitor]) == 0:
                continue
            self.log(
                f"train/{monitor}",
                torch.stack(monitors[monitor]).mean().item(),
                on_step=self.hparams.log_on_step,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
                batch_size=batch_size,
                rank_zero_only=True,
            )

        return loss

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        """
        Implements LightningModule validation_step. Logs validation loss and additional metrics from val_metric.

        :param batch:
        :param batch_idx:
        :param dataloader_idx:
        :return: validation loss for current batch
        """
        val_loss = self(**batch)
        batch_size = (
            batch["sample_id"].max(dim=1).values.sum() if "sample_id" in batch else None
        )
        self.log(
            f"val/{self.hparams.loss_func.__class__.__name__}",
            val_loss,
            on_step=self.hparams.log_on_step,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
            rank_zero_only=True,
        )

        return val_loss

    def configure_optimizers(self) -> dict:
        """
        Implements LightningModule configure_optimizers which defines the configuration of optimizer and learning rate
        scheduler.

        :return: dictionary of optimizers and learning rate schedulers
        """
        # decay = set()
        # no_decay = set()

        # whitelist_params = (
        #     LearnedProjection,
        #     MultiInSizeLinear,
        #     MultiOutSizeLinear,
        #     nn.Linear,
        # )
        # blacklist_params = (
        #     BinaryAttentionBias,
        #     LearnedEmbedding,
        #     RMSNorm,
        #     nn.Embedding,
        #     nn.LayerNorm,
        # )

        # for mn, m in self.named_modules():
        #     for pn, p in m.named_parameters():
        #         if not p.requires_grad:
        #             continue

        #         fpn = f"{mn}.{pn}" if mn else pn
        #         if pn.endswith("bias"):
        #             no_decay.add(fpn)
        #         elif pn.endswith("weight") and isinstance(m, blacklist_params):
        #             no_decay.add(fpn)
        #         else:

        #         # elif pn.endswith("weight") and isinstance(m, whitelist_params):
        #             decay.add(fpn)


        # ! validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        
        # inter_params = decay & no_decay
        # union_params = decay | no_decay
        # assert (
        #     len(inter_params) == 0
        # ), f"parameters {str(inter_params)} made it into both decay/no_decay sets!"
        # assert (
        #     len(param_dict.keys() - union_params) == 0
        # ), f"parameters {str(param_dict.keys() - union_params)} were not separated into either decay/no_decay set!"

        optim_groups = [
            {
                "params": filter(
                    lambda p: p.requires_grad,
                    param_dict.values(),
                ),
                "weight_decay": self.hparams.weight_decay,
            },
            # {
            #     "params": filter(
            #         lambda p: p.requires_grad,
            #         [param_dict[pn] for pn in sorted(list(no_decay))],
            #     ),
            #     "weight_decay": 0.0,
            # },
        ]

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.hparams.lr,
            betas=(self.hparams.beta1, self.hparams.beta2),
            eps=1e-6,
        )
        scheduler = get_scheduler(
            SchedulerType.COSINE_WITH_RESTARTS,
            optimizer,
            num_warmup_steps=self.hparams.num_warmup_steps,
            num_training_steps=self.hparams.num_training_steps,
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss",
                "interval": "step",
            },
        }

    @property
    def train_transform_map(self) -> dict[str, Callable[..., Transformation]]:
        """
        Get a dictionary of Transforms, with a default Transform as defined:
        SampleDimension: Subsample the variate dimension of a time series
        GetPatchSize: Get patch size for a given time series
        PatchCrop: Perform cropping on the time series
        PackFields: Pack each feature columns, including 'target' and 'past_feat_dynamic_real'.
        AddObservedMask: Add the observed_mask feature
        ImputeTimeSeries: Imputes missing values with 0
        Patchify: Perform patching
        AddVariateIndex: Add variate_id feature
        AddTimeIndex: Add time_id feature
        MaskedPrediction: Specify the task,
            i.e., sample the total input length, as well as sample the proportion of look-back window and prediction window length.
        ExtendMask: Add an auxiliary mask.
        FlatPackCollection: Pack/Merge along 'variate_id, time_id, prediction_mask, observed_mask, and target' dimensions.
        FlatPackFields: Pack/Merge 'target'.
        SequencifyField: sequencify the 'patch_size' field.
        SelectFields: Output the data of predefined fields

        :return: defaultdict with default Transform
        """

        def default_train_transform():
            return (
                # ! 250423 adds: 这里原来是SampleDimension。
                # ! 但是VisionTS原本是只能处理单变量序列，因此这里只sample第一个维度。
                # SampleOneDimension(
                #     fields=("target",),
                #     optional_fields=("past_feat_dynamic_real",),
                # )
                SampleDimension(
                    max_dim=self.hparams.max_dim,
                    fields=("target",),
                    optional_fields=("past_feat_dynamic_real",),
                )
                + GetPatchSize(
                    min_time_patches=self.hparams.min_patches,
                    target_field="target",
                    patch_sizes=[1],  # self.module.patch_sizes
                    patch_size_constraints=FixedPatchSizeConstraints(1, 1),  # DefaultPatchSizeConstraints()
                    offset=True,
                )
                + PatchCrop(
                    min_time_patches=self.hparams.min_patches,
                    max_patches=self.module_kwargs['max_seq_len'],
                    will_flatten=True,
                    offset=True,
                    fields=("target",),
                    optional_fields=("past_feat_dynamic_real",),
                )
                + PackFields(
                    output_field="target",
                    fields=("target",),
                    feat=False,
                )
                + PackFields(
                    output_field="past_feat_dynamic_real",
                    fields=tuple(),
                    optional_fields=("past_feat_dynamic_real",),
                    feat=False,
                )
                # + AddObservedMask(
                #     fields=("target",),
                #     optional_fields=("past_feat_dynamic_real",),
                #     observed_mask_field="observed_mask",
                #     collection_type=dict,
                # )
                + ImputeTimeSeries(
                    fields=("target",),
                    optional_fields=("past_feat_dynamic_real",),
                    imputation_method=LastValueImputation(),  # DummyValueImputation(value=0.0)
                )
                # + ImagifyTS(
                #     image_size=self.module.patch_embed.img_size[0],
                #     patch_size=self.module.patch_embed.patch_size[0],
                #     num_patch_input=self.hparams.num_patch_input,
                #     norm_const=self.hparams.norm_const,
                #     min_pred_ratio=self.hparams.min_mask_ratio,
                #     max_pred_ratio=self.hparams.max_mask_ratio,
                #     max_pre_mask_ratio=self.hparams.max_pre_mask_ratio,
                #     pre_mask_prob=self.hparams.pre_mask_prob,
                #     fields=("target",)
                # )
                + ImagifyTS_Multivariate(
                    image_size=self.module.patch_embed.img_size[0],
                    patch_size=self.module.patch_embed.patch_size[0],
                    num_patch_input=self.hparams.num_patch_input,
                    norm_const=self.hparams.norm_const,
                    min_pred_ratio=self.hparams.min_mask_ratio,
                    max_pred_ratio=self.hparams.max_mask_ratio,
                    max_pre_mask_ratio=self.hparams.max_pre_mask_ratio,
                    pre_mask_prob=self.hparams.pre_mask_prob,
                    fields=("target",)
                )
                + SelectFields(fields=[
                    "target",
                    # "target_img",
                    "y",
                    "x",
                    "pad_left",
                    "context_len",
                    "pred_len",
                    "periodicity",
                    "scale_x",
                    "means",
                    "stdev",
                    "norm_const",
                    "lookback_masked",
                    "image_size",
                    "patch_size",
                    "num_patch_input",
                    "nvars",
                    "pad_down",
                    "color_list",
                ])
            )

        return defaultdict(lambda: default_train_transform)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = super().state_dict(destination, prefix, keep_vars)
        state_dict = {k: v for k, v in state_dict.items() if "module_reference" not in k}
        return state_dict
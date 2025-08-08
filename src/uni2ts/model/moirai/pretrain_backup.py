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
        log_image_step: int = 1000,
        # log_image_step: int = 200,
        load_ckpt: bool = True,
        max_pre_mask_ratio: float = 0.5,
        pre_mask_prob: float = 0.1,
        output_dist: Optional[str] = None,
        pixel_loss_weight: float = 0.0,
        pixel_loss_type: str = 'gaussian',
        loss_topk = 1.0,
        nonlinear_dist: bool = False
    ):
        assert (
            num_warmup_steps <= num_training_steps
        ), f"num_warmup_steps ({num_warmup_steps}) should be <= num_training_steps ({num_training_steps})."
        super().__init__()
        self.save_hyperparameters(ignore=["module"])
        # self.module = MoiraiModule(**module_kwargs) if module is None else module
        self.module_kwargs = module_kwargs
        if output_dist is None:
            self.module = models_mae.mae_vit_base_patch16()
        elif output_dist == 'gaussian':
            print(f"Use Gaussian, nonlinear_dist={nonlinear_dist}")
            self.module = models_mae.mae_vit_base_patch16(logvar=True, nonlinear_dist=nonlinear_dist)
        elif output_dist == 'quantile':
            print(f"Use Quantile, nonlinear_dist={nonlinear_dist}")
            self.module = models_mae.mae_vit_base_patch16(logvar=False, quantile=True, nonlinear_dist=nonlinear_dist)

        # 打印一些超参数信息：
        print("output_dist:", self.hparams.output_dist)
        print("pixel_loss_weight:", self.hparams.pixel_loss_weight)
        print("pixel_loss_type:", self.hparams.pixel_loss_type)
        print("loss_topk:", self.hparams.loss_topk)
        print("nonlinear_dist:", self.hparams.nonlinear_dist)
        

        if load_ckpt:
            checkpoint = torch.load(os.path.join(env.VISIONTS_CHECKPOINT_PATH, "mae_visualize_vit_base.pth"), map_location='cpu')
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

    def visualization(self, x, y, y_pred, input_image, reconstructed_image, i, y_pred_std=None, y_pred_25=None, y_pred_75=None):
        # [T, ]

        # Visualization
        plt.subplot(2, 2, 1)
        self.show_image(input_image, 'input')

        plt.subplot(2, 2, 2)
        plt.plot(x.cpu())
        plt.plot(torch.arange(len(y)) + len(x), y.cpu(), label='true', alpha=0.5)
        plt.title('input time series')
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

        plt.title('forecasting')
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
        image_reconstructed = module.unpatchify(vision_out) # [(bs x nvars) x 3 x h x w]
        y_grey = torch.mean(image_reconstructed, 1, keepdim=True) # [B, 1, H, W]
        
        # 计算预测窗口开始的位置offset：
        pred_pixel_offset = self.hparams.num_patch_input * module.patch_embed.patch_size[0]
        # 取出预测窗口部分的数据，但后面可能包含padding部分
        y_pred_img = image_reconstructed[:, :, :, pred_pixel_offset:] # [B, 3, H, W_pred]
        
        
        # 补一下y_pred_img for quantile
        if self.hparams.output_dist == 'quantile':
            # 对25和75分位的类似处理一下
            image_25_reconstructed = module.unpatchify(vision_out_25) # [(bs x nvars) x 3 x h x w]
            image_75_reconstructed = module.unpatchify(vision_out_75) # [(bs x nvars) x 3 x h x w]
            y_grey_25 = torch.mean(image_25_reconstructed, 1, keepdim=True) # [B, 1, H, W]
            y_grey_75 = torch.mean(image_75_reconstructed, 1, keepdim=True) # [B, 1, H, W]
            y_pred_img_25 = image_25_reconstructed[:, :, :, pred_pixel_offset:] # [B, 3, H, W_pred]
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
    def extract_TS_from_image(self, cur_y_grey, cur_periodicity, cur_scale_x, cur_pad_left, cur_context_len, cur_pred_len):
        # ! safe_resize要注意
        y_segmentations = safe_resize((cur_periodicity, int(round(self.image_size * cur_scale_x))), interpolation=Image.BILINEAR)(cur_y_grey)
        y_flatten = rearrange(
            y_segmentations, 
            'b 1 f p -> b (p f) 1', 
            f=cur_periodicity
        ) # flatten
        y_pred = y_flatten[:, cur_pad_left + cur_context_len: cur_pad_left + cur_context_len + cur_pred_len, :]
        
        return y_pred

    def forward(
        self,
        target: Float[torch.Tensor, "*batch 3 image_size image_size"],  # 输入，时间序列转成的图片，3维图片
        target_img: Float[torch.Tensor, "*batch 2 image_size image_size"],  # 一维是图片，另一维是mask
        x: Float[torch.Tensor, "*batch seq_len 1"],
        y: Float[torch.Tensor, "*batch seq_len 1"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        pad_left: Int[torch.Tensor, "*batch"],
        context_len: Int[torch.Tensor, "*batch"],
        pred_len: Int[torch.Tensor, "*batch"],
        periodicity: Int[torch.Tensor, "*batch"],
        scale_x: Float[torch.Tensor, "*batch"],
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
        # print(f"{y.shape = }, {target.shape = }, {target_img.shape = }")
        y = torch.clip(y, -5, 5)
        x = torch.clip(x, -5, 5)
        target = torch.clip(target, -5, 5)
        target_img = torch.clip(target_img, -5, 5)
        periodicity = periodicity.cpu().tolist()
        pad_left = pad_left.cpu().tolist()
        context_len = context_len.cpu().tolist()
        pred_len = pred_len.cpu().tolist()
        scale_x = scale_x.cpu().tolist()

        # y_logvar虽然生成的时候是按照图像大小生成的，但是他只会被展平+resize后和时间序列计算loss来监督训练。
        if not self.hparams.output_dist == 'quantile':
            mask, y_logvar, image_reconstructed, y_grey, y_pred_img = self.forward_module(self.module, target)
        else:
            mask, y_logvar, image_reconstructed, y_grey, y_pred_img, y_grey_25, y_pred_img_25, y_grey_75, y_pred_img_75 = self.forward_module(self.module, target)
        
        if self.global_step % self.log_image_step == 0:
            mask = mask.detach()
            mask = mask.unsqueeze(-1).repeat(1, 1, self.module.patch_embed.patch_size[0]**2 *3)  # (N, H*W, p*p*3)
            mask = self.module.unpatchify(mask)  # 1 is removing, 0 is keeping
            # mask = torch.einsum('nchw->nhwc', mask)
            image_input = target
            image_reconstructed_disp = image_input * (1 - mask) + image_reconstructed * mask
            green_bg = -torch.ones_like(image_reconstructed_disp) * 2
            image_input = image_input * (1 - mask) + green_bg * mask
            image_input = rearrange(image_input, '(b n) c h w -> b n h w c', b=x.shape[0])
            image_reconstructed_disp = rearrange(image_reconstructed_disp, '(b n) c h w -> b n h w c', b=x.shape[0])

        loss_list = []
        
        if not self.hparams.output_dist == 'quantile':
            monitors = {'mse': [], 'mae': [], 'std': [], 'px_loss': [], 'time_loss': [], 'diff_score_pixel': []}
        else:
            monitors = {'mse': [], 'mae': [], 'std': [], 'px_loss': [], 'time_loss': [], 'diff_score_pixel': [], 
                        'q0.25_loss': [], 'q0.50_loss': [], 'q0.75_loss': []}

        # 1. Pixel loss，也即图像loss
        # target_img_mask这里是指padding的mask
        # 也即预测窗口中的后半部分是没有值，应当抛弃掉的。
        target_img_grey = target_img[:, 0].unsqueeze(1) # [B, 1, H, W_pred]
        target_img_mask = target_img[:, 1].unsqueeze(1) # [B, 1, H, W_pred]

        # 图像loss这里用huber效果更好
        if self.hparams.pixel_loss_type == 'mse':
            pixel_loss = (y_pred_img - target_img_grey) ** 2 # [B, 3, H, W_pred]
        elif self.hparams.pixel_loss_type == 'huber':
            pixel_loss = F.huber_loss(y_pred_img, repeat(target_img_grey, 'b 1 h w -> b c h w', c=3), reduction='none') # [B, 3, H, W_pred]
        
        # ! RHO-1部分：
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


        # 去掉padding部分
        pixel_loss = (pixel_loss * target_img_mask).sum() / target_img_mask.sum()

        monitors['px_loss'].append(pixel_loss)


        # TODO: parallel
        # 2、时间序列loss，但是无法并行，需要对batch中的每个样本循环去算
        for i in range(len(y_grey)):
            cur_context_len = context_len[i]
            # ! 由于混入的真实图像数据的context_len会被设置为0，由此来区分图像or真实数据
            # ! 因此不需要对真实图像数据计算TS的loss！
            if cur_context_len == 0: # is image data; ignore!
                continue
            cur_periodicity = periodicity[i]
            cur_scale_x = scale_x[i]  # 图像和TS做resize时的比例
            cur_pad_left = pad_left[i]
            cur_pred_len = pred_len[i]
            cur_x = torch.unsqueeze(x[i, :cur_context_len], 0) # [1, Seq_len, 1]

            cur_y_true = torch.unsqueeze(y[i, :cur_pred_len], 0) # [1, Pred_len, 1]
            cur_y_grey = torch.unsqueeze(y_grey[i], 0) # [1, 1, H, W]
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
            y_pred = self.extract_TS_from_image(cur_y_grey, cur_periodicity, cur_scale_x, cur_pad_left, cur_context_len, cur_pred_len) # [2, L, 1], ([3, L, 1] if use RHO-1); first batch: mean; second batch: std
            if self.hparams.output_dist == 'quantile':
                y_pred_25 = self.extract_TS_from_image(cur_y_grey_25, cur_periodicity, cur_scale_x, cur_pad_left, cur_context_len, cur_pred_len) # [2, L, 1]
                y_pred_75 = self.extract_TS_from_image(cur_y_grey_75, cur_periodicity, cur_scale_x, cur_pad_left, cur_context_len, cur_pred_len) # [2, L, 1]
            
            # if self.hparams.loss_topk < 1.0:
            #     y_pred_ref = y_pred[2]
            #     y_pred = y_pred[:2]
            #     with torch.no_grad():
            #         diff = (y_pred[0] - cur_y_true).abs() - (y_pred_ref - cur_y_true).abs() # [1, L, 1]
            #         loss_mask = self.create_topk_mask(diff[:, :, 0]).unsqueeze(-1) # [1, L, 1]
            # else:
            # 不做rho-1了，所以这里mask全为1，相当于没加。
            loss_mask = torch.ones((1, y_pred.shape[1], 1), device=y_pred.device) # [1, L, 1]

            if self.hparams.output_dist == 'gaussian':
                delta_y = y_pred[0] - cur_y_true # [1, L, 1]
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
                    error = cur_y_true - pred
                    quantile_loss = torch.max(q * error, (q - 1) * error)  # quantile loss公式
                    monitors[f"q{q:.2f}_loss"].append(quantile_loss.mean())
                    # 对不同分位数的loss进行加权
                    cur_time_loss += w * quantile_loss
                # 另外delta_y和delta_y_square还是要记录一下，后面要记录mse和mae
                delta_y = y_pred - cur_y_true
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

            if self.global_step % self.log_image_step == 0 and i == 0:
                if self.hparams.output_dist == 'gaussian':
                    y_pred_std = y_pred_std[0, :, 0].detach()
                else:
                    y_pred_std = None
                # 250307 adds:
                if self.hparams.output_dist == 'quantile':
                    cur_y_pred_25, cur_y_pred_75 = y_pred_25[0, :, 0].detach().float(), y_pred_75[0, :, 0].detach().float()
                else:
                    cur_y_pred_25, cur_y_pred_75 = None, None
                
                self.visualization(cur_x[0, :, 0].detach().float(), cur_y_true[0, :, 0].detach().float(), y_pred[0, :, 0].detach().float(), input_image=image_input[i, 0].detach().float(), reconstructed_image=image_reconstructed_disp[i, 0].detach().float(), i=cur_periodicity, y_pred_std=y_pred_std, y_pred_25=cur_y_pred_25, y_pred_75=cur_y_pred_75)

        if len(loss_list) == 0:
            loss = 0.0
        else:
            loss = torch.stack(loss_list).mean()
        # loss = 0

        # ! 混合两个loss比例：
        # ! 目前pixel_loss_weight = 0.5最好。
        loss = loss * (1 - self.hparams.pixel_loss_weight) + pixel_loss * self.hparams.pixel_loss_weight

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

        # validate that we considered every parameter
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
                SampleOneDimension(
                    fields=("target",),
                    optional_fields=("past_feat_dynamic_real",),
                )
                + GetPatchSize(
                    min_time_patches=self.hparams.min_patches,
                    target_field="target",
                    patch_sizes=[1],
                    patch_size_constraints=FixedPatchSizeConstraints(1, 1),
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
                    imputation_method=LastValueImputation(),
                )
                + ImagifyTS(
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
                    "target_img",
                    "y",
                    "x",
                    "pad_left",
                    "context_len",
                    "pred_len",
                    "periodicity",
                    "scale_x"
                ])
            )

        return defaultdict(lambda: default_train_transform)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = super().state_dict(destination, prefix, keep_vars)
        state_dict = {k: v for k, v in state_dict.items() if "module_reference" not in k}
        return state_dict
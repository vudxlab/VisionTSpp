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

from uni2ts.model.visionts import models_mae, safe_resize
from PIL import Image
from einops import repeat, rearrange
import matplotlib.pyplot as plt

class VisionTSPretrain(L.LightningModule):
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
        # log_image_step: int = 1000,
        log_image_step: int = 200,
        load_ckpt: bool = True,
        max_pre_mask_ratio: float = 0.5,
        pre_mask_prob: float = 0.1,
        task_mode: str = "forecast",
    ):
        assert (
            num_warmup_steps <= num_training_steps
        ), f"num_warmup_steps ({num_warmup_steps}) should be <= num_training_steps ({num_training_steps})."
        super().__init__()
        self.save_hyperparameters(ignore=["module"])
        # self.module = MoiraiModule(**module_kwargs) if module is None else module
        self.module_kwargs = module_kwargs
        self.module = models_mae.mae_vit_base_patch16()
        if load_ckpt:
            checkpoint = torch.load("/home/mouxiangchen/VisionTS/ckpt/mae_visualize_vit_base.pth", map_location='cpu')
            self.module.load_state_dict(checkpoint['model'], strict=True)
        
        self.image_size = self.module.patch_embed.img_size[0]
        self.patch_size = self.module.patch_embed.patch_size[0]
        self.num_patch = self.image_size // self.patch_size
        mask = torch.ones((self.num_patch, self.num_patch))
        mask[:, :self.hparams.num_patch_input] = torch.zeros((self.num_patch, self.hparams.num_patch_input))
        self.mask_ratio = torch.mean(mask).item()
        self.log_image_step = log_image_step
        self.register_buffer("mask", mask.float().reshape((1, -1)))
        self.task_mode = task_mode


    def show_image(self, image, title=''):
        # image is [H, W, 3]
        imagenet_mean = np.array([0.485, 0.456, 0.406])
        imagenet_std = np.array([0.229, 0.224, 0.225])
        assert image.shape[2] == 3
        plt.imshow(torch.clip((image.cpu() * imagenet_std + imagenet_mean) * 255, 0, 255).int())
        plt.title(title, fontsize=16)
        plt.axis('off')

    def visualization(self, x, y, y_pred, input_image, reconstructed_image, i):
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
        plt.plot(torch.arange(len(y)) + len(x), y.cpu(), label='true', alpha=0.5)
        plt.plot(torch.arange(len(y)) + len(x), y_pred.cpu(), label='pred')
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


    def forward(
        self,
        target: Float[torch.Tensor, "*batch 3 image_size image_size"],
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
        y = torch.clip(y, -5, 5)

        _, vision_out, mask = self.module(
            target, 
            mask_ratio=self.mask_ratio, noise=repeat(self.mask, '1 l -> n l', n=target.shape[0])
        )
        image_reconstructed = self.module.unpatchify(vision_out) # [(bs x nvars) x 3 x h x w]
        

        # 5. Forecasting
        y_grey = torch.mean(image_reconstructed, 1, keepdim=True) # color image to grey

        if self.global_step % self.log_image_step == 0:
            mask = mask.detach()
            mask = mask.unsqueeze(-1).repeat(1, 1, self.module.patch_embed.patch_size[0]**2 *3)  # (N, H*W, p*p*3)
            mask = self.module.unpatchify(mask)  # 1 is removing, 0 is keeping
            # mask = torch.einsum('nchw->nhwc', mask)
            image_input = target
            image_reconstructed = image_input * (1 - mask) + image_reconstructed * mask
            green_bg = -torch.ones_like(image_reconstructed) * 2
            image_input = image_input * (1 - mask) + green_bg * mask
            image_input = rearrange(image_input, '(b n) c h w -> b n h w c', b=x.shape[0])
            image_reconstructed = rearrange(image_reconstructed, '(b n) c h w -> b n h w c', b=x.shape[0])

        # TODO: parallel
        loss_list = []
        p_to_loss = {}
        for i in range(len(y_grey)):
            cur_periodicity = periodicity[i].item()
            cur_scale_x = scale_x[i].item()
            cur_pad_left = pad_left[i].item()
            cur_context_len = context_len[i].item()
            cur_pred_len = pred_len[i].item()
            cur_y_true = torch.unsqueeze(y[i, :cur_pred_len], 0)
            cur_y_grey = torch.unsqueeze(y_grey[i], 0)
            cur_x = torch.unsqueeze(x[i, :cur_context_len], 0)

            y_segmentations = safe_resize((cur_periodicity, int(round(self.image_size * cur_scale_x))), interpolation=Image.BILINEAR)(cur_y_grey)
            y_flatten = rearrange(
                y_segmentations, 
                '1 1 f p -> 1 (p f) 1', 
                f=cur_periodicity
            ) # flatten
            y_pred = y_flatten[:, cur_pad_left + cur_context_len: cur_pad_left + cur_context_len + cur_pred_len, :]
            
            loss_list.append(torch.mean((y_pred - cur_y_true) ** 2))

            if self.global_step % self.log_image_step == 0 and i == 0:
                self.visualization(cur_x[0, :, 0].detach(), cur_y_true[0, :, 0].detach(), y_pred[0, :, 0].detach(), input_image=image_input[i, 0].detach(), reconstructed_image=image_reconstructed[i, 0].detach(), i=cur_periodicity)
            
            if cur_periodicity not in p_to_loss:
                p_to_loss[cur_periodicity] = []
            p_to_loss[cur_periodicity].append(loss_list[-1].item())
            # breakpoint()
        loss = torch.stack(loss_list).mean()

        p_to_loss_mean = {}
        for p in p_to_loss:
            p_to_loss_mean[p] = sum(p_to_loss[p]) / (len(p_to_loss[p]))
        return loss, p_to_loss_mean

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Implements LightningModule training_step. Logs training loss.

        :param batch: batched inputs
        :param batch_idx: index of current batch
        :return: training loss for current batch
        """
        loss, p_to_loss_mean = self(**batch)
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
        for p in p_to_loss_mean:
            self.log(
                f"train/loss_P={p}",
                p_to_loss_mean[p],
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
                    min_time_patches=16,
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
                    task=self.task_mode,
                    fields=("target",)
                )
                + SelectFields(
                    fields=[
                        "target",
                        "y",
                        "x",
                        "pad_left",
                        "context_len",
                        "pred_len",
                        "periodicity",
                        "scale_x",
                    ]
                    + (
                        [
                            "impute_context_start",
                            "impute_missing_start",
                            "impute_missing_length",
                        ]
                        if self.task_mode == "impute"
                        else []
                    )
                )
            )

        return defaultdict(lambda: default_train_transform)

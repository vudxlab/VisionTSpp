import os
import numpy as np
import torch
import einops
import torch.nn.functional as F
from PIL import Image
from torch import nn

from . import models_mae, util

MAE_ARCH = {
    # "mae_base": [models_mae.mae_test, "mae_visualize_vit_base.pth"],
    "mae_base": [models_mae.mae_vit_base_patch16, "mae_visualize_vit_base.pth"],
    "mae_large": [models_mae.mae_vit_large_patch16, "mae_visualize_vit_large.pth"],
    "mae_huge": [models_mae.mae_vit_huge_patch14, "mae_visualize_vit_huge.pth"]
}

MAE_DOWNLOAD_URL = "https://dl.fbaipublicfiles.com/mae/visualize/"

class VisionTS(nn.Module):

    def __init__(self, arch='mae_base', finetune_type='ln', ckpt_dir='./ckpt/', ckpt_path=None, load_ckpt=True, logvar=False, 
                 quantile=False, clip_input=0, complete_no_clip=False, color=False):
        super(VisionTS, self).__init__()

        if arch not in MAE_ARCH:
            raise ValueError(f"Unknown arch: {arch}. Should be in {list(MAE_ARCH.keys())}")

        self.vision_model = MAE_ARCH[arch][0](logvar=logvar, quantile=quantile)
        self.logvar = logvar
        self.quantile = quantile
        self.clip_input = clip_input
        self.complete_no_clip = complete_no_clip
        self.color = color

        # train_config_path = "/home/mouxiangchen/VisionTS/inpainting/big-lama/config.yaml"
        # checkpoint_path = "/home/mouxiangchen/VisionTS/inpainting/big-lama/models/best.ckpt"
        # with open(train_config_path, 'r') as f:
        #     train_config = OmegaConf.create(yaml.safe_load(f))
        # train_config.training_model.predict_only = True
        # train_config.visualizer.kind = 'noop'
        # self.inpainting_model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
        # self.inpainting_model.freeze()
        # self.inpainting_model.to('cuda:0')

        # print("sum", sum([p.numel() for _, p in self.inpainting_model.named_parameters()]))
        # breakpoint()

        if load_ckpt:
            if ckpt_path is None:
                ckpt_path = os.path.join(ckpt_dir, MAE_ARCH[arch][1])
            if not os.path.isfile(ckpt_path):
                remote_url = MAE_DOWNLOAD_URL + MAE_ARCH[arch][1]
                util.download_file(remote_url, ckpt_path)
            try:
                print(f"Load {ckpt_path}")
                checkpoint = torch.load(ckpt_path, map_location='cpu')
                # TODO
                # 1. logvar
                if not logvar:
                    for k in list(checkpoint['model'].keys()):
                        if "decoder_logvar" in k:
                            del checkpoint['model'][k]
                # 2. quantile
                if not quantile:
                    for k in list(checkpoint['model'].keys()):
                        if "decoder_pred_1" in k:
                            new_k = k.replace("decoder_pred_1", "decoder_pred")
                            checkpoint['model'][new_k] = checkpoint['model'][k]
                            del checkpoint['model'][k]
                        if "decoder_pred_2" in k:
                            del checkpoint['model'][k]
                        if "decoder_pred_3" in k:
                            del checkpoint['model'][k]
                else:
                    for k in list(checkpoint['model'].keys()):
                        # quantile的模型也把pred_1该车个pred吧～
                        if "decoder_pred_1" in k:
                            new_k = k.replace("decoder_pred_1", "decoder_pred")
                            checkpoint['model'][new_k] = checkpoint['model'][k]
                            del checkpoint['model'][k]
                # 加载模型
                self.vision_model.load_state_dict(checkpoint['model'], strict=True)
            except:
                print(f"Bad checkpoint file. Please delete {ckpt_path} and redownload!")
                raise
        
        if finetune_type != 'full':
            for n, param in self.vision_model.named_parameters():
                if 'ln' == finetune_type:
                    param.requires_grad = 'norm' in n
                elif 'bias' == finetune_type:
                    param.requires_grad = 'bias' in n
                elif 'none' == finetune_type:
                    param.requires_grad = False
                elif 'mlp' in finetune_type:
                    param.requires_grad = '.mlp.' in n
                elif 'attn' in finetune_type:
                    param.requires_grad = '.attn.' in n

    
    def update_config(self, context_len, pred_len, num_patch_input=None, periodicity=1, norm_const=0.4, align_const=0.4, interpolation='bilinear', padding_mode='replicate'):
        
        self.image_size = self.vision_model.patch_embed.img_size[0]
        self.patch_size = self.vision_model.patch_embed.patch_size[0]
        self.num_patch = self.image_size // self.patch_size

        self.context_len = context_len
        self.pred_len = pred_len
        self.periodicity = periodicity
        self.padding_mode = padding_mode
        
        if num_patch_input is not None:
            # extra padding
            extra_padding = pred_len / (self.num_patch - num_patch_input) * num_patch_input - self.context_len
            if extra_padding > 0:
                self.context_len += int(np.ceil(extra_padding))

        self.pad_left = 0
        self.pad_right = 0
        if self.context_len % self.periodicity != 0:
            self.pad_left = self.periodicity - self.context_len % self.periodicity

        if self.pred_len % self.periodicity != 0:
            self.pad_right = self.periodicity - self.pred_len % self.periodicity
        
        input_ratio = (self.pad_left + self.context_len) / (self.pad_left + self.context_len + self.pad_right + self.pred_len)
        if num_patch_input is None:
            self.num_patch_input = int(input_ratio * self.num_patch * align_const)
            if self.num_patch_input == 0:
                self.num_patch_input = 1
        else:
            self.num_patch_input = num_patch_input

        self.num_patch_output = self.num_patch - self.num_patch_input
        adjust_input_ratio = self.num_patch_input / self.num_patch

        interpolation = {
            "bilinear": Image.BILINEAR,
            "nearest": Image.NEAREST,
            "bicubic": Image.BICUBIC,
        }[interpolation]

        self.input_resize = util.safe_resize((self.image_size, int(self.image_size * adjust_input_ratio)), interpolation=interpolation)
        self.scale_x = ((self.pad_left + self.context_len) // self.periodicity) / (int(self.image_size * adjust_input_ratio))
        self.output_resize = util.safe_resize((self.periodicity, int(round(self.image_size * self.scale_x))), interpolation=interpolation)
        self.norm_const = norm_const
        
        mask = torch.ones((self.num_patch, self.num_patch)).to(self.vision_model.cls_token.device)
        mask[:, :self.num_patch_input] = torch.zeros((self.num_patch, self.num_patch_input))
        self.register_buffer("mask", mask.float().reshape((1, -1)))
        self.mask_ratio = torch.mean(mask).item()
    

    def forward(self, x, export_image=False, fp64=False, multivariate=False):
        # Forecasting using visual model.
        # x: look-back window, size: [bs x context_len x nvars]
        # fp64=True can avoid math overflow in some benchmark, like Bitcoin.
        # return: forecasting window, size: [bs x pred_len x nvars]

        # 1. Normalization
        means = x.mean(1, keepdim=True).detach()  # x: [bs x seq_len x nvars], means: [bs x 1 x nvars]
        x_enc = x - means
        stdev = torch.sqrt(
            torch.var(x_enc.to(torch.float64) if fp64 else x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5) # [bs x 1 x nvars]
        # ! norm_const: 一般设置为0.4，用于约束增大标准差stdev，使得norm之后的值的范围更小！
        stdev /= self.norm_const
        x_enc /= stdev
        # Channel Independent
        x_enc = einops.rearrange(x_enc, 'b s n -> b n s') # [bs x nvars x seq_len]

        if x_enc.shape[-1] < self.context_len:
            extra_padding = self.context_len - x_enc.shape[-1]
        else:
            extra_padding = 0

        # 2. Segmentation
        x_pad = F.pad(x_enc, (self.pad_left + extra_padding, 0), mode=self.padding_mode) # [b n s]
        
        x_2d = einops.rearrange(x_pad, 'b n (p f) -> (b n) 1 f p', f=self.periodicity)

        # 3. Render & Alignment
        x_resize = self.input_resize(x_2d)
        masked = torch.zeros((x_2d.shape[0], 1, self.image_size, self.num_patch_output * self.patch_size), device=x_2d.device, dtype=x_2d.dtype)
        x_concat_with_masked = torch.cat([
            x_resize, 
            masked
        ], dim=-1)
        image_input = einops.repeat(x_concat_with_masked, 'b 1 h w -> b c h w', c=3)
        
        # print(f"{image_input.shape = }")
        # print(f"{image_input = }")
        if self.clip_input == 0:
            if self.complete_no_clip:
                print(f"image_input > 5: {torch.any((image_input > 5))}, image_input < -5: {torch.any((image_input < -5))}")
                # pass
            else:
                print(f"image_input > 5: {torch.any((image_input > 5))}, image_input < -5: {torch.any((image_input < -5))}")
                image_input = torch.clip(image_input, -5, 5)
        else:  # self.clip_input == 1 or self.clip_input == 2
            # ImageNet的mean和std
            image_mean = [0.485,0.456,0.406]
            image_std = [0.229,0.224,0.225]
            # 计算 [-mean/std, (1-mean)/std]，得到的结果如下
            thres_down_list = [-2.1179039301310043, -2.0357142857142856, -1.8044444444444445]
            thres_up_list = [2.2489082969432315, 2.428571428571429, 2.6399999999999997]
            thres_down = -1.8044
            thres_up = 2.2489
            
            print(f"image_input > {thres_up}: {torch.any((image_input > thres_up))}, image_input < {thres_down}: {torch.any((image_input < thres_down))}")
            
            image_input = torch.clip(image_input, thres_down, thres_up)
            

        # 4. Reconstruction
        _, y, mask = self.vision_model(
            image_input, 
            mask_ratio=self.mask_ratio, noise=einops.repeat(self.mask, '1 l -> n l', n=image_input.shape[0])
        )
        if self.logvar:
            y, y_logvar = y
            image_reconstructed_logvar = self.vision_model.unpatchify(y_logvar, n_channels=1) # [(bs x nvars) x 1 x h x w]
        if self.quantile:
            y, y_25, y_75 = y
            image_reconstructed_25 = self.vision_model.unpatchify(y_25)
            image_reconstructed_75 = self.vision_model.unpatchify(y_75)
        
        image_reconstructed = self.vision_model.unpatchify(y) # [(bs x nvars) x 3 x h x w]
        # ======================================================
        # imagenet_mean = torch.Tensor([0.485, 0.456, 0.406]).to(image_input).view((-1, 1, 1))
        # imagenet_std = torch.Tensor([0.229, 0.224, 0.225]).to(image_input).view((-1, 1, 1))
        # inpainting_input = torch.clip((image_input * imagenet_std + imagenet_mean), 0, 1)
        # inpainting_mask = self.vision_model.unpatchify(self.mask.unsqueeze(-1).repeat(1, 1, self.vision_model.patch_embed.patch_size[0]**2 *3)).mean(dim=1).to(torch.int64)
        # inpainting_mask = einops.repeat(inpainting_mask, '1 h w -> b 1 h w', b=inpainting_input.shape[0])
        # batch = {'image': inpainting_input, 'mask': inpainting_mask}
        # out_batch = self.inpainting_model(batch)
        # inpainting_output = torch.clip(out_batch['inpainted'], 0, 1)
        # image_reconstructed = (inpainting_output - imagenet_mean) / imagenet_std

        # if not hasattr(self, 'count'):
        #     self.count = 0
        # data = []
        # dataset = 'ETTh1'
        # for b in range(x_pad.shape[0]):
        #     for c in range(x_pad.shape[1]):
        #         image_read = np.array(Image.open(f"../inpainting/output/{dataset}_{self.pred_len}_{self.count}_{b}_{c}_mask001.png"))
        #         data.append(image_read)
        # self.count += 1
        # data = torch.Tensor(np.array(data).astype(np.float32)).to(image_input) / 255
        # data = (data - imagenet_mean) / imagenet_std
        # image_reconstructed = einops.rearrange(data, 'b h w c-> b c h w')
        # ======================================================
        
        # 5. Forecasting
        y_before_resize = torch.mean(image_reconstructed, 1, keepdim=True) # color image to grey, [B, 1, H, W]
        if self.logvar:
            y_before_resize = torch.cat([y_before_resize, image_reconstructed_logvar], dim=1) # [B, 2, H, W]
        if self.quantile:
            y_before_resize = torch.cat([y_before_resize, image_reconstructed_25, image_reconstructed_75], dim=1) # [B, 3, H, W]
        
        y_segmentations = self.output_resize(y_before_resize) # resize back
        y_flatten = einops.rearrange(
            y_segmentations, 
            '(b n) c f p -> b c (p f) n', 
            b=x_enc.shape[0], f=self.periodicity
        ) # flatten
        y = y_flatten[:, :, self.pad_left + self.context_len: self.pad_left + self.context_len + self.pred_len, :] # extract the forecasting window

        if self.logvar:
            y, y_logvar = y[:, 0], y[:, 1] # [B, L, C]
            y_std = torch.exp(y_logvar / 2)
        if self.quantile:
            y, y_25, y_75 = y[:, 0], y[:, 1], y[:, 2] # [B, L, C]
        else:
            y = y[:, 0]

        # 6. Denormalization
        y = y * (stdev.repeat(1, self.pred_len, 1))
        y = y + (means.repeat(1, self.pred_len, 1))
        if self.logvar:
            y_std = y_std * (stdev.repeat(1, self.pred_len, 1))
            y_std = y_std + (means.repeat(1, self.pred_len, 1))
            y = [y, y_std]
        if self.quantile:
            y_25 = y_25 * (stdev.repeat(1, self.pred_len, 1))
            y_25 = y_25 + (means.repeat(1, self.pred_len, 1))
            y_75 = y_75 * (stdev.repeat(1, self.pred_len, 1))
            y_75 = y_75 + (means.repeat(1, self.pred_len, 1))
            y = [y, y_25, y_75]


        if export_image:
            mask = mask.detach()
            mask = mask.unsqueeze(-1).repeat(1, 1, self.vision_model.patch_embed.patch_size[0]**2 *3)  # (N, H*W, p*p*3)
            mask = self.vision_model.unpatchify(mask)  # 1 is removing, 0 is keeping
            # mask = torch.einsum('nchw->nhwc', mask)
            image_reconstructed = image_input * (1 - mask) + image_reconstructed * mask
            green_bg = -torch.ones_like(image_reconstructed) * 2
            image_input = image_input * (1 - mask) + green_bg * mask
            image_input = einops.rearrange(image_input, '(b n) c h w -> b n h w c', b=x_enc.shape[0])
            
            image_reconstructed = einops.rearrange(image_reconstructed, '(b n) c h w -> b n h w c', b=x_enc.shape[0])
            return y, image_input, image_reconstructed
        # # ======================================================
        # imagenet_mean = torch.Tensor([0.485, 0.456, 0.406]).to(image_input)
        # imagenet_std = torch.Tensor([0.229, 0.224, 0.225]).to(image_input)
        # mask = einops.rearrange(mask, '(b n) c h w -> b n h w c', b=x_enc.shape[0])
        # image = torch.clip((image_input * imagenet_std + imagenet_mean) * 255, 0, 255).int()
        # mask = torch.clip(mask * 255, 0, 255).int()
        # dataset = 'ETTh1'
        # os.makedirs(f"../inpainting/input/{dataset}_{self.pred_len}/", exist_ok=True)
        # if not hasattr(self, 'count'):
        #     self.count = 0
        # for b in range(mask.shape[0]):
        #     for c in range(mask.shape[1]):
        #         Image.fromarray(mask[b, c][..., 0].cpu().numpy().astype(np.uint8), mode='L').save(f"../inpainting/input/{dataset}_{self.pred_len}/{self.count}_{b}_{c}_mask001.png")
        #         Image.fromarray(image[b, c].cpu().numpy().astype(np.uint8), mode='RGB').save(f"../inpainting/input/{dataset}_{self.pred_len}/{self.count}_{b}_{c}.png")
        # self.count += 1
        # ======================================================

        return y

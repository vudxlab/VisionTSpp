import matplotlib.pyplot as plt
import torch
import numpy as np
import einops

from visionts import VisionTS


ARCH = 'mae_base' # choose from {'mae_base', 'mae_large', 'mae_huge'}. We recommend 'mae_base'
DEVICE = 'cuda:0'

ckpt_dir = '/home/mouxiangchen/uni2ts/outputs/pretrain/visionts/lotsa_v1_weighted/weighted_7_quantile_512_multi_color_9_heads/checkpoints'
# ckpt_path = f'{ckpt_dir}/epoch=99-step=10000.ckpt'
ckpt_path = f'{ckpt_dir}/processed_epoch=99-step=10000.ckpt'
    
model = VisionTS(ARCH, ckpt_path=ckpt_path, quantile=True, 
                    clip_input=True, complete_no_clip=False, color=True).to(DEVICE)

# # The following code will automatically download MAE (base) checkpoint file into ./ckpt/ if not exists.
# model = VisionTS(ARCH, ckpt_dir='./ckpt/').to(DEVICE)


print(model)

# # Test on a single image


imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

def show_image(image, title=''):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    plt.imshow(torch.clip((image.cpu() * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    plt.title(title, fontsize=16)
    plt.axis('off')

def run(x, y, align_const, norm_const, periodicity):
    # The input of VisionTS is a Tensor with shape [n_batch, context_len, n_vars]
    x = torch.Tensor(einops.rearrange(x, 't -> 1 t 1')).to(DEVICE)
    # The output of VisionTS is a Tensor with shape [n_batch, pred_len, n_vars]
    y = torch.Tensor(einops.rearrange(y, 't -> 1 t 1')).to(DEVICE)
    # Before calling forward, make sure you use update_config() to update hyperparameters, context length or prediction length.
    model.update_config(x.shape[1], y.shape[1], align_const=align_const, norm_const=norm_const, periodicity=periodicity)
    # Forecasting time series using forward()
    with torch.no_grad():
        y_pred, input_image, reconstructed_image = model.forward(x, export_image=True)

    # Visualization
    x = x[:, -300:, :]
    plt.subplot(2, 2, 1)
    show_image(input_image[0, 0], 'input')

    plt.subplot(2, 2, 2)
    plt.plot(x.cpu()[0, :, 0])
    plt.plot(torch.arange(y.shape[1]) + x.shape[1], y.cpu()[0, :, 0], label='true', alpha=0.5)
    plt.title('input time series')
    plt.legend()
    plt.tight_layout()

    plt.subplot(2, 2, 3)
    show_image(reconstructed_image[0, 0], 'reconstructed')

    plt.subplot(2, 2, 4)
    plt.plot(x.cpu()[0, :, 0])
    plt.plot(torch.arange(y.shape[1]) + x.shape[1], y.cpu()[0, :, 0], label='true', alpha=0.5)
    plt.plot(torch.arange(y.shape[1]) + x.shape[1], y_pred.cpu()[0, :, 0], label='pred')
    plt.title('forecasting')
    plt.legend()
    plt.tight_layout()


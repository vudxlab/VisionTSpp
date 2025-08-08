
import sys
sys.path.append("../")

from torch import nn
from visionts import VisionTS

class Model(nn.Module):

    def __init__(self, config):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.task_name = config.task_name
        self.pred_len = config.pred_len
        self.seq_len = config.seq_len
        
        self.quantile_int = config.quantile
        self.quantile = True if self.quantile_int > 0 else False
        
        self.clip_input = config.clip_input
        self.complete_no_clip = config.complete_no_clip
        self.complete_no_clip = True if self.complete_no_clip > 0 else False
        # self.clip_input = True if self.clip_input > 0 else False
        
        self.color = config.color
        self.color = True if self.color > 0 else False

        # self.vm = VisionTS(arch=config.vm_arch, finetune_type=config.ft_type, load_ckpt=config.vm_pretrained == 1, ckpt_dir=config.vm_ckpt)
        self.vm = VisionTS(arch=config.vm_arch, finetune_type=config.ft_type, 
                           load_ckpt=config.vm_pretrained == 1, ckpt_dir=config.checkpoint_dir, ckpt_path=config.checkpoint_path,
                           quantile=self.quantile, clip_input=self.clip_input, complete_no_clip=self.complete_no_clip,
                           color=self.color)

        self.vm.update_config(context_len=config.seq_len, pred_len=config.pred_len, periodicity=config.periodicity, interpolation=config.interpolation, norm_const=config.norm_const, align_const=config.align_const)


    def forecast(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, visual=False, LOOKBACK_LEN_VISUAL=300):
        # 运行forward
        return self.vm.forward(x_enc, export_image=visual, LOOKBACK_LEN_VISUAL=LOOKBACK_LEN_VISUAL)


    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        raise NotImplementedError()


    def anomaly_detection(self, x_enc):
        raise NotImplementedError()

    def classification(self, x_enc, x_mark_enc):
        raise NotImplementedError()


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, visual=False, LOOKBACK_LEN_VISUAL=300):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            if self.quantile:
                # # original:
                # dec_out, dec_out_25, dec_out_75 = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
                # return dec_out[:, -self.pred_len:, :], dec_out_25[:, -self.pred_len:, :], dec_out_75[:, -self.pred_len:, :]  # [B, L, D]
            
                # 9 heads:
                if not visual:
                    dec_out, dec_out_quantile_list = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
                else:
                    [dec_out, dec_out_quantile_list], image_input, image_reconstructed, nvars, color_list = \
                        self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, visual=visual, LOOKBACK_LEN_VISUAL=LOOKBACK_LEN_VISUAL)
                
                dec_out = dec_out[:, -self.pred_len:, :]  # [B, pred_len, D]
                for i in range(len(dec_out_quantile_list)):
                    dec_out_quantile_list[i] = dec_out_quantile_list[i][:, -self.pred_len:, :]  # [B, pred_len, D]
                
                if not visual:
                    return dec_out, dec_out_quantile_list  # [B, pred_len, D]
                else:
                    return [dec_out, dec_out_quantile_list], image_input, image_reconstructed, nvars, color_list  # [B, pred_len, D]
            else:
                dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
                return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None

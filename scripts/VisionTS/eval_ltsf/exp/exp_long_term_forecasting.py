from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import json
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(vali_loader, desc='vali')):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.quantile == 1:
                        outputs, outputs_2, outputs_75 = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.save_dir, self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            pbar = tqdm(train_loader)
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pbar):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.quantile == 1:
                        # outputs, outputs_25, outputs_75 = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        outputs, outputs_quantile_list = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                pbar.set_description("epoch: {0} | loss: {1:.7f}".format(epoch + 1, loss.item()))
                if (i + 1) % 100 == 0:
                    tqdm.write("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    tqdm.write('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        if os.path.isfile(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path))
        else:
            print("Test without train!",best_model_path)

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(f'{self.args.save_dir}/checkpoints/' + setting, 'checkpoint.pth')))

        valid_loss_path = os.path.join(f'{self.args.save_dir}/checkpoints/' + setting, 'valid_loss.json')
        if os.path.isfile(valid_loss_path):
            with open(valid_loss_path) as f:
                valid_loss = json.load(f)
                best_valid_loss = valid_loss['best_valid_loss']
                best_valid_epoch = valid_loss['best_valid_epoch']
        else:
            best_valid_loss = -1
            best_valid_epoch = -1
        
        # ! 250429 adds: 和monash以及pf一样，这里需要对model做一次update_config！！！
        # 更新模型设置？
        self.model.vm.update_config(context_len=self.args.seq_len, pred_len=self.args.pred_len, periodicity=self.args.periodicity, 
                            num_patch_input=self.args.num_patch_input, padding_mode='constant')
        # ! PS：其实在models/VisionTS里面做了update_config，这里其实不做也ok的～
        
        # 250609 adds: 画图用
        LOOKBACK_LEN_VISUAL = self.args.LOOKBACK_LEN_VISUAL
        LOOKBACK_LEN_VISUAL = min(self.args.seq_len, LOOKBACK_LEN_VISUAL)

        preds = []
        trues = []
        # folder_path = f'{self.args.save_dir}/test_results/' + setting + '/'
        folder_path = f'{self.args.save_dir}/'
        if self.args.visual and not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(test_loader, desc='test')):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.quantile == 1:
                        # outputs, outputs_25, outputs_75 = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        if not self.args.visual:
                            outputs, outputs_quantile_list = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        else:
                            # print(self.model.forward.__code__.co_varnames[:self.model.forward.__code__.co_argcount])
                            [outputs, outputs_quantile_list], image_input, image_reconstructed, nvars, color_list = \
                                self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, visual=self.args.visual, LOOKBACK_LEN_VISUAL=LOOKBACK_LEN_VISUAL)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)
        
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                
                
                # 250608 adds: visualization
                FIG_WIDTH = 17.5
                FIG_HEIGHT_PER_VAR = 1.5
                FONT_S = 13
                FONT_L = 15
                IM_ALPHA = 0.8
                if self.args.visual and i % 5 == 0:
                    cur_input = batch_x.detach().cpu().numpy()
                    cur_gt = np.concatenate((cur_input[0, :, :], true[0, :, :]), axis=0)
                    cur_pd = np.concatenate((cur_input[0, :, :], pred[0, :, :]), axis=0)
                    # visual(cur_gt, cur_pd, os.path.join(folder_path, str(i) + '.pdf'))
                    
                    def visual_ts(true, preds=None, name='./pic/test.pdf'):
                        nvars = true.shape[1]
                        
                        # 转换为 numpy array
                        if not isinstance(true, np.ndarray):
                            true = true.cpu().numpy()
                        if preds is not None and not isinstance(preds, np.ndarray):
                            preds = preds.cpu().numpy()
                        
                        # plt.figure(figsize=(10, 3))
                        # 创建 7 行 1 列的子图，设置高度比例
                        fig, axes = plt.subplots(nrows=nvars, ncols=1, figsize=(FIG_WIDTH, nvars * FIG_HEIGHT_PER_VAR), sharex=True,
                                                 gridspec_kw={'height_ratios': [1] * nvars})
                        # 移除子图间的垂直间距
                        plt.subplots_adjust(hspace=0)
                        
                        # 截取可视化区间
                        # print(f"{true.shape = }, {preds.shape = }")
                        true = true[-LOOKBACK_LEN_VISUAL - self.args.pred_len:]
                        if preds is not None:
                            preds = preds[-LOOKBACK_LEN_VISUAL - self.args.pred_len:]
                        
                        # 遍历每个变量
                        for i, ax in enumerate(axes):
                            # 绘制真实值
                            ax.plot(true[:, i], label='GroundTruth', color='gray', linewidth=1.5)
                            
                            # 绘制预测值（如果存在）
                            if preds is not None:
                                # ax.plot(preds[:, i], label='Prediction', color='blue', linewidth=1.5)
                                # ! 只对[LOOKBACK_LEN_VISUAL, len(true)]这段区间内的prediction做visual
                                ax.plot(np.arange(LOOKBACK_LEN_VISUAL, len(true)), preds[LOOKBACK_LEN_VISUAL:, i], label='Prediction', color='blue', linewidth=1.5)
                            
                            # 添加分界虚线
                            y_min, y_max = ax.get_ylim()
                            ax.vlines(x=LOOKBACK_LEN_VISUAL, 
                                    ymin=y_min, 
                                    ymax=y_max, 
                                    linewidth=1, 
                                    linestyles='dashed', 
                                    colors='gray')
                            
                            # 去掉刻度线和刻度标签
                            ax.tick_params(axis='x', which='both', length=0)
                            ax.tick_params(axis='y', which='both', length=0)
                            ax.set_xticks([])
                            ax.set_yticks([])
                            
                            # 可选：添加变量索引标签
                            ax.text(0.007, 0.8, f'Var {i+1}', transform=ax.transAxes, fontsize=FONT_S)
                        
                        # 添加整体图例（放在顶部）
                        if preds is not None:
                            handles, labels = axes[0].get_legend_handles_labels()
                            fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.9, 0.935), prop={'size': FONT_S-2})
                        
                        # 计算 MSE & MAE 并添加为标题
                        if preds is not None:
                            preds_calc = preds[-self.args.pred_len:]
                            true_calc = true[-self.args.pred_len:]
                            mse = np.mean((preds_calc - true_calc) ** 2)
                            mae = np.mean(np.abs(preds_calc - true_calc))
                            fig.suptitle(f'MSE: {mse:.3f}, MAE: {mae:.3f}', fontsize=FONT_L, y=0.915)
                        
                        # 保存图像
                        plt.savefig(name, bbox_inches='tight', dpi=300)
                        plt.show()
                        
                        # print(f"{true.shape = }, {preds.shape = }")
                        # true = true[-LOOKBACK_LEN_VISUAL-self.args.pred_len:]
                        # preds = preds[-LOOKBACK_LEN_VISUAL-self.args.pred_len:] if preds is not None else None
                        
                        
                        # for nvar in range(nvars):
                        #     if preds is not None:
                        #         plt.plot(preds, label='Prediction', color='blue', linewidth=1.5)
                        #     plt.plot(true, label='GroundTruth', color='gray', linewidth=1.5)
                        #     # if preds is not None:
                        #     #     plt.plot(preds, label='Prediction', linewidth=2)
                            
                        #     # 画一条垂直的虚线：要求线的高度和画面高度一致，所在的x的位置是look-back和prediction window分界线
                        #     # vline_pos = min(self.args.seq_len, LOOKBACK_LEN_VISUAL)
                        #     plt.vlines(x=LOOKBACK_LEN_VISUAL, 
                        #             ymin=min(true.min(), preds.min() if preds is not None else true.min()), 
                        #             ymax=max(true.max(), preds.max() if preds is not None else true.max()), 
                        #             linewidth=1, linestyles='dashed', colors='gray')
                            
                        #     plt.legend()
                        
                        # # plt.tight_layout()
                        # plt.savefig(name, bbox_inches='tight')
                        # plt.show()
                    
                    # 再保存一下！！
                    def visual_image(image, title='', cur_nvars=1, cur_color_list=None, save_path='image.pdf'):
                        # image is [H, W, 3]
                        imagenet_mean = np.array([0.485, 0.456, 0.406])
                        imagenet_std = np.array([0.229, 0.224, 0.225])
                        assert image.shape[2] == 3
                        
                        # ! 20250430 adds: 根据color_list来处理图片的颜色通道
                        # channel_masks = (image != 0).any(dim=(0, 1))
                        cur_image = torch.zeros_like(image)
                        cur_image = cur_image.cpu()
                        
                        height_per_var = image.shape[0] // cur_nvars
                        print(f"{height_per_var = }, {image.shape = }, {cur_nvars = }")
                        for i in range(cur_nvars):
                            cur_color = cur_color_list[i]
                            cur_image[i*height_per_var:(i+1)*height_per_var, :, cur_color] = \
                                (image[i*height_per_var:(i+1)*height_per_var, :, cur_color].cpu() * imagenet_std[cur_color] + imagenet_mean[cur_color]) * 255
                        cur_image = torch.clip(cur_image, 0, 255).int()
                        
                        plt.figure(figsize=(3, 3))
                        plt.imshow(cur_image, alpha=IM_ALPHA)
                        plt.title(title, fontsize=16)
                        plt.axis('off')
                        plt.tight_layout()
                        plt.savefig(save_path)
                        plt.show()
                    
                    
                    print(f"{nvars = }, {color_list = }")
                    print(f"{cur_gt.shape = }, {cur_pd.shape = }")
                    print(f"{image_input.shape = }, {image_reconstructed.shape = }")
                    visual_ts(cur_gt, cur_pd, os.path.join(folder_path, str(i) + '_ts.pdf'))
                    visual_image(image_input[0, 0], 'input', nvars, color_list, save_path=os.path.join(folder_path, str(i) + '_input_img.pdf'))
                    visual_image(image_reconstructed[0, 0], 'reconstructed', nvars, color_list, save_path=os.path.join(folder_path, str(i) + '_recon_img.pdf'))
        

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # # result save
        # folder_path = f'{self.args.save_dir}/results/' + setting + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()
        
        if not self.args.visual:
            # 这里额外记录一下MSE和MAE到ckpt目录下的文件中
            save_path = self.args.save_name
            cur_dataset = self.args.data_path.replace('.csv', '')
            # 先生成当前的metrics_df:
            metrics_df = pd.DataFrame({
                "MSE": [mse],
                "MAE": [mae],
            })
            metrics_df.insert(loc=0, column="dataset", value=[cur_dataset])  # 插入dataset列
            
            if os.path.exists(save_path):
                old_metrics_df = pd.read_csv(save_path)
                metrics_df = pd.concat([old_metrics_df, metrics_df], ignore_index=True)
            print(metrics_df)
            metrics_df.to_csv(save_path, index=False)
            print(f"Results saved to {save_path}")
            print("-" * 5, f"Evaluation of {cur_dataset} complete", "-" * 5)

            # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe, best_valid_loss, best_valid_epoch]))
            # np.save(folder_path + 'pred.npy', preds)
            # np.save(folder_path + 'true.npy', trues)

        return

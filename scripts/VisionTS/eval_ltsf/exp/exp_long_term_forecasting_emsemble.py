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

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast_emsemble(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast_emsemble, self).__init__(args)

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
                        outputs, outputs_25, outputs_75 = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
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
        

        preds = []
        trues = []
        preds_emsemble = []
        # folder_path = f'{self.args.save_dir}/test_results/' + setting + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)

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
                        outputs, outputs_25, outputs_75 = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs_backup = outputs
                
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
                
                if self.args.quantile == 1:
                    outputs_25 = outputs_25[:, -self.args.pred_len:, :]
                    outputs_75 = outputs_75[:, -self.args.pred_len:, :]
                    
                    stacked_outputs = torch.stack([outputs_backup, outputs_25, outputs_75])  # 堆叠张量，增加一个新维度
                    mean_outputs = torch.mean(stacked_outputs, dim=0)  # 沿新维度（dim=0）计算平均值
                    
                    pred_emsemble = mean_outputs.detach().cpu().numpy()
                    
                    if test_data.scale and self.args.inverse:
                        shape = pred_emsemble.shape
                        pred_emsemble = test_data.inverse_transform(pred_emsemble.squeeze(0)).reshape(shape)
                    
                    pred_emsemble = pred_emsemble[:, :, f_dim:]
                    preds_emsemble.append(pred_emsemble)   

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)
        
        if self.args.quantile == 1:
            preds_emsemble = np.concatenate(preds_emsemble, axis=0)
            preds_emsemble = preds_emsemble.reshape(-1, preds_emsemble.shape[-2], preds_emsemble.shape[-1])

        # # result save
        # folder_path = f'{self.args.save_dir}/results/' + setting + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        mae_emsemble, mse_emsemble, *_ = metric(preds_emsemble, trues)
        print('mse_emsemble:{}, mae_emsemble:{}'.format(mse_emsemble, mae_emsemble))
        
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('mse_emsemble:{}, mae_emsemble:{}'.format(mse_emsemble, mae_emsemble))
        f.write('\n')
        f.write('\n')
        f.close()
        
        # 这里额外记录一下MSE和MAE到ckpt目录下的文件中
        save_path = self.args.save_name
        cur_dataset = self.args.data_path.replace('.csv', '')
        # 先生成当前的metrics_df:
        metrics_df = pd.DataFrame({
            # "MSE": [mse],
            # "MAE": [mae],
            "MSE": [mse_emsemble],
            "MAE": [mae_emsemble],
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

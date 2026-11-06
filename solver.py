import os
import os.path as osp
import csv
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim

from prep import printProgressBar
from networks import RED_CNN
from measure import compute_measure


class Solver(object):
    ####### CODE ADD: Added val_data_loader param ######
    def __init__(self, args, data_loader, val_data_loader = None):
        self.mode = args.mode
        self.load_mode = args.load_mode
        self.data_loader = data_loader
        ####### CODE ADD ######
        self.val_data_loader = val_data_loader
        ####### CODE ADD ######


        if args.device:
            self.device = torch.device(args.device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.norm_range_min = args.norm_range_min
        self.norm_range_max = args.norm_range_max
        self.trunc_min = args.trunc_min
        self.trunc_max = args.trunc_max

        self.save_path = args.save_path
        self.multi_gpu = args.multi_gpu

        ####### CODE ADD: CSV logging ######
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(osp.join(self.save_path, "save_logs"), exist_ok=True)
        os.makedirs(osp.join(self.save_path, "save_models"), exist_ok=True)
        self.train_log_file = osp.join(self.save_path, 'save_logs', 'loss_log.csv')
        self.validation_log_file = osp.join(self.save_path, 'save_logs', 'validation_log.csv')
        self._initialize_csv(self.train_log_file, ['iteration', 'train_loss'])
        self._initialize_csv(self.validation_log_file, ['iteration', 'rmse', 'psnr', 'ssim'])
        ####### CODE ADD ######

        self.num_epochs = args.num_epochs
        self.print_iters = args.print_iters
        self.decay_iters = args.decay_iters
        self.save_iters = args.save_iters
        self.test_iters = args.test_iters
        self.result_fig = args.result_fig

        self.patch_size = args.patch_size

        self.REDCNN = RED_CNN()
        if (self.multi_gpu) and (torch.cuda.device_count() > 1):
            print('Use {} GPUs'.format(torch.cuda.device_count()))
            self.REDCNN = nn.DataParallel(self.REDCNN)
        self.REDCNN.to(self.device)

        self.lr = args.lr
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.REDCNN.parameters(), self.lr)


    def save_model(self, iter_):
        f = os.path.join(self.save_path, 'save_models', 'REDCNN_{}iter.ckpt'.format(iter_))
        torch.save(self.REDCNN.state_dict(), f)


    def load_model(self, iter_):
        f = os.path.join(self.save_path, 'REDCNN_{}iter.ckpt'.format(iter_))
        if self.multi_gpu:
            state_d = OrderedDict()
            for k, v in torch.load(f):
                n = k[7:]
                state_d[n] = v
            self.REDCNN.load_state_dict(state_d)
        else:
            self.REDCNN.load_state_dict(torch.load(f))


    def lr_decay(self):
        lr = self.lr * 0.5
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    ###### CODE ADD ######
    def normalize_(self, image, MIN_B=-1024.0, MAX_B=3072.0):
       image = (image - MIN_B) / (MAX_B - MIN_B)
       return image

    def unnormalize_(self, image, MIN_B=-1024.0, MAX_B=3072.0):
       image = image * (MAX_B - MIN_B) + MIN_B
       return image
    ###### CODE ADD ######
    
    def denormalize_(self, image):
        image = image * (self.norm_range_max - self.norm_range_min) + self.norm_range_min
        return image


    def trunc(self, mat):
        mat[mat <= self.trunc_min] = self.trunc_min
        mat[mat >= self.trunc_max] = self.trunc_max
        return mat


    def save_fig(self, x, y, pred, fig_name, original_result, pred_result):
        x, y, pred = x.numpy(), y.numpy(), pred.numpy()
        f, ax = plt.subplots(1, 3, figsize=(30, 10))
        ax[0].imshow(x, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[0].set_title('Quarter-dose', fontsize=30)
        ax[0].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(original_result[0],
                                                                           original_result[1],
                                                                           original_result[2]), fontsize=20)
        ax[1].imshow(pred, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[1].set_title('Result', fontsize=30)
        ax[1].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(pred_result[0],
                                                                           pred_result[1],
                                                                           pred_result[2]), fontsize=20)
        ax[2].imshow(y, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[2].set_title('Full-dose', fontsize=30)

        f.savefig(os.path.join(self.save_path, 'fig', 'result_{}.png'.format(fig_name)))
        plt.close()


    def train(self):
        train_losses = []
        total_iters = 0
        start_time = time.time()
        for epoch in range(1, self.num_epochs):
            self.REDCNN.train(True)

            for iter_, (x, y) in enumerate(self.data_loader):
                total_iters += 1

                # add 1 channel
                x = x.unsqueeze(0).float().to(self.device)
                y = y.unsqueeze(0).float().to(self.device)

                if self.patch_size: # patch training
                    x = x.view(-1, 1, self.patch_size, self.patch_size)
                    y = y.view(-1, 1, self.patch_size, self.patch_size)

                pred = self.REDCNN(x)
                loss = self.criterion(pred, y)
                self.REDCNN.zero_grad()
                self.optimizer.zero_grad()

                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())

                # print
                if total_iters % self.print_iters == 0:
                    print("STEP [{}], EPOCH [{}/{}], ITER [{}/{}] \nLOSS: {:.8f}, TIME: {:.1f}s".format(total_iters, epoch, 
                                                                                                        self.num_epochs, iter_+1, 
                                                                                                        len(self.data_loader), loss.item(), 
                                                                                                        time.time() - start_time))
                # learning rate decay
                if total_iters % self.decay_iters == 0:
                    self.lr_decay()
                # save model
                if total_iters % self.save_iters == 0:
                    self.save_model(total_iters)
                    np.save(os.path.join(self.save_path, 'loss_{}_iter.npy'.format(total_iters)), np.array(train_losses))


    #######  CODE ADD #######
    def _initialize_csv(self, file_path, headers):
        if not osp.exists(file_path):
            with open(file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(headers)

    def log_train_loss(self, iteration, train_loss):
        with open(self.train_log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([iteration, train_loss])

    def log_test_metrics(self, iteration, rmse, psnr, ssim):
        with open(self.validation_log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([iteration, rmse, psnr, ssim])

    def validate_2detect(self):
        """Run the model on the whole val_data_loader and return the average
        (PSNR, SSIM, RMSE) of the predictions against the targets.

        Uses the same per-image min-max normalization as train_with_2detect.
        Metrics are computed after mapping the prediction back to the original
        intensity scale, with data_range = (max - min) of the input image.
        """
        self.REDCNN.eval()
        psnr_sum, ssim_sum, rmse_sum, n_images = 0.0, 0.0, 0.0, 0

        with torch.no_grad():
            for target, input in self.val_data_loader:
                input = torch.as_tensor(input).float().to(self.device)
                target = torch.as_tensor(target).float().to(self.device)

                # (B, H, W) / (1, B, H, W) / ... -> (N, 1, H, W): full images, no patches
                h, w = input.shape[-2], input.shape[-1]
                input = input.reshape(-1, 1, h, w)
                target = target.reshape(-1, 1, h, w)

                for k in range(input.shape[0]):
                    inp_k, tgt_k = input[k:k+1], target[k:k+1]

                    MIN_INPUT_VAL = inp_k.min()
                    MAX_INPUT_VAL = inp_k.max()
                    pred_k = self.REDCNN(self.normalize_(inp_k, MIN_INPUT_VAL, MAX_INPUT_VAL))
                    pred_k = self.unnormalize_(pred_k, MIN_INPUT_VAL, MAX_INPUT_VAL)

                    x = inp_k.view(h, w).cpu()
                    y = tgt_k.view(h, w).cpu()
                    p = pred_k.view(h, w).cpu()
                    data_range = (MAX_INPUT_VAL - MIN_INPUT_VAL).item()

                    _, pred_result = compute_measure(x, y, p, data_range)
                    psnr_sum += float(pred_result[0])
                    ssim_sum += float(pred_result[1])
                    rmse_sum += float(pred_result[2])
                    n_images += 1
                    print(f"val image {n_images}: {pred_result}")

        self.REDCNN.train(True)
        n_images = max(n_images, 1)
        return psnr_sum / n_images, ssim_sum / n_images, rmse_sum / n_images


    def train_with_2detect(self):
        train_losses = []

        do_validation = self.val_data_loader is not None

        total_iters = 0
        start_time = time.time()
        for epoch in range(1, self.num_epochs):
            self.REDCNN.train(True)

            for iter_, (target, input) in enumerate(self.data_loader):
                total_iters += 1

                # convert to torch tensor and add 1 channel
                input = torch.tensor(input).unsqueeze(0).float().to(self.device)
                target = torch.tensor(target).unsqueeze(0).float().to(self.device)

                # Normalize using local min-max value.
                # NORMALIZATION IS BEING DONE BEFORE PATCH EXTRACTION TO FOLLOW SAME PROCEDURE DONE ON COREDIFF
                MIN_INPUT_VAL = input.min()
                MAX_INPUT_VAL = input.max()
                rescaled_input = self.normalize_(input, MIN_INPUT_VAL, MAX_INPUT_VAL)
                rescaled_target = self.normalize_(target, MIN_INPUT_VAL, MAX_INPUT_VAL)
                if self.patch_size: # patch training
                    rescaled_input = rescaled_input.view(-1, 1, self.patch_size, self.patch_size)
                    rescaled_target = rescaled_target.view(-1, 1, self.patch_size, self.patch_size)

                pred = self.REDCNN(rescaled_input)
                loss = self.criterion(pred, rescaled_target)
                self.REDCNN.zero_grad()
                self.optimizer.zero_grad()
                

                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())
                self.log_train_loss(total_iters, loss.item())

                # print
                if total_iters % self.print_iters == 0:
                    print("STEP [{}], EPOCH [{}/{}], ITER [{}/{}] \nLOSS: {:.8f}, TIME: {:.1f}s".format(total_iters, epoch, 
                                                                                                        self.num_epochs, iter_+1, 
                                                                                                        len(self.data_loader), loss.item(), 
                                                                                                        time.time() - start_time))
                # learning rate decay
                if total_iters % self.decay_iters == 0:
                    self.lr_decay()
                # save model
                if total_iters % self.save_iters == 0:
                    self.save_model(total_iters)
                    #np.save(os.path.join(self.save_path, 'loss_{}_iter.npy'.format(total_iters)), np.array(train_losses))
            
            ####### CODE ADD: validation done every end of epoch ######  
            val_psnr, val_ssim, val_rmse = self.validate_2detect()
            print("VALIDATION @ STEP [{}] \nPSNR: {:.4f}, SSIM: {:.4f}, RMSE: {:.6f}".format(total_iters, val_psnr, val_ssim, val_rmse))
            self.log_test_metrics(total_iters, val_rmse, val_psnr, val_ssim)
            ####### CODE ADD ######
    #######  CODE ADD #######

    
    def test(self):
        del self.REDCNN
        # load
        self.REDCNN = RED_CNN().to(self.device)
        self.load_model(self.test_iters)

        # compute PSNR, SSIM, RMSE
        ori_psnr_avg, ori_ssim_avg, ori_rmse_avg = 0, 0, 0
        pred_psnr_avg, pred_ssim_avg, pred_rmse_avg = 0, 0, 0

        with torch.no_grad():
            for i, (x, y) in enumerate(self.data_loader):
                shape_ = x.shape[-1]
                x = x.unsqueeze(0).float().to(self.device)
                y = y.unsqueeze(0).float().to(self.device)

                pred = self.REDCNN(x)

                # denormalize, truncate
                x = self.trunc(self.denormalize_(x.view(shape_, shape_).cpu().detach()))
                y = self.trunc(self.denormalize_(y.view(shape_, shape_).cpu().detach()))
                pred = self.trunc(self.denormalize_(pred.view(shape_, shape_).cpu().detach()))

                data_range = self.trunc_max - self.trunc_min

                original_result, pred_result = compute_measure(x, y, pred, data_range)
                ori_psnr_avg += original_result[0]
                ori_ssim_avg += original_result[1]
                ori_rmse_avg += original_result[2]
                pred_psnr_avg += pred_result[0]
                pred_ssim_avg += pred_result[1]
                pred_rmse_avg += pred_result[2]

                # save result figure
                if self.result_fig:
                    self.save_fig(x, y, pred, i, original_result, pred_result)

                printProgressBar(i, len(self.data_loader),
                                 prefix="Compute measurements ..",
                                 suffix='Complete', length=25)
            print('\n')
            print('Original === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(ori_psnr_avg/len(self.data_loader), 
                                                                                            ori_ssim_avg/len(self.data_loader), 
                                                                                            ori_rmse_avg/len(self.data_loader)))
            print('\n')
            print('Predictions === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(pred_psnr_avg/len(self.data_loader), 
                                                                                                  pred_ssim_avg/len(self.data_loader), 
                                                                                                  pred_rmse_avg/len(self.data_loader)))

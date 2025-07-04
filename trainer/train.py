import torch
import torch.nn as nn
import logging
import itertools
import torch.nn.functional as F
logger = logging.getLogger(__name__)
from utils import initialize_optimizer, initialize_scheduler, tensor2img, calculate_psnr, calculate_ssim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from loss.loss import  PerceptualLoss, EdgeLoss, FFTLoss
import time
import os


class Trainer():
    def __init__(self, args, models, dataloaders, ckp):
        self.args = args
        self.ckp = ckp
        self.train_dataloader, self.test_dataloader = dataloaders
        self.max_evaluation_count = self.args.data.max_evaluation_count
        self.selected_gpus = args.train.gpus

        # 判断设备
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.selected_gpus[0]}")
            torch.cuda.set_device(self.selected_gpus[0])
        elif torch.backends.mps.is_available():  # macOS M1/M2 GPU
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # 多 GPU 适配
        self.models = {}
        for name, model in models.items():
            if torch.cuda.device_count() > 1 and torch.cuda.is_available() and len(self.selected_gpus) > 1:
                model = nn.DataParallel(model, device_ids=self.selected_gpus)
            model.to(self.device)
            self.models[name] = model

        # 定义优化器和调度器
        self.optimizers = {name: initialize_optimizer(args, model, name) for name, model in self.models.items()}
        self.schedulers = {name: initialize_scheduler(args, optimizer, name) for name, optimizer in
                           self.optimizers.items()}

        self.existing_step = 0  # 记录已有的训练步数
        self.current_loss = []  # 存储当前的损失值
        # 初始化数据迭代器
        self.loader_train_iter = iter(self.train_dataloader)

        # ----------------------------------
        # 内容感知loss
        self.criterionPL = PerceptualLoss(device=self.device, model_path=self.args.train.vgg_model_path)
        # 鲁棒性L1loss
        self.criterionML = nn.L1Loss().to(self.device)
        self.criterionEDGE = EdgeLoss(device=self.device).to(self.device)

        self.criterionFFT = FFTLoss(device=self.device)

        ### Model Loading
        self.load_previous_ckp(models=self.models)

        # 计算测试轮数和保存模型轮数
        self.test_every = 1
        self.best_psnr = 0

        self.all_best_psnr = 0
        self.best_model_root = os.path.join(self.ckp.log_dir, '0.pth')
        self.all_best_model_root = os.path.join(self.ckp.log_dir, f"all_best_0.pt")

    def load_previous_ckp(self, models=None):
        if self.ckp is not None:
            if models is None:
                models = self.models

            self.existing_step, self.current_loss = self.ckp.load_checkpoint(models, self.optimizers, self.schedulers)

            if self.existing_step > 0:
                logger.info('Resuming training.')

    def train(self):
        # now_iter = int(self.ckp.load_model)
        now_iter = 1
        self.test(now_iter) # 开局先测试
        # self.testall(now_iter)
        last_print_time = time.time()  #
        while now_iter <= self.args.train.max_iter:
            self.models['LFAEUnet'].train()
            try:
                lr, hr = next(self.loader_train_iter)
            except StopIteration:
                self.loader_train_iter = iter(self.train_dataloader)
                # logger.info(f'Iter {now_iter} Resuming Training Data.')
                # 每三个测全部
                if self.test_every % 9 == 0 :
                    self.testall(now_iter)
                else:
                    self.test(now_iter)
                lr, hr = next(self.loader_train_iter)
                self.test_every += 1

            lr_tensor = lr.to(self.device, dtype=torch.float32)
            hr_tensor = hr.to(self.device, dtype=torch.float32)

            out = self.models['LFAEUnet'](lr_tensor)

            self.optimizers['LFAEUnet'].zero_grad()


            loss_PL = self.criterionPL(out, hr_tensor.detach())
            loss_ML = self.criterionML(out, hr_tensor)
            # loss_edge = self.criterionEDGE(out, hr_tensor)
            # loss_fft = self.criterionFFT(out, hr_tensor)
            loss_G = loss_ML +  0.05 * loss_PL

            loss_G.backward()
            self.optimizers['LFAEUnet'].step()
            self.schedulers['LFAEUnet'].step()

            if now_iter % self.args.train.print_lr == 0:
                current_lr = self.optimizers['LFAEUnet'].param_groups[0]['lr']
                logger.info(f"Current LR: {current_lr:.6f}")

            # 每 print_loss 次打印一次损失和耗时（区间耗时）
            if now_iter % self.args.train.print_loss == 0:
                current_time = time.time()
                interval_time = current_time - last_print_time
                interval_str = time.strftime('%H:%M:%S', time.gmtime(interval_time))

                logger.info(
                    'Iter [{:04d}/{}]\t'
                    'Raw Losses [PL: {:.3f} | ML: {:.3f}  ]  '
                    'Weighted Total: {:.3f}\t'
                    'Time: {}'.format(
                        now_iter, self.args.train.max_iter,
                        loss_PL.item(), loss_ML.item(),
                        loss_G.item(),
                        interval_str
                    )
                )
                last_print_time = current_time  # 🔄 更新为本次时间

            if now_iter == self.args.train.max_iter:
                self.save_ckp(now_iter, self.models)
            now_iter += 1

    def test(self, now_iter):
        self.models['LFAEUnet'].eval()
        with torch.no_grad():
            eval_psnr = 0
            eval_ssim = 0
            max_test_samples = self.max_evaluation_count
            start_time = time.time()

            for lr, hr,filename in itertools.islice(self.test_dataloader, max_test_samples):
                lr_tensor = lr.clone().detach().to(dtype=torch.float32, device=self.device)

                height, width = lr_tensor.shape[2], lr_tensor.shape[3]
                img_multiple_of = 8
                H, W = ((height + img_multiple_of) // img_multiple_of) * img_multiple_of, (
                        (width + img_multiple_of) // img_multiple_of) * img_multiple_of
                padh = H - height if height % img_multiple_of != 0 else 0
                padw = W - width if width % img_multiple_of != 0 else 0
                lr_tensor = F.pad(lr_tensor, (0, padw, 0, padh), 'reflect')


                out = self.models['LFAEUnet'](lr_tensor)
                sr = out[:, :, :height, :width]

                sr = tensor2img(sr)
                hr = tensor2img(hr)

                now_psnr = calculate_psnr(sr, hr)
                now_ssim = calculate_ssim(sr, hr)

                eval_psnr += now_psnr
                eval_ssim += now_ssim

            end_time = time.time()
            test_time = end_time - start_time

            avg_psnr = eval_psnr / max_test_samples
            avg_ssim = eval_ssim / max_test_samples

            # 更新最优模型逻辑
            if avg_psnr > self.best_psnr:
                logger.info(f"[now_iter {now_iter}] New best model found! PSNR: {avg_psnr:.3f}, SSIM: {avg_ssim:.4f}")

                # 删除旧的最优模型
                if os.path.exists(self.best_model_root):
                    os.remove(self.best_model_root)

                # 保存新的最优模型
                best_model_path = os.path.join(self.ckp.log_dir, f"{now_iter}.pt")
                self.save_ckp(now_iter, models=self.models)

                # 更新状态
                self.best_psnr = avg_psnr

                self.best_model_root = best_model_path
                # 发现最好的测试全部
                #self.testall(now_iter)

            logger.info('[now_iter {}]\tPSNR: {:.3f} SSIM: {:.4f} | Test Time: {:.2f}s'.format(
                self.args.data.test_data,
                avg_psnr,
                avg_ssim,
                test_time
            ))

    def tile_forward(self,model, input_tensor, tile_size=640, tile_overlap=80):
        B, C, H, W = input_tensor.shape
        stride = tile_size - tile_overlap
        E = torch.zeros_like(input_tensor)  # 用于拼接结果
        W_map = torch.zeros_like(input_tensor)  # 加权融合

        for y in range(0, H, stride):
            for x in range(0, W, stride):
                y1 = y
                x1 = x
                y2 = min(y + tile_size, H)
                x2 = min(x + tile_size, W)

                patch = input_tensor[:, :, y1:y2, x1:x2]
                ph, pw = patch.shape[2], patch.shape[3]

                # Pad small patch to tile_size
                pad_bottom = tile_size - ph
                pad_right = tile_size - pw
                if pad_bottom < ph and pad_right < pw:
                    patch = F.pad(patch, (0, pad_right, 0, pad_bottom), mode='reflect')
                else:
                    patch = F.pad(patch, (0, pad_right, 0, pad_bottom), mode='replicate')  # 或 'constant'



                with torch.no_grad():
                    out_patch = model(patch)[:, :, :ph, :pw]

                E[:, :, y1:y2, x1:x2] += out_patch
                W_map[:, :, y1:y2, x1:x2] += 1

        # 避免除以 0
        W_map[W_map == 0] = 1
        return E / W_map

    def get_safe_tile_params(self,H, W, max_tile=512, overlap=64):
        def last_patch_size(size, tile, overlap):
            if size <= tile:
                return size
            n_tiles = (size - overlap) // (tile - overlap)
            last_start = n_tiles * (tile - overlap)
            return size - last_start

        for tile in range(max_tile, overlap + 1, -8):
            h_last = last_patch_size(H, tile, overlap)
            w_last = last_patch_size(W, tile, overlap)
            if h_last >= overlap and w_last >= overlap:
                return tile, overlap
        raise ValueError("Cannot find suitable tile_size for this image size.")

    def testall(self,now_iter):
        # 测试
        self.models['LFAEUnet'].eval()
        with torch.no_grad():
            eval_psnr = 0
            eval_ssim = 0
            for idx, (lr, hr,filename) in enumerate(self.test_dataloader):
                lr_tensor = lr.clone().detach().to(dtype=torch.float32, device=self.device)
                img_multiple_of = 8
                # Pad the input if not_multiple_of 8
                height, width = lr_tensor.shape[2], lr_tensor.shape[3]
                H, W = ((height + img_multiple_of) // img_multiple_of) * img_multiple_of, (
                        (width + img_multiple_of) // img_multiple_of) * img_multiple_of
                padh = H - height if height % img_multiple_of != 0 else 0
                padw = W - width if width % img_multiple_of != 0 else 0
                lr_tensor = F.pad(lr_tensor, (0, padw, 0, padh), 'reflect')

                if width > 1000:
                    out = self.tile_forward(self.models['LFAEUnet'], lr_tensor, tile_size=512, tile_overlap=64)
                else:
                    out = self.models['LFAEUnet'](lr_tensor)

                sr = out[:, :, :height, :width]

                sr_img = tensor2img(sr)
                hr_img = tensor2img(hr)

                now_psnr = calculate_psnr(sr_img, hr_img)
                now_ssim = calculate_ssim(sr_img, hr_img)

                eval_psnr += now_psnr
                eval_ssim += now_ssim

                if self.args.train.test_only:
                    logger.info(f"[Sample {idx} - {filename}] PSNR: {now_psnr:.3f} | SSIM: {now_ssim:.4f}")


            avg_psnr = eval_psnr / len(self.test_dataloader)
            avg_ssim = eval_ssim / len(self.test_dataloader)
            # 更新最优模型逻辑
            if not self.args.train.test_only:
                if avg_psnr > self.all_best_psnr:
                    logger.info(f"[now_iter {now_iter}] New best model found! PSNR: {avg_psnr:.3f}, SSIM: {avg_ssim:.4f}")

                    # 删除旧的最优模型
                    if os.path.exists(self.all_best_model_root):
                        os.remove(self.all_best_model_root)

                    # 保存新的最优模型
                    best_model_path = os.path.join(self.ckp.log_dir, f"all_best_{now_iter}.pt")
                    self.save_ckp(f"all_best_{now_iter}", models=self.models)

                    # 更新状态
                    self.all_best_psnr = avg_psnr

                    self.all_best_model_root = best_model_path

            logger.info('[Dataset {}]\tAvg PSNR: {:.3f} Avg SSIM: {:.4f}'.format(
                self.args.data.test_data,
                avg_psnr,
                avg_ssim,
            ))


    # 保存检查点
    def save_ckp(self, step, models=None):
        if self.ckp is not None:
            if models is None:
                models = self.model
            self.ckp.save_checkpoint(step, self.current_loss, models, self.optimizers, self.schedulers)




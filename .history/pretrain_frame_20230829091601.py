import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from Utils import try_gpu, check_gpus
from Utils import build_net, build_loss, build_optimizer, build_schedulers
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler


class PretrainFrame():
    def __init__(self, cfg, num_dataset):
        self.last_epoch = cfg.START_EPOCH - 1
        self.num_batch = num_dataset // cfg.DATA.BATCH_SIZE + 1
        self.num_iter = self.num_batch * config.NUM_ITERS_PER_EPOCH


        self.class_list = cfg.DATA.CLASS_LIST
        self.lr = cfg.TRAIN.LEARNING_RATE

        self.net = build_net(cfg)

        if check_gpus(cfg.DEVICE_IDS):
            self.device_ids = cfg.DEVICE_IDS
            if torch.cuda.device_count() > 1:
                self.student = torch.nn.DataParallel(self.student, device_ids=cfg.DEVICE_IDS)
                self.teacher = torch.nn.DataParallel(self.teacher, device_ids=cfg.DEVICE_IDS)
            elif torch.cuda.device_count() == 1:
                pass
            else:
                print('No GPU available, training on CPU')
            self.mdevice = try_gpu(cfg.DEVICE_IDS[0])
            self.student = self.student.to(self.mdevice)
            self.teacher = self.teacher.to(self.mdevice)
        else:
            raise AssertionError("Invalid device ids")
        
        self.loss_fuc = build_loss(cfg)
        self.optimizer = build_optimizer(cfg, self.student)

        (self.lr_scheduler,
        self.wd_scheduler,
        self.momentum_scheduler,
        self.teacher_temp_scheduler,
        self.last_layer_lr_scheduler) = build_schedulers(cfg)

        # FIXME: fp16 and clip_grad
        # if fp16_scaler is not None:
        #     if cfg.optim.clip_grad:
        #         fp16_scaler.unscale_(optimizer)
        #         for v in model.student.values():
        #             v.clip_grad_norm_(cfg.optim.clip_grad)
        #     fp16_scaler.step(optimizer)
        #     fp16_scaler.update()
        # else:
        #     if cfg.optim.clip_grad:
        #         for v in model.student.values():
        #             v.clip_grad_norm_(cfg.optim.clip_grad)
        #     optimizer.step()

        self.fp16_scaler = GradScaler()

        if cfg.NET.PRETRAIN_PATH:
            self.load_weights(cfg.NET.PRETRAIN_PATH)

        if cfg.IS_EVAL:
            for i in self.student.modules():
                if isinstance(i, nn.BatchNorm2d):
                    i.eval()     # 不启用 BatchNormalization 和 Dropout

    def apply_optim_scheduler(optimizer, lr, wd, last_layer_lr):
        for param_group in optimizer.param_groups:
            is_last_layer = param_group["is_last_layer"]
            lr_multiplier = param_group["lr_multiplier"]
            wd_multiplier = param_group["wd_multiplier"]
            param_group["weight_decay"] = wd * wd_multiplier
            param_group["lr"] = (last_layer_lr if is_last_layer else lr) * lr_multiplier

    def set_input(self, imgs):
        self.imgs = imgs

    def optimize(self, it):
        self.imgs = Variable(self.imgs.to(self.mdevice), volatile=False)

        # apply schedules
        lr = self.lr_scheduler[it]
        wd = self.wd_scheduler[it]
        last_layer_lr = self.last_layer_lr_scheduler[it]
        self.apply_optim_scheduler(self.optimizer, lr, wd, last_layer_lr)

        teacher_temp = self.teacher_temp_scheduler[it]
        self.optimizer.zero_grad()
        with autocast():
            s1, t1 = self.net(self.imgs)   # pred: batch_size, num_classes, H, W
            l = self.loss_fuc(s1, t1, teacher_temp)
        self.fp16_scaler.scale(l).backward()
        self.fp16_scaler.step(self.optimizer)
        self.fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            mom = self.momentum_scheduler[it]  # momentum parameter
            for param_q, param_k in zip(self.student.module.parameters(), self.teacher.module.parameters()):
                param_k.data.mul_(mom).add_((1 - mom) * param_q.detach().data)
                
        return l.item()

    def save_weights(self, train_data_path, net_name, cfg_note, epoch, best_loss):
        """save checkpoint with best loss NOT mIoU
        """
        file_list = [w for w in os.listdir(train_data_path) if os.path.isfile(os.path.join(train_data_path, w))]
        weight_list = list(filter(lambda x: (len(x.split('_')) == 4) , file_list))
        weight_list = list(filter(lambda x: (x.split('_')[0] == net_name) & (x.split('_')[1] == cfg_note)
                                   & (x.split('_')[3][-7:] == '.params'), weight_list))
        best_loss_str = str(best_loss)[2:5]
        if weight_list:
            for weight_names in weight_list:
                namesplits = weight_names.split('_')
                if namesplits[3][:3] <= best_loss_str:
                    os.remove(f'{train_data_path}/{weight_names}')
                    path = f"{train_data_path}/{net_name}_{cfg_note}_ep{epoch}_{best_loss_str}.params"
                    torch.save({'epoch': epoch,
                                'model_state_dict': self.student.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'loss': best_loss}, path)
        else:
            path = f"{train_data_path}/{net_name}_{cfg_note}_ep{epoch}_{best_loss_str}.params"
            torch.save({'epoch': epoch,
                        'model_state_dict': self.student.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': best_loss}, path)
            
    def load_weights(self, weight_path):
        checkpoint = torch.load(weight_path)
        self.student.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.loss = checkpoint['loss']


    def update_lr(self):
        self.lr_scheduler.step()
        self.lr = self.lr_scheduler.get_last_lr()[0]
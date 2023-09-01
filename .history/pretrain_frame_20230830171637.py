import os
import sys
import math
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
        self.num_iter_per_epoch = num_dataset // cfg.DATA.BATCH_SIZE + 1  # 训练一个batch即一个iter
        self.start_iter = cfg.START_EPOCH * self.num_iter_per_epoch
        self.it = self.start_iter
        self.epoch = cfg.START_EPOCH

        self.clip_grad = cfg.CLIP_GRAD
        self.learning_rate = cfg.LR_SCHEDULER.LEARNING_RATE
        self.weight_decay = cfg.WD_SCHEDULER.WEIGHT_DECAY
        self.teacher_momentum = cfg.TM_SCHEDULER.TEACHER_MOMENTUM
        self.teacher_temperature = cfg.TT_SCHEDULER.TEACHER_TEMP
        self.last_layer_lr = cfg.LR_SCHEDULER.LEARNING_RATE

        self.net = build_net(cfg)

        if check_gpus(cfg.DEVICE_IDS):
            self.device_ids = cfg.DEVICE_IDS
            if torch.cuda.device_count() > 1:
                self.net = torch.nn.DataParallel(self.net, device_ids=cfg.DEVICE_IDS)
            elif torch.cuda.device_count() == 1:
                pass
            else:
                print('No GPU available, training on CPU')
            self.mdevice = try_gpu(cfg.DEVICE_IDS[0])
            self.net = self.net.to(self.mdevice)
        else:
            raise AssertionError("Invalid device ids")
        
        self.loss_fuc = build_loss(cfg)
        self.optimizer = build_optimizer(cfg, self.net)

        (self.lr_scheduler,
        self.wd_scheduler,
        self.momentum_scheduler,
        self.teacher_temp_scheduler,
        self.last_layer_lr_scheduler) = build_schedulers(cfg, self.num_iter_per_epoch)

            
        if cfg.IS_FP16:
            self.fp16_scaler = GradScaler()
        else:
            self.fp16_scaler = None

        if cfg.PRETRAIN_PATH:
            self.load_weights(cfg.PRETRAIN_PATH)

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

    def set_input(self, crops_list):
        self.crops_list = crops_list

    def optimize(self):
        self.epoch = self.it // self.num_iter_per_epoch + 1

        for crop in self.crops_list:
            crop = Variable(crop.to(self.mdevice), volatile=False)

        # apply schedules
        self.learning_rate = self.lr_scheduler[self.it]
        self.weight_decay = self.wd_scheduler[self.it]
        self.last_layer_lr = self.last_layer_lr_scheduler[self.it]
        self.apply_optim_scheduler(self.optimizer, self.learning_rate, self.weight_decay, self.last_layer_lr)

        # forward and backward
        self.teacher_temperature = self.teacher_temp_scheduler[self.it]
        self.optimizer.zero_grad()
        with autocast():
            student_output, teacher_output = self.net(self.crops_list)   # pred: batch_size, num_classes, H, W
            loss = self.loss_fuc(student_output, teacher_output, self.teacher_temperature)
        if self.fp16_scaler is not None:
            self.fp16_scaler.scale(loss).backward()
        else:
            loss.backward()

        # clip gradient and update parameters
        if self.fp16_scaler is not None:
            if self.clip_grad:
                self.fp16_scaler.unscale_(self.optimizer)  # unscale the gradients of optimizer's assigned params in-place
                for v in self.net.student.values():
                    v.clip_grad_norm_(self.clip_grad)
            self.fp16_scaler.step(self.optimizer)
            self.fp16_scaler.update()
        else:
            if self.clip_grad:
                for v in self.net.student.values():
                    v.clip_grad_norm_(self.clip_grad)
            self.optimizer.step()

        # check if loss is valid
        if not math.isfinite(loss.item()):
            print(f"Loss is {loss.item()}, stopping training", force=True)
            sys.exit(1)

        # EMA update for the teacher
        with torch.no_grad():
            self.teacher_momentum = self.momentum_scheduler[self.it]  # momentum parameter
            for param_q, param_k in zip(self.student.module.parameters(), self.teacher.module.parameters()):
                param_k.data.mul_(self.teacher_momentum).add_((1 - self.teacher_momentum) * param_q.detach().data)
        
        self.it += 1
        return loss.item()

    def save_best_weights(self, output_path, net_name, cfg_note, epoch, best_loss):
        """save checkpoint with best loss NOT mIoU"""
        file_list = [w for w in os.listdir(output_path) if os.path.isfile(os.path.join(output_path, w))]
        weight_list = list(filter(lambda x: (len(x.split('_')) == 4) , file_list))
        weight_list = list(filter(lambda x: (x.split('_')[0] == net_name) & (x.split('_')[1] == cfg_note)
                                   & (x.split('_')[3][-7:] == '.params'), weight_list))
        best_loss_str = str(best_loss)[2:5]
        if weight_list:
            for weight_names in weight_list:
                namesplits = weight_names.split('_')
                if namesplits[3][:3] <= best_loss_str:
                    os.remove(f'{output_path}/{weight_names}')
                    path = f"{output_path}/{net_name}_{cfg_note}_ep{epoch}_{best_loss_str}.params"
                    torch.save({'model_state_dict': self.student.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict()}, path)
        else:
            path = f"{output_path}/{net_name}_{cfg_note}_ep{epoch}_{best_loss_str}.params"
            torch.save({'model_state_dict': self.student.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict()}, path)
            
    def save_weights(self, output_path, net_name, cfg_note, epoch):
        output_path = output_path + '/autosave'
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        path = f"{output_path}/{net_name}_{cfg_note}_ep{epoch}.params"
        torch.save({'model_state_dict': self.student.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()}, path)

            
    def load_weights(self, weight_path):
        checkpoint = torch.load(weight_path)
        self.student.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


    def update_lr(self):
        self.lr_scheduler.step()
        self.learning_rate = self.lr_scheduler.get_last_lr()[0]
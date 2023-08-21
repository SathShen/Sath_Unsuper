import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from Utils import try_gpu, check_gpus
from Utils import build_net, build_loss, build_optimizer, build_lrscheduler
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler


class PretrainFrame():
    def __init__(self, cfgs):
        self.last_epoch = cfgs.TRAIN.START_EPOCH - 1
        self.class_list = cfgs.DATA.CLASS_LIST
        self.lr = cfgs.TRAIN.LEARNING_RATE

        self.student_model_dict = dict()
        self.teacher_model_dict = dict()
        self.student_backbone = build_net(cfgs)
        self.teacher_backbone = build_net(cfgs)
                student_model_dict["backbone"] = student_backbone
        teacher_model_dict["backbone"] = teacher_backbone
        
        if check_gpus(cfgs.DEVICE_IDS):
            self.device_ids = cfgs.DEVICE_IDS
            if torch.cuda.device_count() > 1:
                self.student_backbone = torch.nn.DataParallel(self.student_backbone, device_ids=cfgs.DEVICE_IDS)
                self.teacher_backbone = torch.nn.DataParallel(self.teacher_backbone, device_ids=cfgs.DEVICE_IDS)
            elif torch.cuda.device_count() == 1:
                pass
            else:
                print('No GPU available, training on CPU')
            self.mdevice = try_gpu(cfgs.DEVICE_IDS[0])
            self.student_backbone = self.student_backbone.to(self.mdevice)
            self.teacher_backbone = self.teacher_backbone.to(self.mdevice)
        else:
            raise AssertionError("Invalid device ids")
        
        self.loss_fuc = build_loss(cfgs)
        self.optimizer = build_optimizer(cfgs, self.student_backbone)
        self.lr_scheduler = build_lrscheduler(cfgs, self.optimizer, self.last_epoch)
        self.scaler = GradScaler()

        if cfgs.NET.PRETRAIN_PATH:
            self.load_weights(cfgs.NET.PRETRAIN_PATH)

        if cfgs.IS_EVAL:
            for i in self.student_backbone.modules():
                if isinstance(i, nn.BatchNorm2d):
                    i.eval()     # 不启用 BatchNormalization 和 Dropout

    def set_input(self, imgs):
        self.imgs = imgs

    def forward(self, volatile=False):
        self.imgs = Variable(self.imgs.to(self.mdevice), volatile=volatile)

    def optimize(self):
        self.forward()
        self.optimizer.zero_grad()
        with autocast():
            preds = self.student_backbone(self.imgs)   # pred: batch_size, num_classes, H, W
            l = self.loss_fuc(preds)
            
        self.scaler.scale(l).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        # l.backward()
        # self.optimizer.step()
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
                                'model_state_dict': self.student_backbone.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'loss': best_loss}, path)
        else:
            path = f"{train_data_path}/{net_name}_{cfg_note}_ep{epoch}_{best_loss_str}.params"
            torch.save({'epoch': epoch,
                        'model_state_dict': self.student_backbone.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': best_loss}, path)
            
    def load_weights(self, weight_path):
        checkpoint = torch.load(weight_path)
        self.student_backbone.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.loss = checkpoint['loss']


    def update_lr(self):
        self.lr_scheduler.step()
        self.lr = self.lr_scheduler.get_last_lr()[0]
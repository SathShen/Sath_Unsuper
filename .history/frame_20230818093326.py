import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from Utils import SegmentationMetrics, try_gpu, check_gpus
from tqdm import tqdm
from Utils import build_net, build_loss, build_optimizer, build_scheduler, batch_one_hot_decode
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler


def cvt_posibility2class(preds):
    """
    preds: (batch_size, num_classes, height, width)
    out: (batch_size, height, width)"""
    preds = preds.argmax(dim=1)
    return preds.type(torch.uint8)


class TrainFrame():
    def __init__(self, cfgs):
        self.last_epoch = cfgs.TRAIN.START_EPOCH - 1
        self.class_list = cfgs.DATA.CLASS_LIST
        self.lr = cfgs.TRAIN.LEARNING_RATE

        self.net = build_net(cfgs)
        
        if check_gpus(cfgs.DEVICE_IDS):
            self.device_ids = cfgs.DEVICE_IDS
            if torch.cuda.device_count() > 1:
                self.net = torch.nn.DataParallel(self.net, device_ids=cfgs.DEVICE_IDS)
                # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False) # 分布式计算
            elif torch.cuda.device_count() == 1:
                pass
            else:
                print('No GPU available, training on CPU')
            self.mdevice = try_gpu(cfgs.DEVICE_IDS[0])
            self.net = self.net.to(self.mdevice)
        else:
            raise AssertionError("Invalid device ids")
        
        # self.net = self.net.to(self.mdevice)
        self.loss_fuc = build_loss(cfgs)
        self.optimizer = build_optimizer(cfgs, self.net)
        self.scheduler = build_scheduler(cfgs, self.optimizer, self.last_epoch)
        self.scaler = GradScaler()

        if cfgs.NET.PRETRAIN_PATH:
            self.load_weights(cfgs.NET.PRETRAIN_PATH)

        if cfgs.IS_EVAL:
            for i in self.net.modules():
                if isinstance(i, nn.BatchNorm2d):
                    i.eval()     # 不启用 BatchNormalization 和 Dropout
        self.train_metrics = SegmentationMetrics(cfgs.DATA.NUM_CLASSES)
        self.valid_metrics = SegmentationMetrics(cfgs.DATA.NUM_CLASSES)

    def set_input(self, imgs):
        self.imgs = imgs

    def forward(self, volatile=False):
        self.imgs = Variable(self.imgs.to(self.mdevice), volatile=volatile)

    def optimize(self):
        self.forward()
        self.optimizer.zero_grad()
        with autocast():
            preds = self.net(self.imgs)   # pred: batch_size, num_classes, H, W
            metrics_preds = cvt_posibility2class(preds).to(self.mdevice)
            metrics_labs = batch_one_hot_decode(
                self.labs, list(range(len(self.class_list))), self.mdevice).to(self.mdevice)
            self.train_metrics.add_imgs(metrics_preds, metrics_labs)
            l = self.loss_fuc(preds, self.labs)

            
        self.scaler.scale(l).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        # l.backward()
        # self.optimizer.step()
        return l.item()

    def save_weights(self, train_data_path, net_name, cfg_note, epoch, best_mIoU, loss, dataset='train'):
        file_list = [w for w in os.listdir(train_data_path) if os.path.isfile(os.path.join(train_data_path, w))]
        weight_list = list(filter(lambda x: (len(x.split('_')) == 5) , file_list))
        weight_list = list(filter(lambda x: (x.split('_')[0] == net_name) & (x.split('_')[1] == cfg_note)
                                  & (x.split('_')[3] == dataset) & (x.split('_')[4][-7:] == '.params'), weight_list))
        best_mIoU_str = str(best_mIoU)[2:5]
        if weight_list:
            for weight_names in weight_list:
                namesplits = weight_names.split('_')
                if namesplits[4][:3] <= best_mIoU_str:
                    os.remove(f'{train_data_path}/{weight_names}')
                    path = f"{train_data_path}/{net_name}_{cfg_note}_ep{epoch}_{dataset}_{best_mIoU_str}.params"
                    torch.save({'epoch': epoch,
                                'model_state_dict': self.net.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'loss': loss}, path)
        else:
            path = f"{train_data_path}/{net_name}_{cfg_note}_ep{epoch}_{dataset}_{best_mIoU_str}.params"
            torch.save({'epoch': epoch,
                        'model_state_dict': self.net.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': loss}, path)
            
    def load_weights(self, weight_path):
        checkpoint = torch.load(weight_path)
        self.net.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.loss = checkpoint['loss']


    def update_lr(self):
        self.scheduler.step()
        self.lr = self.scheduler.get_last_lr()[0]


    def validiate(self, valid_data_loader_iter):
        with torch.no_grad():
            for imgs, labs in tqdm(valid_data_loader_iter, ncols=80):
                imgs = imgs.to(self.mdevice)
                labs = labs.to(self.mdevice)
                preds = self.net(imgs)
                metrics_preds = cvt_posibility2class(preds).to(self.mdevice)
                metrics_labs = batch_one_hot_decode(
                    labs, list(range(len(self.class_list))), self.mdevice).to(self.mdevice)
                self.valid_metrics.add_imgs(metrics_preds, metrics_labs)


def test():
    a = torch.randn((4, 3, 5, 5))
    b = cvt_posibility2class(a)
    print(a)
    print(b)


if __name__ == "__main__":
    test()
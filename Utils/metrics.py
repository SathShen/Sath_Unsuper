import numpy as np
import torch


class SegmentationMetrics(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.accu_matrix = np.zeros((self.num_classes, self.num_classes))

    def add_imgs(self, preds, labels):
        """
        支持单张图片或batch
        preds, labels ：H,W or
        preds, labels ：N,H,W
        """
        preds = np.array(preds.to('cpu'))
        labels = np.array(labels.to('cpu'))
        self.accu_matrix += self.get_confusion_matrix(preds, labels)

    def reset(self):
        self.accu_matrix = np.zeros((self.num_classes, self.num_classes))

    def get_confusion_matrix(self, preds, labels):
        """
        num_classes*labels代表混淆矩阵的纵坐标， preds代表横坐标，当预测正确,label对应pred，在对角线上
        preds,labels都要求为uint8, 支持batch (batch, Height, width)
        """
        assert preds.shape == labels.shape, 'preds与labels的shape不一致'
        assert ((preds >= 0) & (preds < self.num_classes)).all(), 'preds数值不正确！'
        assert ((labels >= 0) & (labels < self.num_classes)).all(), 'labels数值不正确！'
        confusion_Idx = self.num_classes * labels.flatten() + preds.flatten()
        count = np.bincount(confusion_Idx, minlength=self.num_classes**2)
        matrix = count.reshape(self.num_classes, self.num_classes)
        return matrix

    def pixel_accuracy(self):
        # return all class overall pixel accuracy
        #  PA = acc = (TP + TN) / (TP + TN + FP + TN)
        assert self.accu_matrix is not None, '请先使用add_imgs设置preds与labels'
        pa = np.diag(self.accu_matrix).sum() / self.accu_matrix.sum()
        return pa

    def class_precision(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # precision = (TP) / TP + FP
        assert self.accu_matrix is not None, '请先使用add_imgs设置preds与labels'
        inter = np.diag(self.accu_matrix)
        accu = self.accu_matrix.sum(axis=0)   # 0是竖着加
        cpa = np.divide(inter, accu, out=np.zeros_like(inter, dtype=np.float64), where=accu!=0)
        return cpa  # 返回的是一个列表值，表示各类别的查准率

    def macro_precision(self):
        assert self.accu_matrix is not None, '请先使用add_imgs设置preds与labels'
        cpa = self.class_precision()
        mpa = np.mean(cpa)
        return mpa

    def frequency_weighted_precision(self):
        assert self.accu_matrix is not None, '请先使用add_imgs设置preds与labels'
        frequency_weights = self.accu_matrix.sum(axis=1) / self.accu_matrix.sum()
        cpa = self.class_precision()
        fw_precision = (frequency_weights * cpa).sum()
        return fw_precision

    def class_recall(self):
        assert self.accu_matrix is not None, '请先使用add_imgs设置preds与labels'
        inter = np.diag(self.accu_matrix)
        accu = self.accu_matrix.sum(axis=1)
        cr = np.divide(inter, accu, out=np.zeros_like(inter, dtype=np.float64), where=accu!=0)
        return cr

    def macro_recall(self):
        assert self.accu_matrix is not None, '请先使用add_imgs设置preds与labels'
        cr = self.class_recall()
        mr = np.mean(cr)
        return mr

    def frequency_weighted_recall(self):
        assert self.accu_matrix is not None, '请先使用add_imgs设置preds与labels'
        frequency_weights = self.accu_matrix.sum(axis=1) / self.accu_matrix.sum()
        cr = self.class_recall()
        fw_precision = (frequency_weights * cr).sum()
        return fw_precision

    def class_IoU(self):
        assert self.accu_matrix is not None, '请先使用add_imgs设置preds与labels'
        inter = np.diag(self.accu_matrix)
        union = self.accu_matrix.sum(axis=1) + self.accu_matrix.sum(axis=0) - np.diag(self.accu_matrix)
        class_IoU = np.divide(inter, union, out=np.zeros_like(inter, dtype=np.float64), where=union!=0) # union不为0的地方才计算，否则填充为0
        return class_IoU

    def macro_IoU(self):
        assert self.accu_matrix is not None, '请先使用add_imgs设置preds与labels'
        class_IoU = self.class_IoU()
        macro_IoU = np.mean(class_IoU)
        return macro_IoU

    def frequency_weighted_IoU(self):
        assert self.accu_matrix is not None, '请先使用add_imgs设置preds与labels'
        frequency_weights = self.accu_matrix.sum(axis=1) / self.accu_matrix.sum()
        class_IoU = self.class_IoU()
        fwIoU = (frequency_weights * class_IoU).sum()
        return fwIoU

    def class_F1score(self):
        assert self.accu_matrix is not None, '请先使用add_imgs设置preds与labels'
        cr = self.class_recall()
        cp = self.class_precision()
        a = (2 * cr * cp)
        b = (cr + cp)
        class_f1 = np.divide(a, b, out=np.zeros_like(a, dtype=np.float64), where=b!=0)
        return class_f1

    def macro_F1score(self):
        assert self.accu_matrix is not None, '请先使用add_imgs设置preds与labels'
        cf1 = self.class_F1score()
        macro_f1 = np.mean(cf1)
        return macro_f1

    def frequency_weighted_F1score(self):
        assert self.accu_matrix is not None, '请先使用add_imgs设置preds与labels'
        frequency_weights = self.accu_matrix.sum(axis=1) / self.accu_matrix.sum()
        cf1 = self.class_F1score()
        fw_f1 = (frequency_weights * cf1).sum()
        return fw_f1


def metric_test():
    # preds = torch.tensor([[[0, 1, 0],
    #                        [2, 1, 0],
    #                        [2, 2, 1]],
    #                       [[0, 1, 0],
    #                        [2, 1, 0],
    #                        [2, 2, 1]]])
    # labels = torch.tensor([[[1, 1, 1],
    #                        [2, 1, 1],
    #                        [2, 2, 1]],
    #                       [[0, 1, 0],
    #                        [2, 1, 0],
    #                        [2, 2, 1]]])
    preds = torch.tensor([[0, 1, 0],
                           [2, 1, 0],
                           [2, 2, 1]])
    labels = torch.tensor([[1, 1, 1],
                           [2, 1, 1],
                           [2, 2, 1]])
    metrics = SegmentationMetrics(3)
    metrics.add_imgs(preds, labels)
    print(metrics.accu_matrix)
    print(metrics.pixel_accuracy())
    print(metrics.class_precision(), metrics.class_recall(), metrics.class_IoU(), metrics.class_F1score())
    print(metrics.macro_precision(), metrics.macro_recall(), metrics.macro_IoU(), metrics.macro_F1score())
    print(metrics.frequency_weighted_precision(), metrics.frequency_weighted_recall(),
          metrics.frequency_weighted_IoU(), metrics.frequency_weighted_F1score())


if __name__ == '__main__':
    metric_test()

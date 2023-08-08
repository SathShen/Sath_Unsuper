

class SoftIoULoss(nn.Module):
    def __init__(self, size_average=True, ignore_index=-100):
        super().__init__()
        self.size_average = size_average

    def forward(self, preds, targets):
        eps = 1e-6
        preds = torch.sigmoid(preds)    # N,Class,W,H    每个点样本有class个概率，每个独热编码样本点有class-1个0和一个1

        inter = preds * targets
        inter = inter.sum(dim=(2, 3))

        union = preds + targets - (preds * targets)
        union = union.sum(dim=(2, 3))

        loss = 1 - inter / (union + eps)
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
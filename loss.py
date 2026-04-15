import torch
import torch.nn as nn

def dice_loss(pred, target, smooth=1):
    pred = torch.sigmoid(pred)
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=(2,3))
    denom = pred.sum(dim=(2,3)) + target.sum(dim=(2,3))

    dice = (2 * intersection + smooth) / (denom + smooth)
    return 1 - dice.mean()

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        d_loss = dice_loss(pred, target)
        return 0.5 * bce_loss + 0.5 * d_loss

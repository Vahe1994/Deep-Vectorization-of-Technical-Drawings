
import torch, torch.nn as nn
from torch.autograd import Variable
import numpy as np
from skimage.measure import compare_psnr

SMOOTH = 1e-6


def IOU(outputs: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W

    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))  # Will be zzero if both are 0

    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds

    return thresholded.mean()


def PSNR(outputs: torch.Tensor, labels: torch.Tensor):
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W

    compare_psnr

    mse = torch.nn.functional.mse_loss(outputs, labels)
    10 * np.log10(1 / mse.item())


class CleaningLoss(nn.Module):
    def __init__(self, kind='MSE', with_restore=True, alpha=1):
        super().__init__()

        if kind == 'MSE':
            self.loss_extraction = nn.MSELoss()
            self.loss_restoration = nn.MSELoss()
        else:
            self.loss_extraction = nn.BCELoss()
            self.loss_restoration = nn.BCELoss()

        self.alpha = alpha
        self.with_restor = with_restore

    def forward(self, y_pred_extract, y_pred_restore, y_true_extract, y_true_restore):
        loss = 0
        y_true_extract = y_true_extract.unsqueeze(1)
        y_true_restore = y_true_restore.unsqueeze(1)
        if self.with_restor:
            loss += self.loss_restoration(y_pred_restore, y_true_restore)

        loss += self.loss_extraction(y_pred_extract, y_true_extract)

        return loss


def MSE_loss(X_batch, y_batch, model=None, device='cpu'):
    X_batch = Variable(torch.FloatTensor(X_batch)).to(device)
    y_batch = Variable(y_batch.type(torch.FloatTensor)).to(device)

    y_batch = y_batch.unsqueeze(1)
    logits, _ = model(X_batch)
    loss_fun = nn.MSELoss()
    return loss_fun(logits, y_batch)


def MSE_synthetic_loss(X_batch, y_batch_er, y_batch_e, model=None, device=None):
    if device is not None:
        X_batch = Variable(torch.FloatTensor(X_batch)).to(device)
        y_batch_er = Variable(y_batch_er.type(torch.FloatTensor)).to(device)
        y_batch_e = Variable(y_batch_e.type(torch.FloatTensor)).to(device)

    y_batch_er = y_batch_er.unsqueeze(1)
    y_batch_e = y_batch_e.unsqueeze(1)

    logits_er, logits_e = model(X_batch)
    loss_first = nn.MSELoss()
    loss_second = nn.MSELoss()

    return loss_first(logits_er, y_batch_er) + loss_second(logits_e, y_batch_e)


def BCE_loss(X_batch, y_batch, model=None, device='cpu'):
    X_batch = Variable(torch.FloatTensor(X_batch)).to(device)
    y_batch = Variable(y_batch.type(torch.FloatTensor)).to(device)

    y_batch = y_batch.unsqueeze(1)
    logits = model(X_batch)
    loss_fun = nn.BCELoss()
    return loss_fun(logits, y_batch)
import torch
import torchvision


def mse_loss(predictions, labels):
    mse = torch.nn.MSELoss()

    loss = mse(predictions, labels)
    return loss


def bce_loss(predictions, labels):
    bce = torch.nn.BCELoss()

    loss = bce(predictions, labels)
    return loss


def bce_loss_with_att(predictions, labels, pred_mask, mask):
    resize = torchvision.transforms.Resize(pred_mask.shape[-2:])

    bce = torch.nn.BCELoss()
    mse = torch.nn.MSELoss()

    loss1 = bce(predictions, labels)
    loss2 = mse(pred_mask[:, 0, :, :], resize(mask))
    return loss1 + loss2


def mse_loss_att(predictions, labels):

    loss = torch.abs(predictions**2 - labels**2)
    loss = torch.mean(loss[labels != 0.5])
    return loss

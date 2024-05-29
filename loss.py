"""
author: mengxue
email: mx.zhang.rs@gmail.com
last date: May 29 2024
"""


import torch
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss


def classification_loss(batch_pred, batch_gt, batch_mask=None):
    if batch_mask is None:
        batch_mask = torch.ones_like(batch_gt)
    CLF = (sigmoid_focal_loss(batch_pred.reshape(-1, 1), batch_gt.reshape(-1, 1), alpha=0.5,
                              gamma=0., reduction='none') * batch_mask.reshape(-1, 1)).sum()
    CLF = CLF / (batch_mask.reshape(-1, 1).sum() + torch.finfo(batch_pred.dtype).eps)
    return CLF


def reconstruction_loss(recon_x, x, x_mask=None):
    if x_mask is None:
        x_mask = torch.ones_like(x)
    MSE = F.mse_loss(recon_x.reshape(-1, 1), x.reshape(-1, 1), reduction='none')
    MSE = (MSE * x_mask.reshape(-1, 1)).sum() / (x_mask.reshape(-1, 1).sum() + torch.finfo(recon_x.dtype).eps)
    return MSE


def kl_loss(mu1, logvar1, mu2=None, logvar2=None):
    if mu2 is None or logvar2 is None:
        # see Appendix B from VAE paper: https://arxiv.org/abs/1312.6114
        KLD = 0.5 * torch.mean(logvar1.exp() + mu1.pow(2) - 1 - logvar1)
    else:
        # Equation 6~7: https://arxiv.org/abs/1606.05908
        x0 = (logvar1 - logvar2).exp()
        x1 = (mu2 - mu1).pow(2) / logvar2.exp()
        x2 = -1.
        x3 = logvar2 - logvar1
        KLD = 0.5 * torch.mean(x0 + x1 + x2 + x3)
    return KLD


CLF_LOSS = classification_loss
REC_LOSS = reconstruction_loss
KL_LOSS = kl_loss
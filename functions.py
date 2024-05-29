import numpy as np
import torch
from loss import CLF_LOSS, REC_LOSS, KL_LOSS
from tools.metrics import compute_classification_metrics as clf_scores


def train(model, train_data_loader, ratio1, ratio2, lr=1e-4, grad_clip=1e-5, device=torch.device("cuda")):
    # Loop for training
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.)
    model.train()

    for batch, (x_dict, label_dict, _) in enumerate(train_data_loader):
        batch_x0 = torch.permute(x_dict['ecv'], (1, 0, 2)).to(device).float()
        batch_x1 = torch.permute(x_dict['idx'], (1, 0, 2)).to(device).float()
        batch_gt = torch.permute(label_dict['y'], (1, 0)).to(device).float()
        batch_mask = torch.permute(label_dict["y_mask"], (1, 0)).to(device).float()

        pred_y, pred_x0, pred_x1, mu0, logvar0, mu1, logvar1 = model(x0=batch_x0, x1=batch_x1)

        # Equation 7
        loss_dk = REC_LOSS(pred_x1, batch_x1) * ratio2[0] + KL_LOSS(mu1, logvar1) * ratio2[1]

        # Equation 8
        loss_ecv = (CLF_LOSS(pred_y, batch_gt, batch_mask) +
                    REC_LOSS(pred_x1, batch_x1) * ratio1[1] +
                    KL_LOSS(mu0, logvar0, mu1.detach(), logvar1.detach()) * ratio1[2])

        # Equation 9
        loss = loss_dk + loss_ecv
        loss.backward()

        """
        clip gradient norm
        https://github.com/pytorch/pytorch/issues/309
        """
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad()


def test(model, dataset, dataloader, n_infer=1, device=torch.device("cuda"), use_mask=True):
    """
    test is for data-driven model
    """
    # Turn on evaluation mode which disables dropout.
    model.eval()
    time_step = dataset.time_step
    all_time = np.array(eval("dataset.x_ecv_time"))[time_step - 1:]
    t_mask = dataset.t_mask[time_step - 1:]
    mask = dataset.mask[time_step - 1:]

    all_gt_ = np.array(dataset.y)
    all_gt_ = all_gt_[:, dataset.all_idx[:, 0], dataset.all_idx[:, 1]]
    all_pred = np.zeros(shape=[t_mask.shape[0], dataset.y.shape[1], dataset.y.shape[2], n_infer], dtype=np.float32)

    with torch.no_grad():
        for batch, (x_dict, _, cord_dict) in enumerate(dataloader):
            batch_x = torch.permute(x_dict['x_ecv'].to(device), (1, 0, 2))
            loc = cord_dict["loc"][:, :].detach().cpu().numpy()

            if n_infer > 1:
                batch_pred = model.infer_with_uncertainty(batch_x, n_infer).squeeze(-2)
            else:
                batch_pred = model.infer(batch_x)

            batch_pred = torch.sigmoid(batch_pred)
            batch_pred = batch_pred.detach().cpu().numpy()
            all_pred[:, loc[:, 0], loc[:, 1]] = batch_pred

    all_pred = all_pred[:, dataset.all_idx[:, 0], dataset.all_idx[:, 1], :]
    all_gt = np.zeros_like(all_pred[..., 0])
    all_gt[mask] = all_gt_

    if use_mask:
        all_pred = all_pred[mask]
        all_gt = all_gt[mask]
        all_time = all_time[mask]

    return_dict = {'preds': all_pred, 'gt': all_gt, 'time': all_time, 'mask': mask,}
    return return_dict


def compute_metrics(all_pred, all_gt, all_time=None, all_mask=None, metrics=None, attachment=False):
    if all_mask is not None:
        all_pred = all_pred[all_mask]
        all_gt = all_gt[all_mask]
        if all_time is not None:
            all_time = all_time[all_mask]
    return_metrics = {
        "pred": all_pred,
        "gt": all_gt,
        "time": all_time,
        "bacc": clf_scores(all_pred.flatten(), all_gt.flatten(), metric_str="acc"),
        "loss_clf": clf_scores(all_pred.flatten(), all_gt.flatten(), metric_str="log_loss"),
        "f1": clf_scores(all_pred.flatten(), all_gt.flatten(), metric_str="f1"),
        "roc": clf_scores(all_pred.flatten(), all_gt.flatten(), metric_str="roc_auc"),
        "pr": clf_scores(all_pred.flatten(), all_gt.flatten(), metric_str="pr_auc"),
    }
    if attachment:
        for k, v in metrics.items():
            if k not in return_metrics.keys:
                return_metrics[k] = v
    return return_metrics
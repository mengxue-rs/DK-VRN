"""
author: mengxue
email: mx.zhang.rs@gmail.com
last date: May 29 2024
"""
import torch
from dk_vrn import DK_VRN
from functions import train, test, compute_metrics
from tools.drought_index import DROUGHT_INDEX

"""
First step: generation of Domain Knowledge (SPEIs) from ECVs and conversion to probability
"""
idx_data = None
if idx_data is not None:
    idx_prob = DROUGHT_INDEX.idx2prob(idx_data)

"""
Second step: ECVs (ERA5), Domain Knowledge (SPEIs), and Ground Truth (EM-DAT) for training
"""
tpb = 256 # number of time step
b_s = 4 # batch size
L1 = 2 # feature number of ERA5
L2 = 3 # feature number of SPEIs

x0 = torch.zeros((tpb, b_s, L1))
x1 = torch.zeros((tpb, b_s, L2))
y = torch.zeros((tpb, b_s))

"""
Third step: network init and training
"""
network = DK_VRN(ninp1=L1, ninp2=L2, nhid=32, en_hid=32, de_hid=32, nout=1)
n_inference = 10 # number of generated outputs (Eq. 14)


lr = 1e-4 # learning rate
grad_clip = 1e-5 # gradient clipping
n_epochs = 200
ratio1 = [1.0, 1.0, 1e-2] # ratio of loss gradients (Eq. 8)
ratio2 = [1.0, 1.0] # ratio of loss gradients (Eq. 7)

train_data = None
val_data = None
test_data = None
train_data_loader = None
val_data_loader = None
test_data_loader = None

if train_data and val_data and train_data_loader and val_data_loader:
    best_val_roc = None
    device = torch.device("cuda")
    save_path = './checkpoints/dk_vrn'

    for epoch in range(1, n_epochs + 1):
        train(network, train_data_loader, ratio1, ratio2, lr, grad_clip, device)
        val_metrics = test(network, val_data, val_data_loader, n_infer=1, device=device)

        if not best_val_roc or (val_metrics["roc"] > best_val_roc):
            best_val_roc = val_metrics["roc"]

            print('save model at epoch ' + str(epoch))
            with open(save_path, 'wb') as f:
                torch.save(network, f)
else:
    print("ECV: {}".format(x0.detach().cpu().numpy().shape))
    print("Index: {}".format(x1.detach().cpu().numpy().shape))
    print("Label: {}".format(y.detach().cpu().numpy().shape))

"""
Last step: network inference
"""
if test_data and test_data_loader:
    device = torch.device("cuda")
    with open(save_path, 'rb') as f:
        trained_network = torch.load(f)
    metrics = test(trained_network, test_data, test_data_loader, n_infer=1, device=device)
    metrics = compute_metrics(metrics["preds"], metrics["gt"], metrics['time'], metrics['mask'])
    time = metrics["time"].tolist()

    print('=' * 89)
    print_str = '| '
    for k, v in metrics.items():
        if k not in ["pred", "gt", "time"]:
            print_str = print_str + ' | ' + k + ' {:5.3f}'.format(metrics[k])
    print(print_str)
    print('=' * 89)

else:
    pred_y = network.infer(x0=x0)
    pred_ny = network.infer_with_uncertainty(x0=x0, n_z=n_inference)
    print("Pred Prob: {}".format(pred_y.detach().cpu().numpy().shape))
    print("Pred Prob with Uncertainty: {}".format(pred_ny.detach().cpu().numpy().shape))

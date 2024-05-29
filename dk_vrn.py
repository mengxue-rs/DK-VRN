"""
author: mengxue
email: mx.zhang.rs@gmail.com
last date: May 29 2024
"""


import torch.nn as nn
import torch


def _reparameterize(mu, logvar, seed=None, ratio=1.0):
    std = torch.exp(0.5 * logvar)
    if seed is not None:
        torch.manual_seed(seed)
    eps = torch.randn_like(std)
    return mu + eps * std * ratio


class DK_VRN(nn.Module):
    def __init__(self, ninp1, ninp2, nhid, 
                 en_hid=None, de_hid=None, nout=None,
                 dropout_rate=0.5, en_layer=2, de_layer=1, use_bn=True):
        super(DK_VRN, self).__init__()
        self.ninp1 = ninp1
        self.ninp2 = ninp2
        self.nhid = nhid
        
        self.en_hid = nhid if en_hid is None else en_hid
        self.de_hid = nhid if de_hid is None else de_hid
        self.nout = ninp1 if nout is None else nout
        
        self.dp = dropout_rate
        self.en_layer = en_layer
        self.de_layer = de_layer
        self.use_bn = use_bn
        
        # Main Branch
        self.embedding0 = LSTM_Module(self.ninp1, self.nhid, self.en_layer, dropout_p=self.dp, bn=self.use_bn)
        self.encode_mu0 = MLP_Module(self.nhid, self.en_hid, bn=False)
        self.encode_logvar0 = MLP_Module(self.nhid, self.en_hid, bn=False)

        self.decode0 = LSTM_Module(self.en_hid, self.de_hid, self.de_layer, dropout_p=self.dp, bn=self.use_bn)
        self.decode1 = MLP_Module(self.de_hid, self.ninp1)
        self.fc = MLP_Module(self.en_hid, self.nout)

        # Domain Knowledge Branch
        self.spei_embedding0 = LSTM_Module(self.ninp2, self.nhid, self.en_layer, dropout_p=self.dp, bn=self.use_bn)
        self.spei_encode_mu = MLP_Module(self.nhid, self.en_hid, bn=False)
        self.spei_encode_logvar = MLP_Module(self.nhid, self.en_hid, bn=False)

        self.spei_decode0 = LSTM_Module(self.en_hid, self.de_hid, self.de_layer, dropout_p=self.dp, bn=self.use_bn)
        self.spei_decode1 = MLP_Module(self.de_hid, self.ninp2)

    def encode(self, x, log_std=True):
        mu = self.encode_mu0(x)
        logvar = self.encode_logvar0(x)
        logvar = logvar if log_std else torch.exp(0.5 * logvar)
        return mu, logvar

    def spei_encode(self, x, log_std=True):
        mu = self.spei_encode_mu(x)
        logvar = self.spei_encode_logvar(x)
        logvar = logvar if log_std else torch.exp(0.5 * logvar)
        return mu, logvar

    def forward(self, x0, x1):
        x0 = self.embedding0(x0)
        mu0, logvar0 = self.encode(x0)
        z0 = _reparameterize(mu0, logvar0)

        x0_ = self.decode0(z0)
        x0_ = self.decode1(x0_)
        y = self.fc(z0)

        x1 = self.spei_embedding0(x1)
        mu1, logvar1 = self.spei_encode(x1)
        z1 = _reparameterize(mu1, logvar1)

        x1_ = self.spei_decode0(z1)
        x1_ = self.spei_decode1(x1_)
        return y, x0_, x1_, mu0, logvar0, mu1, logvar1

    def infer(self, x0):
        x0 = self.embedding0(x0)
        mu0, logvar0 = self.encode(x0)
        z0 = _reparameterize(mu0, logvar0)

        y = self.fc(z0)
        return y

    def infer_with_uncertainty(self, x0, n_z=10, ratio=1.0):
        x = self.embedding0(x0)
        mu0, logvar0 = self.encode(x)
        y_all = []
        for j in range(n_z):
            z0 = _reparameterize(mu0, logvar0, seed=j, ratio=ratio)
            y = self.fc(z0)
            y_all.append(y)

        return torch.stack(y_all, dim=-1)

    def spei_infer(self, x0=None, mu1=None, logvar1=None):
        if x0 is not None:
            x1 = self.spei_embedding0(x0)
            mu1, logvar1 = self.spei_encode(x1)
        z1 = _reparameterize(mu1, logvar1)
        x1 = self.spei_decode1(self.spei_decode0(z1))
        return x1


class LSTM_Module(nn.Module):
    def __init__(self, ninp, nhid, nlayer=1, dropout_p=0.5, bn=False):
        super(LSTM_Module, self).__init__()
        dropout_p = dropout_p if nlayer > 1 else 0.
        self.rnn = nn.LSTM(input_size=ninp, hidden_size=nhid, num_layers=nlayer, dropout=dropout_p)
        self.use_bn = nn.BatchNorm1d(num_features=nhid) if bn else nn.Identity()

    def forward(self, x):
        z, _ = self.rnn(x)
        z = torch.transpose(self.use_bn(torch.transpose(z, dim0=2, dim1=1)), dim0=2, dim1=1)
        return z


class MLP_Module(nn.Module):
    def __init__(self, ninp, nhid, bn=False):
        super(MLP_Module, self).__init__()
        self.fc = nn.Linear(ninp, nhid)
        self.use_bn = nn.BatchNorm1d(num_features=nhid) if bn else nn.Identity()

    def forward(self, x):
        z = self.fc(x)
        if len(z.shape) > 2:
            z = torch.transpose(self.use_bn(torch.transpose(z, dim0=2, dim1=1)), dim0=2, dim1=1)
        else:
            z = self.use_bn(z)
        return z

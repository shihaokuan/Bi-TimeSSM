import torch
import torch.nn as nn
from mamba_ssm import Mamba
from RevIN.RevIN import RevIN
import numpy as np

class Model(torch.nn.Module):
    def __init__(self, configs, corr=None, threshold=0.65):
        super(Model, self).__init__()
        self.configs = configs

        if self.configs.revin == 1:
            self.revin_layer = RevIN(self.configs.enc_in)

        self.lin1 = nn.Linear(self.configs.seq_len, self.configs.n1)
        self.dropout1 = nn.Dropout(self.configs.dropout)

        self.lin2 = nn.Linear(self.configs.n1, self.configs.n2)
        self.dropout2 = nn.Dropout(self.configs.dropout)

        if corr is None:
            corr = np.random.rand(10, 10)

        self.ch_ind = 1

        if self.ch_ind == 1:
            self.d_model_param1 = 1
            self.d_model_param2 = 1
        else:
            self.d_model_param1 = self.configs.n2
            self.d_model_param2 = self.configs.n1

        self.mamba1 = Mamba(d_model=self.d_model_param1, d_state=self.configs.d_state, 
                            d_conv=self.configs.dconv, expand=self.configs.e_fact)
        self.mamba2 = Mamba(d_model=self.configs.n2, d_state=self.configs.d_state, 
                            d_conv=self.configs.dconv, expand=self.configs.e_fact)
        self.mamba3 = Mamba(d_model=self.configs.n1, d_state=self.configs.d_state, 
                            d_conv=self.configs.dconv, expand=self.configs.e_fact)
        self.mamba4 = Mamba(d_model=self.d_model_param2,d_state=self.configs.d_state,
                            d_conv=self.configs.dconv,expand=self.configs.e_fact)

        self.encoder2 = Encoder([
            EncoderLayer(
                Mamba(d_model=configs.n2, d_state=configs.d_state, 
                      d_conv=configs.dconv, expand=configs.e_fact),
                Mamba(d_model=configs.n2, d_state=configs.d_state, 
                      d_conv=configs.dconv, expand=configs.e_fact),
                d_model=configs.n2, d_ff=getattr(configs, "d_ff", 128),
                dropout=configs.dropout, activation=getattr(configs, "activation", "relu"),
                residual=configs.residual == 1
            ) for _ in range(getattr(configs, "e_layers", 4))
        ], norm_layer=nn.LayerNorm(configs.n2))
        
        self.lin3 = nn.Linear(self.configs.n2, self.configs.n1)
        self.lin4 = nn.Linear(2 * self.configs.n1, self.configs.pred_len)

    def forward(self, x):
        if self.configs.revin == 1:
            x = self.revin_layer(x, 'norm')
        else:
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x /= stdev

        x = torch.permute(x, (0, 2, 1))
        if self.ch_ind == 1:
            x = torch.reshape(x, (x.shape[0] * x.shape[1], 1, x.shape[2]))

        x = self.lin1(x)
        x_res1 = x
        x = self.dropout1(x)

        x3=self.mamba3(x)
        if self.ch_ind==1:
            x4=torch.permute(x,(0,2,1))
        else:
            x4=x
        x4=self.mamba4(x4)
        if self.ch_ind==1:
            x4=torch.permute(x4,(0,2,1))
        x4 = x4 + x3

        x = self.lin2(x)
        x_res2 = x
        x = self.dropout2(x)

        x2 = self.encoder2(x)

        if self.configs.residual == 1:
            x = x_res2 + x2
        else:
            x = x2

        x = self.lin3(x)
        if self.configs.residual == 1:
            x = x + x_res1

        x = torch.cat([x, x4], dim=2)
        x = self.lin4(x)
        if self.ch_ind == 1:
            x = torch.reshape(x, (-1, self.configs.enc_in, self.configs.pred_len))

        x = torch.permute(x, (0, 2, 1))

        if self.configs.revin == 1:
            x = self.revin_layer(x, 'denorm')
        else:
            x = x * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.configs.pred_len, 1))
            x = x + (means[:, 0, :].unsqueeze(1).repeat(1, self.configs.pred_len, 1))

        return x

class Add_Norm(nn.Module):
    def __init__(self, d_model, dropout, residual=True):
        super(Add_Norm, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.residual = residual

    def forward(self, new, old=None):
        new = self.dropout(new)
        if self.residual and old is not None:
            new = new + old
        return self.norm(new)

class EncoderLayer(nn.Module):
    def __init__(self, mamba_forward, mamba_backward, d_model, d_ff, dropout, activation="relu", residual=True):
        super(EncoderLayer, self).__init__()
        self.mamba_forward = mamba_forward
        self.mamba_backward = mamba_backward
        self.addnorm_for = Add_Norm(d_model, dropout, residual)
        self.addnorm_back = Add_Norm(d_model, dropout, residual)

        self.ffn = nn.Sequential(
            nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1),
            nn.ReLU() if activation == "relu" else nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        )
        self.addnorm_ffn = Add_Norm(d_model, dropout, residual)

    def forward(self, x):
        output_forward = self.mamba_forward(x)
        output_forward = self.addnorm_for(output_forward, x)

        output_backward = self.mamba_backward(x.flip(dims=[1])).flip(dims=[1])
        output_backward = self.addnorm_back(output_backward, x)

        output = output_forward + output_backward

        temp = output
        output = self.ffn(output.transpose(-1, 1)).transpose(-1, 1)
        output = self.addnorm_ffn(output, temp)
        return output

class Encoder(nn.Module):
    def __init__(self, mamba_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.mamba_layers = nn.ModuleList(mamba_layers)
        self.norm = norm_layer

    def forward(self, x):
        for mamba_block in self.mamba_layers:
            x = mamba_block(x)

        if self.norm is not None:
            x = self.norm(x)

        return x

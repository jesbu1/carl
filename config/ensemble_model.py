import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F
from config.utils import swish, get_affine_params
TORCH_DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class EnsembleModel(nn.Module):
    def __init__(self, ensemble_size, in_features, out_features, hidden_size, num_layers, weight_decays):
        super().__init__()

        self.num_nets = ensemble_size
        self.in_features = in_features
        self.out_features = out_features
        self.num_layers = num_layers
        self.weight_decays = weight_decays

        self.linear_layers = nn.ParameterList()
        self.linear_layers.extend(get_affine_params(ensemble_size, in_features, hidden_size))
        for i in range(num_layers - 2):
            self.linear_layers.extend(get_affine_params(ensemble_size, hidden_size, hidden_size))
        self.linear_layers.extend(get_affine_params(ensemble_size, hidden_size, out_features))

        self.inputs_mu = nn.Parameter(torch.zeros(1, in_features), requires_grad=False)
        self.inputs_sigma = nn.Parameter(torch.zeros(1, in_features), requires_grad=False)

        self.max_logvar = nn.Parameter(torch.ones(1, out_features // 2, dtype=torch.float32) / 2.0)
        self.min_logvar = nn.Parameter(- torch.ones(1, out_features // 2, dtype=torch.float32) * 10.0)

    def compute_decays(self):

        decay = 0
        for layer, weight_decay in zip(self.linear_layers[::2], self.weight_decays):
            decay += weight_decay * (layer ** 2).sum() / 2.0

        return decay

    def fit_input_stats(self, data):

        mu = np.mean(data, axis=0, keepdims=True)
        sigma = np.std(data, axis=0, keepdims=True)
        sigma[sigma < 1e-12] = 1.0

        self.inputs_mu.data = torch.from_numpy(mu).to(TORCH_DEVICE).float()
        self.inputs_sigma.data = torch.from_numpy(sigma).to(TORCH_DEVICE).float()

    def forward(self, inputs, ret_logvar=False):

        # Transform inputs
        # NUM_NETS x BATCH_SIZE X INPUT_LENGTH

        inputs = (inputs - self.inputs_mu) / self.inputs_sigma

        for i, layer in enumerate(zip(self.linear_layers[::2], self.linear_layers[1::2])):
            weight, bias = layer
            inputs = inputs.matmul(weight) + bias
            if i < self.num_layers - 1:
                inputs = swish(inputs)

        mean = inputs[:, :, :self.out_features // 2]

        logvar = inputs[:, :, self.out_features // 2:-1]
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        catastrophe_pred = logvar[..., -1:]
        if ret_logvar:
            return mean, logvar, catastrophe_pred

        return mean, torch.exp(logvar), catastrophe_pred

import numpy as np
import tensorflow as tf
import torch
from torch import nn as nn
TORCH_DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def swish(x):
    return x * torch.sigmoid(x)

def truncated_normal(size, std):
    # We use TF to implement initialization function for neural network weight because:
    # 1. Pytorch doesn't support truncated normal
    # 2. This specific type of initialization is important for rapid progress early in training in cartpole

    # Do not allow tf to use gpu memory unnecessarily
    cfg = tf.ConfigProto()
    cfg.gpu_options.allow_growth = True

    sess = tf.Session(config=cfg)
    val = sess.run(tf.truncated_normal(shape=size, stddev=std))

    # Close the session and free resources
    sess.close()

    return torch.tensor(val, dtype=torch.float32)


def get_affine_params(ensemble_size, in_features, out_features):

    w = truncated_normal(size=(ensemble_size, in_features, out_features),
                         std=1.0 / (2.0 * np.sqrt(in_features)))
    w = nn.Parameter(w)

    b = nn.Parameter(torch.zeros(ensemble_size, 1, out_features, dtype=torch.float32))

    return w, b

def get_affine_params_4d(ensemble_size, in_features_1, in_features_2, out_features):

    w = truncated_normal(size=(ensemble_size, in_features_1, in_features_2, out_features),
                         std=1.0 / (2.0 * np.sqrt(in_features_1*in_features_2)))
    w = nn.Parameter(w)

    b = nn.Parameter(torch.zeros(ensemble_size, 1, 1, out_features, dtype=torch.float32))

    return w, b

def reparameterize(mu, logvar):
    logvar = torch.clamp(logvar, -20, 20)
    std = torch.exp(0.5 * logvar)
    #std = torch.clamp(std, 0, 100)
    eps = torch.randn_like(std)
    ret = eps.mul(std) + mu
    return ret

def kl_loss(mu, logvar):
    #if torch.any(mu > 1000): import pdb; pdb.set_trace()
    logvar = torch.clamp(logvar, -20, 20)
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(-1).mean(-1).sum(-1)
    return kl

def torch_truncated_normal(size, mean=0, std=1):
    tensor = torch.zeros(size, device=TORCH_DEVICE)
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor

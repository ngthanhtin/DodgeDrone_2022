"""This is OpenAI' Spinning Up PyTorch implementation of Soft-Actor-Critic.

For the official documentation, see below:
https://spinningup.openai.com/en/latest/algorithms/sac.html#documentation-pytorch-version

Source:
https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/core.py
"""
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


LOG_STD_MAX = 2
LOG_STD_MIN = -20

from stable_baselines3.common.utils import get_device

DEVICE = get_device('auto')

class SquashedGaussianMLPActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit, act_bias):
        super().__init__()
        
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = torch.Tensor(act_limit).to(DEVICE)
        self.act_bias = torch.Tensor(act_bias).to(DEVICE)

    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        # Pre-squash distribution and sample
        try:
            pi_distribution = Normal(mu, std)
        except ValueError:
            pdb.set_trace()
            pass

        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()
        
        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            tmp = 2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))
            tmp = tmp.sum(axis=(2 if len(tmp.shape) > 1 else -1))
            
            logp_pi -= tmp
        else:
            logp_pi = None
        
        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action + self.act_bias
        
        return pi_action, logp_pi



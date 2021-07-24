import numpy as np
import torch
import torch.distributions
import torch.nn as nn
import torch.nn.functional as F

from dreamer.models.distribution import TanhBijector, SampleDist


class Action(nn.Module):
    def __init__(self, action_size, feature_size, hidden_size, layers, dist='tanh_normal',
                 activation=nn.ELU, min_std=1e-4, init_std=5, mean_scale=5):
        super().__init__()

    def build_model(self):
        pass

    def forward(self, state_features):
        pass

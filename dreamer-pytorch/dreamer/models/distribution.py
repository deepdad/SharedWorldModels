import torch
import torch.nn.functional as F
import torch.distributions
from torch.distributions import constraints
import numpy as np

class TanhBijector(torch.distributions.Transform):
    def __init__(self):
        super().__init__()
        self.bijective = True

# domain = self.parts[0].domain

    @property
    def sign(self):
        return 1.

    def _call(self, x):
        return torch.tanh(x)

    def _inverse(self, y: torch.Tensor):
        y = torch.where(
            (torch.abs(y) <= 1.),
            torch.clamp(y, -0.99999997, 0.99999997),
            y
        )

        y = atanh(y)
        return y

    def log_abs_det_jacobian(self, x, y):
        return 2. * (np.log(2) - x - F.softplus(-2. * x))


# class ReshapeTransform(Transform):
    @constraints.dependent_property
    def domain(self):
#        return constraints.independent(constraints.real, len(self.in_shape))
        return constraints.real  # it may actually be the open interval (-1, 1)

    @constraints.dependent_property
    def codomain(self):
        return constraints.real  # need to verify, this is just a placeholder
        # return constraints.independent(constraints.real, len(self.out_shape))

# class AffineTransform(Transform):
#    @constraints.dependent_property(is_discrete=False)
#    def domain(self):
#        if self.event_dim == 0:
#            return constraints.real
#        return constraints.independent(constraints.real, self.event_dim)

class SampleDist:

    def __init__(self, dist: torch.distributions.Distribution, samples=100):
        self._dist = dist
        self._samples = samples
        self._validate_args = None
        self.validate_args = None
   # super(Normal, self).__init__(batch_shape, validate_args=validate_args)

    @property
    def name(self):
        return 'SampleDist'

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def mean(self):
        dist = self._dist.expand((self._samples, *self._dist.batch_shape))
        sample = dist.rsample()
        return torch.mean(sample, 0)

    def mode(self):
        dist = self._dist.expand((self._samples, *self._dist.batch_shape))
        sample = dist.rsample()
        logprob = dist.log_prob(sample)
        batch_size = sample.size(1)
        feature_size = sample.size(2)
        indices = torch.argmax(logprob, dim=0).reshape(1, batch_size, 1).expand(1, batch_size, feature_size)
        return torch.gather(sample, 0, indices).squeeze(0)

    def entropy(self):
        dist = self._dist.expand((self._samples, *self._dist.batch_shape))
        sample = dist.rsample()
        logprob = dist.log_prob(sample)
        return -torch.mean(logprob, 0)

    def sample(self):
        return self._dist.sample()


def atanh(x):
    return 0.5 * torch.log((1 + x) / (1 - x))

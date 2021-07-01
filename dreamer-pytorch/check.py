pcont_pred:  Independent(Bernoulli(logits: torch.Size([5, 1, 1])), 1)

   def _validate_sample(self, value):
        """
        Argument validation for distribution methods such as `log_prob`,
        `cdf` and `icdf`. The rightmost dimensions of a value to be
        scored via these methods must agree with the distribution's batch
        and event shapes.

        Args:
            value (Tensor): the tensor whose log probability is to be
                computed by the `log_prob` method.
        Raises
            ValueError: when the rightmost dimensions of `value` do not match the
                distribution's batch and event shapes.
        """
        event_dim_start = len(value.size()) - len(self._event_shape)
        if value.size()[event_dim_start:] != self._event_shape:
            raise ValueError('The right-most size of value must match event_shape: {} vs {}.'.
                             format(value.size(), self._event_shape))

        actual_shape = value.size()
        expected_shape = self._batch_shape + self._event_shape
        for i, j in zip(reversed(actual_shape), reversed(expected_shape)):
            if i != 1 and j != 1 and i != j:
                raise ValueError('Value is not broadcastable with batch_shape+event_shape: {} vs {}.'.
                                 format(actual_shape, expected_shape))
        try:
            support = self.support
        except NotImplementedError:
            warnings.warn(f'{self.__class__} does not define `support` to enable ' +
                          'sample validation. Please initialize the distribution with ' +
                          '`validate_args=False` to turn off validation.')
            return
        assert support is not None
        if not support.check(value).all():
>           raise ValueError('The value argument must be within the support')
E           ValueError: The value argument must be within the support


(venv) dry@dry-VM:~/DeepLearningLab2021/Project/dreamer-pytorch$ python
Python 3.8.5 (default, May 27 2021, 13:30:53) 
[GCC 9.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
>>> m = Bernouilli()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'Bernouilli' is not defined
>>> m = torch.distributions.bernoulli.Bernoulli(torch.tensor([0.3]))
>>> m.sample()
tensor([0.])
>>> m.log_prob(m.sample())
tensor([-0.3567])
>>> s = torch.Tensor(0.9)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: new(): data must be a sequence (got float)
>>> s = torch.Tensor([0.9])
>>> s
tensor([0.9000])
>>> m = torch.distributions.bernoulli.Bernoulli(logits: torch.Size([5, 1, 1])), 1)
  File "<stdin>", line 1
    m = torch.distributions.bernoulli.Bernoulli(logits: torch.Size([5, 1, 1])), 1)
                                                      ^
SyntaxError: invalid syntax
>>> m = torch.distributions.bernoulli.Bernoulli(logits=torch.Size([5, 1, 1])), 1)
  File "<stdin>", line 1
    m = torch.distributions.bernoulli.Bernoulli(logits=torch.Size([5, 1, 1])), 1)
                                                                                ^
SyntaxError: unmatched ')'
>>> m = torch.distributions.bernoulli.Bernoulli(logits=torch.Size([5, 1, 1]))
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/home/dry/venv/lib/python3.8/site-packages/torch/distributions/bernoulli.py", line 42, in __init__
    self.logits, = broadcast_all(logits)
  File "/home/dry/venv/lib/python3.8/site-packages/torch/distributions/utils.py", line 29, in broadcast_all
    raise ValueError('Input arguments must all be instances of numbers.Number, '
ValueError: Input arguments must all be instances of numbers.Number, torch.Tensor or objects implementing __torch_function__.
>>> torch.Size([5,1,1])
torch.Size([5, 1, 1])
>>> p = torch.Tensor(torch.Size([5,1,1]))
>>> p
tensor([[[1.5995e-36]],

        [[0.0000e+00]],

        [[6.8664e-44]],

        [[7.0065e-44]],

        [[1.5978e-36]]])
>>> m = torch.distributions.bernoulli.Bernoulli(p)
>>> m
Bernoulli(probs: torch.Size([5, 1, 1]))
>>> m.log_prob(s)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/home/dry/venv/lib/python3.8/site-packages/torch/distributions/bernoulli.py", line 93, in log_prob
    self._validate_sample(value)
  File "/home/dry/venv/lib/python3.8/site-packages/torch/distributions/distribution.py", line 277, in _validate_sample
    raise ValueError('The value argument must be within the support')
ValueError: The value argument must be within the support
>>> s
tensor([0.9000])
>>> p.support
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: 'Tensor' object has no attribute 'support'
>>> m.support
Boolean()
>>> m.log_prob(True)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/home/dry/venv/lib/python3.8/site-packages/torch/distributions/bernoulli.py", line 93, in log_prob
    self._validate_sample(value)
  File "/home/dry/venv/lib/python3.8/site-packages/torch/distributions/distribution.py", line 255, in _validate_sample
    raise ValueError('The value argument to log_prob must be a Tensor')
ValueError: The value argument to log_prob must be a Tensor
>>> m.log_prob(torch.Tensor([True]))
tensor([[[-15.9424]],

        [[-15.9424]],

        [[-15.9424]],

        [[-15.9424]],

        [[-15.9424]]])
>>> m.log_prob(torch.Tensor([0.999]))
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/home/dry/venv/lib/python3.8/site-packages/torch/distributions/bernoulli.py", line 93, in log_prob
    self._validate_sample(value)
  File "/home/dry/venv/lib/python3.8/site-packages/torch/distributions/distribution.py", line 277, in _validate_sample
    raise ValueError('The value argument must be within the support')
ValueError: The value argument must be within the support
>>> m.log_prob(torch.Tensor([True, False, True, False]))
tensor([[[-1.5942e+01, -1.1921e-07, -1.5942e+01, -1.1921e-07]],

        [[-1.5942e+01, -1.1921e-07, -1.5942e+01, -1.1921e-07]],

        [[-1.5942e+01, -1.1921e-07, -1.5942e+01, -1.1921e-07]],

        [[-1.5942e+01, -1.1921e-07, -1.5942e+01, -1.1921e-07]],

        [[-1.5942e+01, -1.1921e-07, -1.5942e+01, -1.1921e-07]]])
>>> m.log_prob(torch.Tensor([[True, False, True, False], [False, False, False, False]]))
tensor([[[-1.5942e+01, -1.1921e-07, -1.5942e+01, -1.1921e-07],
         [-1.1921e-07, -1.1921e-07, -1.1921e-07, -1.1921e-07]],

        [[-1.5942e+01, -1.1921e-07, -1.5942e+01, -1.1921e-07],
         [-1.1921e-07, -1.1921e-07, -1.1921e-07, -1.1921e-07]],

        [[-1.5942e+01, -1.1921e-07, -1.5942e+01, -1.1921e-07],
         [-1.1921e-07, -1.1921e-07, -1.1921e-07, -1.1921e-07]],

        [[-1.5942e+01, -1.1921e-07, -1.5942e+01, -1.1921e-07],
         [-1.1921e-07, -1.1921e-07, -1.1921e-07, -1.1921e-07]],

        [[-1.5942e+01, -1.1921e-07, -1.5942e+01, -1.1921e-07],
         [-1.1921e-07, -1.1921e-07, -1.1921e-07, -1.1921e-07]]])
>>> 

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
            raise ValueError('The value argument must be within the support')

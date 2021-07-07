from dreamer.agents.dreamer_agent import DreamerAgent
from dreamer.models.agent import AtariDreamerModel


class BenchmarkDreamerAgent(DreamerAgent):
    """
    This used to be called DMCDreamerAgent, but RLBench environments have the same
    API. The naming is a little confusing; here, AtariDreamerModel is imported.
    """
    def __init__(self, ModelCls=AtariDreamerModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)

    def make_env_to_model_kwargs(self, env_spaces):
        return dict(image_shape=env_spaces.observation.shape,
                    output_size=env_spaces.action.shape[0],
                    action_shape=env_spaces.action.shape[0],
                    action_dist='tanh_normal')

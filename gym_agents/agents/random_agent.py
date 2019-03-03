from .base_agent import BaseAgent


class RandomAgent(BaseAgent):
    def __init__(self, action_space, observation_space):
        super().__init__(action_space, observation_space)

    def act(self, observation, reward, done):
        return self.action_space.sample()

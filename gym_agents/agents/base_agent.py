from abc import ABC, abstractmethod


class BaseAgent(ABC):
    def __init__(self, action_space, *args, **kwargs):
        self.action_space = action_space
        super().__init__()

    @abstractmethod
    def act(self, observation, reward, done):
        pass

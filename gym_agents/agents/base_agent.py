from abc import ABC, abstractmethod
from collections import deque


class BaseAgent(ABC):
    def __init__(self, action_space, observation_space, *args, **kwargs):
        self.action_space = action_space
        self.observation_space = observation_space
        self.memory = deque(maxlen=2000)
        super().__init__()

    @abstractmethod
    def act(self, observation, reward, done):
        pass

    def remember(self, state, action, reward, next_state, done):
        """
        Remember the effect that taking `action` on `state` had.
        :param state: The state on which the action was taken
        :param action: The action that was taken
        :param reward: The given reward by doing `action` on `state`
        :param next_state: The resulting state, after taking `action` on `state`
        :param done: Whether taking `action` on `state` ended the game
        :return:
        """
        self.memory.append((state, action, reward, next_state, done))

    def save(self):
        """
        Save the model of the agent
        """
        pass

    def load(self):
        """
        Load the model of the agent
        """
        pass

    def step_done(self, step_num):
        """
        Step has been taken, update as needed
        :return:
        """
        pass

    def episode_done(self, episode_num):
        """
        Episode has finished, update as needed
        :return:
        """
        pass

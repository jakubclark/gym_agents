import random
from collections import deque
from logging import getLogger

import numpy as np
from click import echo
from keras import Sequential
from keras.layers.core import Dense
from keras.optimizers import Adam

from ..util import flatten_shape
from .base_agent import BaseAgent

log = getLogger(__name__)


class DQNAgent(BaseAgent):
    def __init__(self, action_space, observation_space, *args, **kwargs):
        self.state_size = flatten_shape(observation_space)
        self.action_size = action_space.n
        self.memory = deque(maxlen=2000)
        self.gamma = kwargs.pop('gamma', 0.95)
        self.epsilon = kwargs.pop('epsilon', 1)
        self.epsilon_min = kwargs.pop('epsilon_min', 0.01)
        self.epsilon_decay = kwargs.pop('epsilon_decay', 0.99)
        self.learning_rate = kwargs.pop('learning_rate', 1e-3)
        self.batch_size = kwargs.pop('batch_size', 32)

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

        if kwargs.get('model', None) is not None:
            self.load(kwargs.pop('model', None))

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, reward, done):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self):
        sub_batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in sub_batch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * np.amax(t)
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        echo(f'Loading {name} dqn model')
        self.model.load_weights(name)

    def save(self, name):
        echo(f'Saving dqn model to {name}')
        self.model.save_weights(name)

    def step_done(self, step_num):
        if len(self.memory) > self.batch_size:
            self.replay()

    def episode_done(self, episode_num):
        if episode_num % 10 == 0:
            self.save(f'save/dqn_agent{episode_num}')
        self.target_model.set_weights(self.model.get_weights())

import json
from logging import INFO, basicConfig, getLogger

import click
import numpy as np

from . import flatten_shape
from .agents import create_agent
from .envs import create_env

basicConfig(filename='gym_agents.log',
            level=INFO,
            filemode='w',
            format='%(asctime)s %(name)-12s %(levelname)-8s %(funcName)s %(message)s',
            datefmt='%d-%m %H:%M:%S')
log = getLogger(__name__)


class Runner:

    def __init__(self, load_model_path, agent_id, environment_id, num_steps,
                 train_starts, save_freq, update_freq, train_freq, config):
        self.load_model_path = load_model_path
        self.agent_id = agent_id
        self.environment_id = environment_id
        self.num_steps = num_steps
        self.train_starts = train_starts
        self.save_freq = save_freq
        self.update_freq = update_freq
        self.train_freq = train_freq

        self.env = create_env(self.environment_id)
        self.agent = create_agent(self.agent_id,
                                  self.env.action_space,
                                  self.env.observation_space,
                                  **config)

        self.state_size = flatten_shape(self.env.observation_space)

        self.train_episode_rewards = [0.0]
        self.test_episode_rewards = [0.0]

        self.train_epsilons = []

        self.train_episode_steps = [0]
        self.test_episode_steps = [0]

        self.saved_mean = -500
        self.saved_means = []
        self.model_file_path = load_model_path or f'models/{self.environment_id}-{self.agent_id}.model'

        click.echo(self.model_file_path)

    def play_training_games(self):
        for epi in range(self.train_starts):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            reward = 0
            done = False
            while not done:
                action = self.env.action_space.sample()
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])
                self.agent.remember(state, action, reward, next_state, done)

        with click.progressbar(range(self.num_steps)) as bar:
            self._play_training_games(bar)

    def _play_training_games(self, bar):
        state = self.env.reset()
        state = np.reshape(state, [1, self.state_size])
        reward = 0
        done = False

        for step in bar:
            self.train_episode_steps[-1] += 1

            # Action part
            action = self.agent.act(state, reward, done)
            next_state, reward, done, info = self.env.step(action)
            next_state = np.reshape(next_state, [1, self.state_size])

            # Training part
            self.agent.remember(state, action, reward, next_state, done)

            self.train_episode_rewards[-1] += reward
            state = next_state

            if step % self.train_freq == 0:
                self.agent.step_done(step)

            if done:
                n_episodes_mean = np.mean(
                    self.train_episode_rewards[-self.save_freq + 1:])

                epi = len(self.train_episode_rewards)

                if epi % self.save_freq == 0 and n_episodes_mean > self.saved_mean:
                    s = (f'Saving model due to increase in mean reward, over the last '
                         f'{self.save_freq} episodes: {self.saved_mean}->{n_episodes_mean}')
                    click.echo(f'\n{s}')
                    log.info(s)
                    self.agent.save(self.model_file_path)
                    self.saved_mean = n_episodes_mean
                    self.saved_means.append({
                        'episode_num': epi,
                        f'{self.save_freq}_episode_mean': self.saved_mean
                    })

                last_episode_reward = self.train_episode_rewards[-1]
                log.info(
                    (
                        f'Episode: {epi}, '
                        f'Episode Score: {last_episode_reward}, '
                        f'Step: {step}/{self.num_steps}, '
                        f'Mean from last {self.save_freq} episodes: {n_episodes_mean}')
                )

                if epi % self.update_freq == 0:
                    self.agent.episode_done(epi)

                state = self.env.reset()
                state = np.reshape(state, [1, self.state_size])
                self.train_episode_rewards.append(0.0)
                self.train_epsilons.append(self.agent.epsilon)
                self.train_episode_steps.append(0)
                continue

    def play_testing_games(self, display=False):
        click.echo(f'Restoring best performing model')
        self.agent.load(self.model_file_path)

        state, reward, done = self.reset_env()

        for i in range(100):
            while not done:
                action = self.agent.act_model(state, reward, done)
                state, reward, done, _ = self.env.step(action)
                state = np.reshape(state, [1, self.state_size])

                self.test_episode_rewards[-1] += reward
                self.test_episode_steps[-1] += 1
                if display:
                    self.env.render()

            epi = len(self.test_episode_rewards)
            score = self.test_episode_rewards[-1]
            s = f'Test Episode: {epi}/100, Score: {score}'
            click.echo(s)
            log.info(s)

            self.test_episode_rewards.append(0.0)
            self.test_episode_steps.append(0)

            state, reward, done = self.reset_env()

    def reset_env(self):
        state = np.reshape(self.env.reset(), [1, self.state_size])
        reward = 0
        done = False
        return state, reward, done

    @property
    def config(self) -> dict:
        return {
            'runner_config': {
                'loaded_model': self.load_model_path,
                'agent_id': self.agent_id,
                'environment_id': self.environment_id,
                'num_steps': self.num_steps,
                'train_starts': self.train_starts,
                'save_freq': self.save_freq,
                'update_freq': self.update_freq,
                'state_size': self.state_size,
                'saved_mean': self.saved_mean,
                'saved_means': self.saved_means,
                'saved_model': self.model_file_path
            },
            'agent_config': {
                'initial': self.agent.initial_config,
                'final': self.agent.status
            },
            'agent_performance': self.performance,
            'data': {
                'train_episode_rewards': self.train_episode_rewards[:-1],
                'train_episode_epsilons': self.train_epsilons[:-1],
                'train_episode_steps': self.train_episode_steps[:-1]
            },
            'data_test': {
                'test_episode_rewards': self.test_episode_rewards[:-1],
                'test_episode_steps': self.test_episode_steps[:-1]
            },
            'agent_history': self.agent.history
        }

    @property
    def performance(self) -> dict:
        return {
            'train_average_reward': np.mean(self.train_episode_rewards[:-1]),
            'test_average_reward': np.mean(self.test_episode_rewards[:-1]),
            'train_average_steps': np.mean(self.train_episode_steps[:-1]),
            'test_average_steps': np.mean(self.test_episode_steps[:-1]),
            'train_games_played': len(self.train_episode_rewards) - 1,
            'test_games_played': len(self.test_episode_rewards) - 1
        }

    def save_config(self, filename=None):
        filename = filename or f'{self.environment_id}-{self.agent_id}-config_performance.json'
        with open(filename, 'w') as fh:
            json.dump(self.config, fh, indent=2)

import json
from logging import INFO, basicConfig, getLogger

import click
import numpy as np

from .agents import agents
from .envs import create_env
from .util import flatten_shape

basicConfig(filename='gym_agents.log',
            level=INFO,
            filemode='w',
            format='%(asctime)s %(name)-12s %(levelname)-8s %(funcName)s %(message)s',
            datefmt='%d-%m %H:%M:%S')
log = getLogger(__name__)


@click.group(invoke_without_command=True)
@click.option('--model_path', '-m', default=None, help='Path to agent\'s model')
@click.option('--agent_id', '-a', default='RandomAgent', type=str, help='The agent id to use.')
@click.option('--environment_id', '-e', default='Breakout-v0', type=str, help='The environment id to use.')
@click.option('--num_steps', '-s', default=10000, type=int, help='Number of steps to run per episode')
@click.option('--train_starts', default=50, type=int, help='Number of episodes to run before training actually begins.')
@click.option('--save_freq', default=10, type=int, help='Number of episodes to run in between potential model saving')
@click.option('--update_freq', default=20, type=int, help='Number of episodes to run in between target model updates')
@click.option('--train_freq', default=30, type=int, help='Number of episodes in between model training')
@click.option('--play', is_flag=True, help='Have the agent play the game, without training')
@click.pass_context
def main(ctx, model_path, agent_id, environment_id, num_steps,
         train_starts, save_freq, update_freq, train_freq, play):
    if ctx.invoked_subcommand is not None:
        return

    runner = Runner(model_path, agent_id, environment_id, num_steps,
                    train_starts, save_freq, update_freq, train_freq)

    if play:
        runner.play_testing_games()
        runner.save_config()
        return

    runner.play_training_games()
    runner.play_testing_games()
    runner.save_config()


@main.command()
def list_agents():
    res = [k for k in agents.keys()]
    click.echo(res)


@main.command()
def list_environments():
    from gym import envs
    envids = [spec.id for spec in envs.registry.all()]
    click.echo(envids)


def play_game(env, agent):
    state_size = flatten_shape(env.observation_space)
    episode_reward = 0
    state, reward, done = env.reset(), 0, False
    state = np.reshape(state, [1, state_size])
    while True:
        action = agent.act_model(state, reward, done)
        state, reward, done, _ = env.step(action)
        state = np.reshape(state, [1, state_size])
        episode_reward += reward
        env.render()
        if done:
            click.echo(f'Episode reward: {episode_reward}')
            episode_reward = 0
            state = env.reset()
            state = np.reshape(state, [1, state_size])


class Runner:

    def __init__(self, load_model_path, agent_id, environment_id, num_steps,
                 train_starts, save_freq, update_freq, train_freq):
        self.load_model_path = load_model_path
        self.agent_id = agent_id
        self.environment_id = environment_id
        self.num_steps = num_steps
        self.train_starts = train_starts
        self.save_freq = save_freq
        self.update_freq = update_freq
        self.train_freq = train_freq

        self.env = create_env(self.environment_id)
        self.agent = agents[self.agent_id](self.env.action_space,
                                           self.env.observation_space)

        self.state_size = flatten_shape(self.env.observation_space)

        self.train_episode_rewards = [0.0]
        self.test_episode_rewards = [0.0]

        self.saved_mean = -500
        self.model_file_path = f'models/{self.environment_id}-{self.agent_id}.model'

        if load_model_path:
            self.agent.load(load_model_path)

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

        state = self.env.reset()
        state = np.reshape(state, [1, self.state_size])
        reward = 0
        done = False

        for step in range(self.num_steps):

            # Action part
            action = self.agent.act(state, reward, done)
            next_state, reward, done, info = self.env.step(action)
            next_state = np.reshape(next_state, [1, self.state_size])

            # Training part
            self.agent.remember(state, action, reward, next_state, done)

            self.train_episode_rewards[-1] += reward
            state = next_state

            if epi % self.train_freq == 0:
                self.agent.step_done(step)

            if done:
                n_episodes_mean = np.mean(
                    self.train_episode_rewards[-self.save_freq+1:-1])

                epi = len(self.train_episode_rewards)

                if epi % self.save_freq == 0 and n_episodes_mean > self.saved_mean:
                    s = (f'Saving model due to increase in mean reward, over the last '
                         f'{self.save_freq} episodes: {self.saved_mean}->{n_episodes_mean}')
                    click.echo(s)
                    log.info(s)
                    self.agent.save(self.model_file_path)
                    self.saved_mean = n_episodes_mean

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
                continue

    def play_testing_games(self):
        click.echo(f'Restoring best performing model')
        self.agent.load(self.model_file_path)

        state, reward, done = self.reset_env()

        for i in range(100):
            while not done:
                action = self.agent.act_model(state, reward, done)
                state, reward, done, _ = self.env.step(action)
                state = np.reshape(state, [1, self.state_size])

                self.test_episode_rewards[-1] += reward

            epi = len(self.test_episode_rewards)
            score = self.test_episode_rewards[-1]
            s = f'Test Episode: {epi}/100, Score: {score} '
            click.echo(s)
            log.info(s)

            self.test_episode_rewards.append(0.0)

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
                'train_episode_rewards': self.train_episode_rewards,
                'test_episode_rewards': self.test_episode_rewards,
                'saved_mean': self.saved_mean,
                'saved_model': self.model_file_path
            },
            'agent_config': self.agent.status,
            'agent_performance': self.performance
        }

    @property
    def performance(self) -> dict:
        return{
            'train_average': np.mean(self.train_episode_rewards),
            'test_average': np.mean(self.test_episode_rewards),
            'training_games_played': len(self.train_episode_rewards),
            'test_games_played': len(self.test_episode_rewards)
        }

    def save_config(self):
        with open(f'{self.environment_id}-{self.agent_id}-config_performance.json', 'w') as fh:
            json.dump(self.config, fh, indent=2)

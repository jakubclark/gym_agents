from logging import INFO, basicConfig, getLogger

import click
import numpy as np

from .agents import agents
from .envs import create_env
from .util import flatten_shape

basicConfig(filename='gym_agents.log',
            level=INFO,
            filemode='w')
log = getLogger(__name__)


@click.group(invoke_without_command=True)
@click.option('--display', '-d', is_flag=True, help='Display the game as it\'s running.')
@click.option('--load', '-l', default=None, help='Load the agent\'s model')
@click.option('--agent_id', '-a', default='RandomAgent', type=str, help='The agent id to use.')
@click.option('--environment_id', '-e', default='Breakout-v0', type=str, help='The environment id to use.')
@click.option('--num_episodes', '-n', default=500, type=int, help='Number of episodes to run.')
@click.option('--num_steps', '-s', default=500, type=int, help='Number of steps to run per episode')
@click.option('--train_starts', default=50, type=int, help='Number of episodes to run before training actually begins.')
@click.option('--save_freq', default=10, type=int, help='Number of episodes to run in between potential model saving')
@click.option('--update_freq', default=1, type=int, help='Number of episodes to run in between model updates')
@click.option('--train_freq', default=5, type=int, help='Number of episodes in between model training')
@click.option('--play', is_flag=True, help='Have the agent play the game, without training')
@click.pass_context
def main(ctx, display, load, agent_id, environment_id, num_episodes,
         num_steps, train_starts, save_freq, update_freq, train_freq, play):
    if ctx.invoked_subcommand is not None:
        return

    click.echo(f'Creating {environment_id} environment.')
    env = create_env(environment_id)

    state_size = flatten_shape(env.observation_space)

    click.echo(f'Creating {agent_id} agent.')
    agent = agents[agent_id](
        env.action_space, env.observation_space,
        model=load)

    if play:
        play_game(env, agent)
        return

    reward = 0
    done = False

    episode_rewards = []
    saved_mean = -500

    model_file_path = f'models/{environment_id}-{agent_id}.model'

    # Take random actions, and remember the consequences
    for epi in range(train_starts):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        done = False
        while not done:
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)

    for epi in range(num_episodes):
        click.echo(f'Running episode number {epi}')
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        episode_rewards.append(0.0)
        for step in range(num_steps):
            if display:
                env.render()

            # Action part
            action = agent.act(state, reward, done)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            # Training part
            agent.remember(state, action, reward, next_state, done)

            reward = -10 if done else reward
            episode_rewards[-1] += reward
            state = next_state

            if epi % train_freq == 0:
                agent.step_done(step)

            if done:
                n_episodes_mean = np.mean(episode_rewards[-save_freq+1:-1])

                if epi % save_freq == 0 and n_episodes_mean > saved_mean:
                    s = f'Saving model due to increase in mean reward: {saved_mean}->{n_episodes_mean}'
                    click.echo(s)
                    log.info(s)
                    agent.save(model_file_path)
                    saved_mean = n_episodes_mean

                log.info(
                    f'Episode: {epi}/{num_episodes}, Score: {episode_rewards[-1]}, Mean from last {save_freq} episodes: {n_episodes_mean}')

                if epi % update_freq == 0:
                    agent.episode_done(epi)

                break
    agent.load(model_file_path)
    play_game(env, agent)
    env.close()


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

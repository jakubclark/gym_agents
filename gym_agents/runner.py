from logging import INFO, basicConfig, getLogger

import click
import gym
import numpy as np

from .agents import agents
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
@click.option('--save_freq', default=10, type=int,
              help='Number of episodes to run in between potential model saving')
@click.pass_context
def main(ctx, display, load, agent_id, environment_id, num_episodes, num_steps, train_starts, save_freq):
    if ctx.invoked_subcommand is not None:
        return

    log.info(f'Creating {environment_id} environment.')
    env = gym.make(environment_id)

    state_size = flatten_shape(env.observation_space)

    log.info(f'Creating {agent_id} agent.')
    agent = agents[agent_id](
        env.action_space, env.observation_space,
        model=load)

    reward = 0
    done = False

    episode_rewards = []
    saved_mean = -500

    for epi in range(num_episodes):
        log.info(f'Running episode number {epi}')
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

            if epi >= train_starts:
                if done:
                    n_episodes_mean = np.mean(episode_rewards[-save_freq+1:-1])

                    if epi % save_freq == 0 and n_episodes_mean > saved_mean:
                        s = f'Saving model due to increase in mean reward: {saved_mean}->{n_episodes_mean}'
                        click.echo(s)
                        log.info(s)
                        agent.save(f'models/{environment_id}-{agent_id}.model')
                        saved_mean = n_episodes_mean

                    log.info(
                        f'Episode: {epi}/{num_episodes}, Score: {episode_rewards[-1]}')
                    agent.episode_done(epi)
                agent.step_done(step)

            if done:
                break
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

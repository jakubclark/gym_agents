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
@click.option('--num_episodes', '-n', default=100, type=int, help='Number of episodes to run.')
@click.option('--num_steps', '-s', default=500, type=int, help='Number of steps to run per episode')
@click.pass_context
def main(ctx, display, load, agent_id, environment_id, num_episodes, num_steps):
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

    for epi in range(num_episodes):
        log.info(f'Running episode number {epi}')
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        total_reward = 0

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
            total_reward += reward

            state = next_state

            if done:
                agent.episode_done(epi)
                log.info(
                    f'Episode: {epi}/{num_episodes}, score: {total_reward}')
                break

            agent.step_done(step)
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

from logging import getLogger

import click
import gym

from .agents import agents

log = getLogger(__name__)


@click.group(invoke_without_command=True)
@click.option('--display', '-d', is_flag=True, help='Display the game as it\'s running.')
@click.option('--agent_id', '-a', default='RandomAgent', type=str, help='The agent id to use.')
@click.option('--environment_id', '-e', default='Breakout-v0', type=str, help='The environment id to use.')
@click.option('--num_episodes', '-e', default=100, type=int, help='Number of episodes to run.')
@click.pass_context
def main(ctx, display, agent_id, environment_id, num_episodes):
    if ctx.invoked_subcommand is not None:
        return
    log.info(f'Creating {environment_id} environment.')
    env = gym.make(environment_id)

    log.info(f'Creating {agent_id} agent.')
    agent = agents[agent_id](env.action_space)

    reward = 0
    done = False
    for _ in range(num_episodes):
        observation = env.reset()
        while True:
            action = agent.act(observation, reward, done)
            ob, reward, done, _ = env.step(action)
            if display:
                env.render()
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

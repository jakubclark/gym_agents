import click

from .agents import agents
from .experiments import run_experiments as experiments
from .runner import Runner


@click.group(invoke_without_command=True)
@click.option('--display', '-d', is_flag=True, help='Display the agent when testing')
@click.option('--model_path', '-m', default=None, help='Path to agent\'s model')
@click.option('--agent_id', '-a', default='DQNAgent', type=str, help='The agent id to use.')
@click.option('--environment_id', '-e', default='CustomMountainCar-v0', type=str, help='The environment id to use.')
@click.option('--num_steps', '-s', default=10000, type=int, help='Number of steps to run per episode')
@click.option('--train_starts', default=50, type=int, help='Number of episodes to run before training actually begins.')
@click.option('--save_freq', default=10, type=int, help='Number of episodes to run in between potential model saving')
@click.option('--update_freq', default=5, type=int, help='Number of episodes to run in between target model updates')
@click.option('--train_freq', default=5, type=int, help='Number of episodes in between model training')
@click.option('--play', is_flag=True, help='Have the agent play the game, without training')
@click.pass_context
def main(ctx, display, model_path, agent_id, environment_id, num_steps,
         train_starts, save_freq, update_freq, train_freq, play):
    if ctx.invoked_subcommand is not None:
        return

    runner = Runner(model_path, agent_id, environment_id, num_steps,
                    train_starts, save_freq, update_freq, train_freq, {})

    if play:
        runner.play_testing_games(display=display)
        runner.save_config()
        return

    runner.play_training_games()
    runner.play_testing_games(display=display)
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


@main.command()
def run_experiments():
    experiments()


if __name__ == '__main__':
    main()

from multiprocessing import Process

import click

from .runner import Runner

agent = 'DQNAgent'
env = 'CustomMountainCar-v0'

epsilon_configs = [
    {'epsilon_decay': 0.9},
    {'epsilon_decay': 0.99},
    {'epsilon_decay': 0.999}
]

layer_configs = [
    {'num_layers': 1},
    {'num_layers': 2},
    {'num_layers': 3},
]


def run_experiments():
    processes = []

    click.echo(
        'Staring the experiments. WARNING: Standard Output will be all over the place')

    def target(i_, iv_, config_):
        json_filename = f'{env}-{agent}-{iv_}-{i_}.json'
        model_filename = f'models/{env}-{agent}-{iv_}-{i_}.model'

        runner = Runner(model_filename, 'DQNAgent', 'CustomMountainCar-v0',
                        100000, 50, 10, 4, 4, config_)
        runner.play_training_games()
        runner.play_testing_games()
        runner.save_config(filename=json_filename)

        click.echo(f'Finished running process for config: {config}')

    iv = 'epsilon_decay'
    for i, config in enumerate(epsilon_configs):
        click.echo(f'Creating process for config: {config}')
        p = Process(target=target, args=(i, iv, config,))
        processes.append(p)
        p.start()

    iv = 'num_layers'
    for i, config in enumerate(layer_configs):
        click.echo(f'Creating process for config: {config}')
        p = Process(target=target, args=(i, iv, config,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()


if __name__ == '__main__':
    run_experiments()

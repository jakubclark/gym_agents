from multiprocessing import Process

import click

from .runner import Runner

agent = 'DQNAgent'
env = 'CustomMountainCar-v0'

iv_configs = {
    'learning_rate': [
        {'learning_rate': 1e-2},
        {'learning_rate': 1e-4},
        {'learning_rate': 1e-5}
    ],
    'epsilon_decay': [
        {'epsilon_decay': 0.9},
        {'epsilon_decay': 0.999},
        {'epsilon_decay': 0.9999}
    ],
    'num_layers': [
        {'num_layers': 0},
        {'num_layers': 2},
        {'num_layers': 3},
    ],
    'controlled': [
        {'epsilon_decay': 0.99, 'num_layers': 1, 'learning_rate': 1e-3}
    ]
}


def run_experiments():
    def target(i_, iv_, config_):
        json_filename = f'{env}-{agent}-{iv_}-{i_}.json'
        model_filename = f'models/{env}-{agent}-{iv_}-{i_}.model'

        runner = Runner(model_filename, 'DQNAgent', 'CustomMountainCar-v0',
                        100000, 50, 10, 4, 4, config_)
        runner.play_training_games()
        runner.play_testing_games()
        runner.save_config(filename=json_filename)

        click.echo(f'Finished running process for config: {config}')

    processes = []
    click.echo(
        'Staring the experiments. WARNING: Standard Output will be all over the place')

    for iv, configs in iv_configs.items():
        for i, config in enumerate(configs):
            click.echo(f'Creating process. IV: {iv}, Config: {config}')
            p = Process(target=target, args=(i, iv, config,))
            processes.append(p)
            p.start()

    for p in processes:
        p.join()


if __name__ == '__main__':
    run_experiments()

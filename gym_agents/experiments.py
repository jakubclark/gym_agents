from multiprocessing import Process

import click

from .constants import EXPERIMENT_RESULTS_PATH, MODELS_PATH
from .runner import Runner

agent = 'DQNAgent'
env = 'CustomMountainCar-v0'

ed = 'epsilon_decay'
nl = 'num_layers'
lr = 'learning_rate'

default_epsilon_decay = 0.99
default_num_layers = 1
default_learning_rate = 1e-3

iv_configs = {
    lr: [
        {lr: 1e-2, ed: default_epsilon_decay, nl: default_num_layers},
        {lr: 1e-3, ed: default_epsilon_decay, nl: default_num_layers},
        {lr: 1e-4, ed: default_epsilon_decay, nl: default_num_layers}
    ],
    ed: [
        {ed: 0.9, lr: default_learning_rate, nl: default_num_layers},
        {ed: 0.99, lr: default_learning_rate, nl: default_num_layers},
        {ed: 0.999, lr: default_learning_rate, nl: default_num_layers}
    ],
    nl: [
        {nl: 0, ed: default_epsilon_decay, lr: default_learning_rate},
        {nl: 1, ed: default_epsilon_decay, lr: default_learning_rate},
        {nl: 2, ed: default_epsilon_decay, lr: default_learning_rate}
    ]
}


def run_experiments(configs: dict = None):
    def target(i_, iv_, config_):
        json_filename = f'{EXPERIMENT_RESULTS_PATH}/{env}-{agent}-{iv_}-{i_}.json'
        model_filename = f'{MODELS_PATH}/{env}-{agent}-{iv_}-{i_}.model'

        runner = Runner(model_filename, 'DQNAgent', 'CustomMountainCar-v0',
                        100000, 50, 10, 4, 4, config_)
        runner.play_training_games()
        runner.play_testing_games()
        runner.save_config(filename=json_filename)

        click.echo(f'Finished running process for config: {config}')

    configs = configs or iv_configs
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

from multiprocessing import Process

from click import echo

from gym_agents.runner import Runner

iv = 'epsilon_decay'
agent = 'DQNAgent'
env = 'CustomMountainCar-v0'

agent_configs = [
    {iv: 0.99},
    {iv: 0.999},
    {iv: 0.9999}
]


def target(i, config):
    json_filename = f'{env}-{agent}-{iv}-{i}.json'
    model_filename = f'models/{env}-{agent}-{iv}-{i}.model'

    runner = Runner(model_filename, 'DQNAgent', 'CustomMountainCar-v0',
                    100000, 50, 10, 4, 4, config)
    runner.play_training_games()
    runner.play_testing_games()
    runner.save_config(filename=json_filename)

    echo(f'Finished running process for config: {config}')


if __name__ == '__main__':
    processes = []

    echo('Staring the experiments. WARNING: Standard Output will be all over the place')

    for i, config in enumerate(agent_configs):
        echo(f'Creating process for config: {config}')
        p = Process(target=target, args=(i, config,))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()

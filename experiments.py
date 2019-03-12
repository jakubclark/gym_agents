from gym_agents.runner import Runner

epsilon_decay = 'epsilon_decay'
agent = 'DQNAgent'
env = 'CustomMountainCar-v0'

agent_configs = [
    {epsilon_decay: 0.99},
    {epsilon_decay: 0.999},
    {epsilon_decay: 0.9999}
]


def main():

    for i, config in enumerate(agent_configs):
        filename = f'{agent}-{env}-{i}'
        runner = Runner(None, 'DQNAgent', 'CustomMountainCar-v0',
                        100000, 50, 4, 4, 4, config)
        runner.play_testing_games()
        runner.play_training_games()
        runner.save_config(filename=filename)


if __name__ == '__main__':
    main()

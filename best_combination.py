from gym_agents.experiments import run_experiments


best_epsilon_decay = 0.999
best_learning_rate = 0.01
best_num_layers = 2

if __name__ == '__main__':
    configs = {
        'best_performing': [{
            'epsilon_decay': 0.999,
            'learning_rate': 0.01,
            'num_layers': 2
        }]
    }

    run_experiments(configs)

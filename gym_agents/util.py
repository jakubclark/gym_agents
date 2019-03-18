import json
from pprint import PrettyPrinter

import click
import numpy as np
import pandas as pd
from matplotlib import patches as mpatches
from matplotlib import pyplot as plt

from .agents import create_agent
from .constants import EXPERIMENT_RESULTS_PATH, MODELS_PATH, POLICY_PLOTS_PATH
from .envs import create_env

printer = PrettyPrinter(indent=2)


def make_plot(x, y=None, xlabel=None, ylabel=None, title=None):
    if y:
        plt.plot(x, y)
    else:
        plt.plot(x)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.show()


def generate_policy_report(agent_id, env_id, model_file_path, iv, iteration):
    click.echo('Policy the agent uses when playing:')

    policy_fig_name = f'{POLICY_PLOTS_PATH}/{env_id}-{agent_id}-{iv}-{iteration}.png'

    env = create_env(env_id)
    agent = create_agent(agent_id, env.action_space, env.observation_space)
    agent.load(model_file_path)

    X = np.random.uniform(-1.2, 0.6, 10000)
    Y = np.random.uniform(-0.07, 0.07, 10000)
    Z = []

    for i in range(len(X)):
        arr = np.array(([[X[i], Y[i]]]))
        action = agent.act_model(arr, None, None)
        Z.append(action)
    Z = pd.Series(Z)
    colors_ = {0: 'blue', 1: 'lime', 2: 'red'}
    labels = ['Left', 'Right', 'Nothing']

    fig = plt.figure(3, figsize=[7, 7])
    ax = fig.gca()
    plt.set_cmap('brg')
    ax.scatter(X, Y, c=Z)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_title('Policy')
    recs = []

    for i in range(0, 3):
        recs.append(
            mpatches.Rectangle(
                (0, 0),
                1, 1,
                fc=colors_[i]
            )
        )
    plt.legend(recs, labels, loc=4, ncol=3)
    fig.savefig(policy_fig_name)
    plt.axvline(0.5)
    plt.show()


def generate_game_report(agent_id, env_id, model_filepath):
    click.echo('Report for a single game, using the trained agent:')
    env = create_env(env_id)
    agent = create_agent(agent_id, env.action_space, env.observation_space)
    agent.load(model_filepath)

    positions, velocities, actions = [], [], []

    state = env.reset()
    state = np.reshape(state, [1, 2])
    done = False
    while not done:

        pos = state[0][0]
        vel = state[0][1]
        positions.append(pos)
        velocities.append(vel)

        action = agent.act_model(state, None, None)

        actions.append(action)

        state, reward, done, _ = env.step(action)
        state = np.reshape(state, [1, 2])

    make_plot(positions, xlabel='Step Number',
              ylabel='Cart Position', title='Cart Position over Time')
    make_plot(velocities, xlabel='Step Number',
              ylabel='Cart Velocity', title='Cart Velocity over Time')

    fig = plt.figure(3, figsize=[7, 7])
    ax = fig.gca()
    ax.scatter(positions, velocities)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_title('Position vs Velocity')
    plt.show()


def generate_report(agent_id, env_id, iv, iteration):

    json_filename = f'{EXPERIMENT_RESULTS_PATH}/{env_id}-{agent_id}-{iv}-{iteration}.json'
    model_filename = f'{MODELS_PATH}/{env_id}-{agent_id}-{iv}-{iteration}.model'

    try:
        with open(json_filename) as fh:
            data = json.load(fh)
    except FileNotFoundError:
        click.echo(
            f'Did not find `{json_filename}`. Run `python experiments.py` first.')

    initial_agent_config = data['agent_config']['initial']
    final_agent_config = data['agent_config']['final']

    click.echo(f'Report for the Independent Variable:{iv}')

    click.echo('Initial agent config:')
    printer.pprint(initial_agent_config)
    click.echo('Final agent config:')
    printer.pprint(final_agent_config)

    click.echo('Train Episode Plots:')

    train_episode_reward_history = data['data']['train_episode_rewards']
    train_episode_steps_history = data['data']['train_episode_steps']
    make_plot(train_episode_reward_history, xlabel='Episode Number',
              ylabel='Train Episode Reward', title='Train Episode vs Reward')
    make_plot(train_episode_steps_history, xlabel='Episode Number',
              ylabel='Train Episode Steps', title='Train Episode vs Number of Steps')

    click.echo('Test Episode Plots:')
    test_episode_reward_history = data['data_test']['test_episode_rewards']
    test_episode_steps_history = data['data_test']['test_episode_steps']
    make_plot(test_episode_reward_history, xlabel='Episode Number',
              ylabel='Test Episode Reward', title='Test Episode vs Reward')
    make_plot(test_episode_steps_history, xlabel='Episode Number',
              ylabel='Test Episode Steps', title='Test Episode vs Number of Steps')

    click.echo('Agent & Model During Training:')
    epsilon_history = data['data']['train_episode_epsilons']
    loss_history = data['agent_history']['loss']
    make_plot(epsilon_history, xlabel='Episode Number',
              ylabel='Epsilon', title='Train Episode vs (final) Episode Epsilon')
    make_plot(loss_history, xlabel='Model Fit Call',
              ylabel='Loss Rate', title='Model Fit Call vs Loss Rate')

    save_freq = data['runner_config']['save_freq']
    saved_mean = data['runner_config']['saved_mean']
    saved_means = data['runner_config']['saved_means']

    click.echo('Saved Mean over time:')
    x = []
    y = []
    for mean in saved_means:
        x.append(mean['episode_num'])
        y.append(mean[f'{save_freq}_episode_mean'])
    make_plot(x, y, 'Episode Number', f'Last {save_freq} Episode Mean')

    generate_game_report(agent_id, env_id, model_filename)
    generate_policy_report(agent_id, env_id, model_filename, iv, iteration)

    agent_performance = data['agent_performance']

    train_average_reward = agent_performance['train_average_reward']
    test_average_reward = agent_performance['test_average_reward']
    click.echo(f'Average reward during training: {train_average_reward}')
    click.echo(f'Average reward during testing: {test_average_reward}')

    train_average_steps = agent_performance['train_average_steps']
    test_average_steps = agent_performance['test_average_steps']
    click.echo(f'Average num of steps during training: {train_average_steps}')
    click.echo(f'Average num of steps during testing: {test_average_steps}')

    train_games_played = agent_performance['train_games_played']
    test_games_played = agent_performance['test_games_played']
    click.echo(f'Number of training games played: {train_games_played}')
    click.echo(f'Number of testing games played: {test_games_played}')

    click.echo(f'The saved model had a mean reward of: {saved_mean}')

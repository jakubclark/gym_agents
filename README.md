# gym_agents

Smart agents for various [Gym](http://gym.openai.com/) Environments

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

What things you need to install the software and how to install them

* Python 3.6 ([pyenv](https://github.com/pyenv/pyenv) is recommended for controlling python versions)
* [pipenv](https://pipenv.readthedocs.io/en/latest/)

## Running `gym_agents`

```bash
$ pipenv run python -m gym_agents.py
```

## Development

In order to deelop locally, follow the following steps:

```bash
$ git clone git@github.com:jakubclark/gym-agents.git
$ pipenv install --dev
```

## Coding style

Code style is maintained by [autopep8](https://github.com/hhatto/autopep8) and [isort](https://github.com/timothycrosley/isort).

Simply run:

```bash
$ make format
$ make isort
```

## CLI

```bash
$ python -m gym_agents --help
```
```
Usage: runner.py [OPTIONS] COMMAND [ARGS]...
Options:
  -d, --display              Display the agent when testing
  -m, --model_path TEXT      Path to agent's model
  -a, --agent_id TEXT        The agent id to use.
  -e, --environment_id TEXT  The environment id to use.
  -s, --num_steps INTEGER    Number of steps to run per episode
  --train_starts INTEGER     Number of episodes to run before training
                             actually begins.
  --save_freq INTEGER        Number of episodes to run in between potential
                             model saving
  --update_freq INTEGER      Number of episodes to run in between target model
                             updates
  --train_freq INTEGER       Number of episodes in between model training
  --play                     Have the agent play the game, without training
  --help                     Show this message and exit.

Commands:
  list_agents
  list_environments
```

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
  -d, --display               Display the game as its running.
  -a, --agent_id TEXT         The agent id to use.
  -e, --environment_id TEXT   The environment id to use.
  -e, --num_episodes INTEGER  Number of episodes to run.
  --help                      Show this message and exit.

Commands:
  list_agents
  list_environments
```

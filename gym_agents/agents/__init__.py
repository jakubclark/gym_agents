from .random_agent import RandomAgent
from .dqn_agent import DQNAgent

agents = {
    'RandomAgent': RandomAgent,
    'DQNAgent': DQNAgent
}


def create_agent(agent_id, act_space, obs_space, *args, **kwargs):
    return agents[agent_id](act_space, obs_space, *args, **kwargs)

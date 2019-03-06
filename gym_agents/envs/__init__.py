from gym import make

from gym.envs.registration import register


register(
    id='CustomMountainCar-v0',
    entry_point='gym_agents.envs.classic_control.mountain_car:MountainCarEnv',
    max_episode_steps=200
)


def create_env(env_name):
    return make(env_name)

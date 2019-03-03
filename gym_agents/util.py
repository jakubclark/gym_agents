from gym.spaces import Box


def flatten_shape(observation_space: Box):
    res = 1
    for i in observation_space.shape:
        res *= i
    return res

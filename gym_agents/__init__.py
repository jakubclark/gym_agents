def flatten_shape(observation_space):
    res = 1
    for i in observation_space.shape:
        res *= i
    return res

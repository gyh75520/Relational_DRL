from gym.envs.registration import register
register(
    id='BoxWorldNoFrameskip-v4',
    entry_point='gym_boxworld.envs:BoxWoldEnv'
)
register(
    id='BoxRandWorldNoFrameskip-v4',
    entry_point='gym_boxworld.envs:BoxWoldRandEnv'
)

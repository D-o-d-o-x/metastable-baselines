from gym.envs.registration import register
from env import *

def register():
    register(
        # unique identifier for the env `name-version`
        id="Columbus-Test317-v0",
        # path to the class for creating the env
        # Note: entry_point also accept a class as input (and not only a string)
        entry_point=env.ColumbusEnv,
        # Max number of steps per episode, using a `TimeLimitWrapper`
        max_episode_steps=500,
    )

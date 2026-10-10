import gymnasium as gym
import highway_env

from matplotlib import pyplot as plt
import pprint

ENV_ID = 'highway-v0'        # use 'highway-fast-v0' while testing

def make_env(render_mode='rgb_array'):
    return gym.make(ENV_ID, render_mode=render_mode)


if __name__ == "__main__":
    env = make_env(render_mode='human')
    env.reset(seed=0)   # starts a new episode

    done = truncated = False
    while not (done or truncated):
        action = env.unwrapped.action_type.actions_indexes["IDLE"]
        obs, reward, done, truncated, info = env.step(action)

    env.close()
    pprint.pprint(env.unwrapped.config)
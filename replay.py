#!/usr/bin/python3.10
import gym
from gym.envs.registration import register
import numpy as np
import os
import time
import datetime

from stable_baselines3 import SAC, PPO, A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, MultiInputActorCriticPolicy

from metastable_baselines.trl_pg import TRL_PG
import columbus


def main(load_path, n_eval_episodes=0):
    load_path = load_path.replace('.zip', '')
    load_path = load_path.replace("'", '')
    load_path = load_path.replace(' ', '')
    file_name = load_path.split('/')[-1]
    # TODO: Ugly, Ugly, Ugly:
    env_name = file_name.split('_')[0]
    alg_name = file_name.split('_')[1]
    alg_deriv = file_name.split('_')[2]
    use_sde = file_name.find('sde') != -1
    print(env_name, alg_name, alg_deriv, use_sde)
    env = gym.make(env_name)
    if alg_name == 'ppo':
        Model = PPO
    elif alg_name == 'trl' and alg_deriv == 'pg':
        Model = TRL_PG
    else:
        raise Exception('Algorithm not implemented for replay')

    model = Model.load(load_path, env=env)

    if n_eval_episodes:
        mean_reward, std_reward = evaluate_policy(
            model, env, n_eval_episodes=n_eval_episodes, deterministic=False)
        print('Reward: '+str(round(mean_reward, 3)) +
              '±'+str(round(std_reward, 2)))

    input('<ready?>')
    obs = env.reset()
    episode_reward = 0
    while True:
        time.sleep(1/30)
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, done, info = env.step(action)
        env.render()
        episode_reward += reward
        if done:
            episode_reward = 0.0
            obs = env.reset()
    env.reset()


if __name__ == '__main__':
    main(input('[path to model> '))

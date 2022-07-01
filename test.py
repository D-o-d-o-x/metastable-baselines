#!/usr/bin/python3
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

#root_path = os.getcwd()
root_path = '.'


def main(env_name='ColumbusCandyland_Aux10-v0', timesteps=10_000_000, showRes=True, saveModel=True, n_eval_episodes=0):
    env = gym.make(env_name)
    use_sde = True
    ppo = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        tensorboard_log=root_path+"/logs_tb/" +
        env_name+"/ppo"+(['', '_sde'][use_sde])+"/",
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        normalize_advantage=True,
        ent_coef=0.02,  # 0.1
        vf_coef=0.5,
        use_sde=use_sde,  # False
        clip_range=0.2,
    )
    trl_pg = TRL_PG(
        "MlpPolicy",
        env,
        verbose=0,
        tensorboard_log=root_path+"/logs_tb/"+env_name +
        "/trl_pg"+(['', '_sde'][use_sde])+"/",
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        normalize_advantage=True,
        ent_coef=0.03,  # 0.1
        vf_coef=0.5,
        use_sde=use_sde,
        clip_range=2,  # 0.2
    )

    print('TRL_PG:')
    testModel(trl_pg, timesteps, showRes,
              saveModel, n_eval_episodes)
    # print('PPO:')
    # testModel(ppo, timesteps, showRes,
    #          saveModel, n_eval_episodes)


def testModel(model, timesteps, showRes=False, saveModel=False, n_eval_episodes=16):
    env = model.get_env()
    try:
        model.learn(timesteps)
    except KeyboardInterrupt:
        print('[!] Training Terminated')
        pass

    if saveModel:
        now = datetime.datetime.now().strftime('%d.%m.%Y-%H:%M')
        loc = root_path+'/models/' + \
            model.tensorboard_log.replace(
                root_path+'/logs_tb/', '').replace('/', '_')+now+'.zip'
        print(model.get_parameters())
        model.save(loc)

    if n_eval_episodes:
        mean_reward, std_reward = evaluate_policy(
            model, env, n_eval_episodes=n_eval_episodes, deterministic=False)
        print('Reward: '+str(round(mean_reward, 3)) +
              '±'+str(round(std_reward, 2)))

    if showRes:
        input('<ready?>')
        obs = env.reset()
        # Evaluate the agent
        episode_reward = 0
        while True:
            time.sleep(1/30)
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, done, info = env.step(action)
            env.render()
            episode_reward += reward
            if done:
                #print("Reward:", episode_reward)
                episode_reward = 0.0
                obs = env.reset()
    env.reset()


if __name__ == '__main__':
    # main('LunarLanderContinuous-v2')
    # main('ColumbusJustState-v0')
    # main('ColumbusStateWithBarriers-v0')
    main('ColumbusEasierObstacles-v0')

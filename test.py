#!/bin/python3
import gym
from gym.envs.registration import register
import numpy as np
import os
import time
import datetime

from stable_baselines3 import SAC, PPO, A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, MultiInputActorCriticPolicy

from sb3_trl.trl_pg import TRL_PG
import columbus

#root_path = os.getcwd()
root_path = '.'


def main(env_name='ColumbusCandyland_Aux10-v0', timesteps=50000, showRes=False, saveModel=True, n_eval_episodes=16):
    env = gym.make(env_name)
    test_sde = False
    ppo = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        tensorboard_log=root_path+"/logs_tb/"+env_name+"/ppo/",
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        normalize_advantage=True,
        ent_coef=0.15,  # 0.1
        vf_coef=0.5,
        use_sde=False,  # False
    )
    trl_pg = TRL_PG(
        "MlpPolicy",
        env,
        verbose=0,
        tensorboard_log=root_path+"/logs_tb/"+env_name+"/trl_pg/",
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        normalize_advantage=True,
        ent_coef=0.15,  # 0.1
        vf_coef=0.5,
        use_sde=False,  # False
    )
    if test_sde:
        ppo_latent_sde = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            tensorboard_log=root_path+"/logs_tb/"+env_name+"/ppo_latent_sde/",
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            normalize_advantage=True,
            ent_coef=0.15,  # 0.1
            vf_coef=0.5,
            use_sde=True,  # False
            sde_sample_freq=30*15,  # -1
        )
        trl_pg_latent_sde = TRL_PG(
            "MlpPolicy",
            env,
            verbose=0,
            tensorboard_log=root_path+"/logs_tb/"+env_name+"/trl_pg_latent_sde/",
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            normalize_advantage=True,
            ent_coef=0.15,  # 0.1
            vf_coef=0.5,
            use_sde=True,  # False
            sde_sample_freq=30*15,  # -1
        )
    # sac_latent_sde = SAC(
    #    "MlpPolicy",
    #    env,
    #    verbose=0,
    #    tensorboard_log=root_path+"/logs_tb/"+env_name+"/sac_latent_sde/",
    #    use_sde=True,
    #    sde_sample_freq=30*15,
    #    ent_coef=0.0016, #0.0032
    #    gamma=0.99, # 0.95
    #    learning_rate=0.001 # 0.015
    # )

    print('TRL_PG:')
    testModel(trl_pg, timesteps, showRes,
              saveModel, n_eval_episodes)
    print('PPO:')
    testModel(ppo, timesteps, showRes,
              saveModel, n_eval_episodes)


def testModel(model, timesteps, showRes=False, saveModel=False, n_eval_episodes=16):
    env = model.get_env()
    model.learn(timesteps)

    if saveModel:
        now = datetime.datetime.now().strftime('%d.%m.%Y-%H:%M')
        loc = root_path+'/models/' + \
            model.tensorboard_log.replace(
                root_path+'/logs_tb/', '').replace('/', '_')+now+'.zip'
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
    main('LunarLanderContinuous-v2')

import gym
from gym.envs.registration import register
import numpy as np
import time
import datetime

from stable_baselines3 import SAC, PPO, A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, MultiInputActorCriticPolicy

from sb3_trl.trl_pg import TRL_PG
import columbus


def main(env_name='ColumbusEasierObstacles-v0'):
    env = gym.make(env_name)
    ppo_latent_sde = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        tensorboard_log="./logs_tb/"+env_name+"/ppo_latent_sde/",
        use_sde=True,
        sde_sample_freq=30*15,
        ent_coef=0.0016/1.25, #0.0032
        vf_coef=0.00025/2, #0.0005
        gamma=0.99, # 0.95
        learning_rate=0.005/5 # 0.015
    )
    sac_latent_sde = SAC(
        "MlpPolicy",
        env,
        verbose=0,
        tensorboard_log="./logs_tb/"+env_name+"/sac_latent_sde/",
        use_sde=True,
        sde_sample_freq=30*15,
        ent_coef=0.0016, #0.0032
        gamma=0.99, # 0.95
        learning_rate=0.001 # 0.015
    )
    #trl = TRL_PG(
    #    "MlpPolicy",
    #    env,
    #    verbose=0,
    #    tensorboard_log="./logs_tb/"+env_name+"/trl_pg/",
    #)

    #print('PPO_LATENT_SDE:')
    #testModel(ppo_latent_sde, 1000000, showRes = True, saveModel=True, n_eval_episodes=3)
    print('SAC_LATENT_SDE:')
    testModel(ppo_latent_sde, 250000, showRes = True, saveModel=True, n_eval_episodes=0)
    #print('TRL_PG:')
    #testModel(trl)


def testModel(model, timesteps=150000, showRes=False, saveModel=False, n_eval_episodes=16):
    env = model.get_env()
    model.learn(timesteps)

    if saveModel:
        now = datetime.datetime.now().strftime('%d.%m.%Y-%H:%M')
        model.save('models/'+model.tensorboard_log.replace('./logs_tb/','').replace('/','_')+now+'.zip')

    if n_eval_episodes:
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes, deterministic=False)
        print('Reward: '+str(round(mean_reward,3))+'±'+str(round(std_reward,2)))

    if showRes:
        input('<ready?>')
        obs = env.reset()
        # Evaluate the agent
        episode_reward = 0
        for _ in range(30*60*5):
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

if __name__=='__main__':
    main()

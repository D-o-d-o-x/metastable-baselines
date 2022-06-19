import gym
from gym.envs.registration import register
import numpy as np
import time

from stable_baselines3 import SAC, PPO, A2C
from stable_baselines3.common.evaluation import evaluate_policy

from sb3_trl.trl_pg import TRL_PG
from columbus import env

register(
    id='ColumbusTestRay-v0',
    entry_point=env.ColumbusTestRay,
    max_episode_steps=30*60*5,
)

def main():
    #env = gym.make("LunarLander-v2")
    env = gym.make("ColumbusTestRay-v0")

    ppo = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./logs_tb/test/ppo",
        use_sde=False,
        ent_coef=0.0001,
        learning_rate=0.0004
    )
    ppo_sde = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./logs_tb/test/ppo_sde/",
        use_sde=True,
        sde_sample_freq=30*20,
        ent_coef=0.000001,
        learning_rate=0.0003
    )
    a2c = A2C(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./logs_tb/test/a2c/",
    )
    trl = TRL_PG(
        "MlpPolicy",
        env,
        verbose=0,
        tensorboard_log="./logs_tb/test/trl_pg/",
    )

    #print('PPO:')
    #testModel(ppo, 500000, showRes = True, saveModel=True, n_eval_episodes=4)
    print('PPO_SDE:')
    testModel(ppo_sde, 100000, showRes = True, saveModel=True, n_eval_episodes=0)
    #print('A2C:')
    #testModel(a2c, showRes = True)
    #print('TRL_PG:')
    #testModel(trl)


def testModel(model, timesteps=100000, showRes=False, saveModel=False, n_eval_episodes=16):
    env = model.get_env()
    model.learn(timesteps)

    if n_eval_episodes:
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes, deterministic=False)
        print('Reward: '+str(round(mean_reward,3))+'±'+str(round(std_reward,2)))

    if showRes:
        model.save("model")
        input('<ready?>')
        obs = env.reset()
        # Evaluate the agent
        episode_reward = 0
        for _ in range(1000):
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

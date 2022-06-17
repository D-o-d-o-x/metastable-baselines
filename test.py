import gym
import numpy as np
import time

from stable_baselines3 import SAC, PPO, A2C
from stable_baselines3.common.evaluation import evaluate_policy

from sb3_trl.trl_pg import TRL_PG

def main():
    env = gym.make("LunarLander-v2")
    ppo = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        tensorboard_log="./logs_tb/test/",
    )
    a2c = A2C(
        "MlpPolicy",
        env,
        verbose=0,
        tensorboard_log="./logs_tb/test/",
    )
    trl = TRL_PG(
        "MlpPolicy",
        env,
        verbose=0,
        tensorboard_log="./logs_tb/test/",
    )

    print('PPO:')
    testModel(ppo)
    print('A2C:')
    testModel(a2c)
    print('TRL_PG:')
    testModel(trl)


def testModel(model, timesteps=50000, showRes=False):
    env = model.get_env()
    model.learn(timesteps)

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=16, deterministic=False)

    print('Reward: '+str(round(mean_reward,3))+'±'+str(round(std_reward,2)))

    if showRes:
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

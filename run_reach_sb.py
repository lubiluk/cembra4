import gym
import panda_gym

from stable_baselines3 import HER, SAC
from stable_baselines3.sac import MlpPolicy

env = gym.make("PandaReach-v1", render=True)

model = HER.load("data/reach_sb", env=env)

obs = env.reset()
for _ in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(action)
    env.render()

    if done:
        obs = env.reset()
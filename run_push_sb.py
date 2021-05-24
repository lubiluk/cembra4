import gym
import panda_gym

from stable_baselines3 import HER
from wrappers import DoneOnSuccessWrapper

env = DoneOnSuccessWrapper(gym.make('PandaPush-v1', render=True))

model = HER.load("data/push_sb_best/best_model", env=env)

obs = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(action)
    env.render()

    if done:
        obs = env.reset()
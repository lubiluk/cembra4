import gym
import panda_gym

from stable_baselines3 import HER
from stable_baselines3.sac import MlpPolicy
from sb3_contrib.common.wrappers.time_feature import TimeFeatureWrapper

env = TimeFeatureWrapper(gym.make('PandaReach-v2', render=True))

model = HER.load("data/fetch_push_sb", env=env)

obs = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(action)
    env.render()

    if done:
        obs = env.reset()
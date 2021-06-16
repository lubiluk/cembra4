import gym
import panda_gym

from stable_baselines3 import HER, SAC
from stable_baselines3.sac import MlpPolicy
from wrappers import DoneOnSuccessWrapper

env =  DoneOnSuccessWrapper(gym.make("FreePandaPush-v1", render=True, object_shape="cube"))

model = HER.load("data/free_push_cube_sb", env=env)

obs = env.reset()
for _ in range(10000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(action)
    env.render()

    if done:
        obs = env.reset()
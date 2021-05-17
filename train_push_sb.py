import os
import gym
import panda_gym
import numpy as np
import torch as th
import torch.nn as nn

from stable_baselines3 import HER
from sb3_contrib.common.wrappers.time_feature import TimeFeatureWrapper
from sb3_contrib.tqc import TQC
from sb3_contrib.tqc import MlpPolicy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.obs_dict_wrapper import ObsDictWrapper

log_dir = "./data/push_sb_log"
save_path = "./data/push_sb"
best_save_path = "./data/push_sb_best"

os.makedirs(log_dir, exist_ok=True)

def make_env():
    return TimeFeatureWrapper(gym.make('PandaReach-v1', render=False))

env = make_env()
eval_env = ObsDictWrapper(DummyVecEnv([make_env]))

policy_kwargs = dict(
    net_arch=[512, 512, 512],
    n_critics=2
)

model = HER(
    MlpPolicy,
    env,
    TQC,
    verbose=1,
    online_sampling=True,
    buffer_size=1_000_000,
    batch_size=2048,
    learning_rate=0.001,
    learning_starts=1000,
    gamma=0.95,
    tau=0.05,
    goal_selection_strategy='future',
    n_sampled_goal=4,
    policy_kwargs=policy_kwargs,
)

eval_callback = EvalCallback(eval_env,
                             best_model_save_path=best_save_path,
                             log_path=log_dir,
                             eval_freq=4096,
                             deterministic=True,
                             render=False)

model.learn(total_timesteps=1_000_000, callback=eval_callback)
model.save(save_path)

obs = env.reset()
for _ in range(100):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()

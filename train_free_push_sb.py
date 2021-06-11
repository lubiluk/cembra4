import os
import gym
import panda_gym
import numpy as np
import torch as th
import torch.nn as nn

from stable_baselines3 import HER, SAC
from stable_baselines3.sac import MlpPolicy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.obs_dict_wrapper import ObsDictWrapper

from wrappers import DoneOnSuccessWrapper

th.backends.cudnn.benchmark = True
th.autograd.set_detect_anomaly(False)
th.autograd.profiler.profile(enabled=False)

log_dir = "./data/push_sb_log"
save_path = "./data/push_sb"
best_save_path = "./data/push_sb_best"

os.makedirs(log_dir, exist_ok=True)

def make_env():
    return DoneOnSuccessWrapper(gym.make('FreePandaPush-v1', render=False))

env = make_env()
eval_env = ObsDictWrapper(DummyVecEnv([make_env]))

policy_kwargs = dict(
    activation_fn=th.nn.ReLU,
    net_arch=[128, 128],
)

model = HER(
    MlpPolicy,
    env,
    SAC,
    verbose=1,
    online_sampling=False,
    buffer_size=1_000_000,
    batch_size=256,
    learning_rate=0.001,
    learning_starts=1000,
    gamma=0.95,
    ent_coef='auto',
    goal_selection_strategy='future',
    n_sampled_goal=4,
    policy_kwargs=policy_kwargs,
)

eval_callback = EvalCallback(eval_env,
                             best_model_save_path=best_save_path,
                             log_path=log_dir,
                             eval_freq=10_000,
                             deterministic=True,
                             render=False)

model.learn(total_timesteps=3_000_000, callback=eval_callback)
model.save(save_path)

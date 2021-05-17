import gym
import panda_gym
import numpy as np
import torch as th
import torch.nn as nn

from stable_baselines3 import HER, SAC
from stable_baselines3.sac import MlpPolicy

th.backends.cudnn.benchmark = True
th.autograd.set_detect_anomaly(False)
th.autograd.profiler.profile(enabled=False)

env = gym.make('PandaReach-v1', render=False)

policy_kwargs = dict(
    activation_fn=th.nn.ReLU,
    net_arch=[64, 64],
)

model = HER(
    MlpPolicy,
    env,
    SAC,
    online_sampling=False,
    verbose=1,
    max_episode_length=100,
    buffer_size=1_000_000,
    batch_size=256,
    learning_rate=0.001,
    learning_starts=1000,
    gamma=0.95,
    ent_coef='auto',
    n_sampled_goal=4,
    goal_selection_strategy='future',
    policy_kwargs=policy_kwargs
)

model.learn(total_timesteps=30000)
model.save("data/reach_sb")


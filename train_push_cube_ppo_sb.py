import gym
import panda_gym

from stable_baselines.common.policies import MlpLnLstmPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2

from wrappers import DoneOnSuccessWrapper
from gym.wrappers.flatten_observation import FlattenObservation

# from gym.wrappers.filter_observation import FilterObservation
from gym.wrappers.time_limit import TimeLimit

SAVE_PATH = "./data/push_sb"


def wrap(env):
    return FlattenObservation(
        DoneOnSuccessWrapper(TimeLimit(env, max_episode_steps=50), reward_offset=0)
    )


# multiprocess environment
env = make_vec_env(
    "PandaPush-v1",
    n_envs=32,
    wrapper_class=wrap,
    env_kwargs={"render": False},
)

model = PPO2(MlpLnLstmPolicy, env, verbose=1)
model.learn(total_timesteps=1_000_000)
model.save(SAVE_PATH)


# Enjoy trained agent
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

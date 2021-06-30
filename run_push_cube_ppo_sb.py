import gym
import panda_gym

from stable_baselines.common.policies import MlpLnLstmPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2

from wrappers import DoneOnSuccessWrapper
from gym.wrappers.flatten_observation import FlattenObservation
from gym.wrappers.filter_observation import FilterObservation

from train_reach_ppo_sb import SAVE_PATH

env = FlattenObservation(
    FilterObservation(
        DoneOnSuccessWrapper(
            gym.make("PandaReach-v1", render=True, reward_type="dense"),
            reward_offset=0,
        ),
        filter_keys=["observation", "desired_goal"],
    )
)


model = PPO2.load(SAVE_PATH)


# Enjoy trained agent
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
    if dones:
        env.reset()

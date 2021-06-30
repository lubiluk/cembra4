import gym
import panda_gym

from stable_baselines.common.policies import MlpLnLstmPolicy, MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2

from wrappers import DoneOnSuccessWrapper
from gym.wrappers.flatten_observation import FlattenObservation
from gym.wrappers.filter_observation import FilterObservation

from train_reach_ppo_sb import SAVE_PATH, wrap


if __name__ == "__main__":
    # multiprocess environment
    env = wrap(gym.make("PandaReachDense-v1", render=True))
    # Enjoy trained agent
    model = PPO2.load(SAVE_PATH)

    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()

        if dones:
            env.reset()

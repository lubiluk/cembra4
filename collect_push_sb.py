import gym
import panda_gym
import numpy as np
import h5py

from gym import spaces
from stable_baselines3 import HER, SAC
from stable_baselines3.sac import MlpPolicy
from pathlib import Path
from wrappers import DoneOnSuccessWrapper

DST_HDF = "data/collect_push_sb.hdf5"
DST_DIR = "data/collect_push_sb"
dir = Path(DST_DIR)
dir.mkdir(exist_ok=True)
# Clear old data from dir if exists
for child in dir.glob("*"):
    child.unlink()
N = 10_000


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length,) + shape


class Wrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Dict(
            dict(
                observation=env.observation_space["observation"]["observation"],
                desired_goal=env.observation_space["desired_goal"],
                achieved_goal=env.observation_space["achieved_goal"],
            )
        )
        img_dim = env.observation_space["observation"]["camera"].shape
        depth_dim = env.observation_space["observation"]["depth"].shape
        gimg_dim = env.observation_space["observation"]["goal_camera"].shape
        gdepth_dim = env.observation_space["observation"]["goal_depth"].shape
        lin_dim = env.observation_space["observation"]["observation"].shape
        act_dim = env.action_space.shape
        self.file = h5py.File(DST_HDF, "w")
        self.img_dset = self.file.create_dataset(
            "camera", combined_shape(N, img_dim), dtype="uint8"
        )
        self.depth_dset = self.file.create_dataset(
            "depth", combined_shape(N, depth_dim), dtype="float32"
        )
        self.gimg_dset = self.file.create_dataset(
            "goal_camera", combined_shape(N, gimg_dim), dtype="uint8"
        )
        self.gdepth_dset = self.file.create_dataset(
            "goal_depth", combined_shape(N, gdepth_dim), dtype="float32"
        )
        self.lin_dset = self.file.create_dataset(
            "observation", combined_shape(N, lin_dim), dtype="f"
        )
        self.act_dset = self.file.create_dataset(
            "action", combined_shape(N, act_dim), dtype="f"
        )
        self.done_dset = self.file.create_dataset(
            "done", N, dtype="f"
        )
        self.rew_dset = self.file.create_dataset(
            "reward", N, dtype="f"
        )
        self.dset_ptr = 0

    def __del__(self):
        self.file.close()

    def observation(self, obs):
        lin = obs["observation"]["observation"]
        img = obs["observation"]["camera"]
        gimg = obs["observation"]["goal_camera"]
        depth = obs["observation"]["depth"]
        gdepth = obs["observation"]["goal_depth"]

        if self.dset_ptr < N: # Prevent saving last observation unnecesarily
            self.img_dset[self.dset_ptr] = img
            self.gimg_dset[self.dset_ptr] = gimg
            self.depth_dset[self.dset_ptr] = depth
            self.gdepth_dset[self.dset_ptr] = gdepth
            self.lin_dset[self.dset_ptr] = lin

        obs["observation"] = lin
        return obs

    def step(self, action):
        observation, reward, done, info = self.env.step(action)

        self.act_dset[self.dset_ptr] = action
        self.done_dset[self.dset_ptr] = done
        self.rew_dset[self.dset_ptr] = reward
        self.dset_ptr += 1

        return self.observation(observation), reward, done, info


env = Wrapper(DoneOnSuccessWrapper(gym.make("PandaPushCam-v1", render=True)))

model = HER.load("trained/push_sb", env=env)

obs = env.reset()
for _ in range(N):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(action)

    if done:
        obs = env.reset()

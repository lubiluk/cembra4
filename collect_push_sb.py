import gym
import panda_gym
import numpy as np
import h5py

from gym import spaces
from stable_baselines3 import HER, SAC
from stable_baselines3.sac import MlpPolicy
from pathlib import Path
from wrappers import DoneOnSuccessWrapper

DST_HDF = "S:/collect_push_sb.hdf5"
DST_DIR = "S:/collect_push_sb"
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
                observation=env.observation_space["observation"]["full_observation"],
                desired_goal=env.observation_space["desired_goal"],
                achieved_goal=env.observation_space["achieved_goal"],
            )
        )
        img1_dim = env.observation_space["observation"]["camera1"].shape
        img2_dim = env.observation_space["observation"]["camera2"].shape
        depth1_dim = env.observation_space["observation"]["depth1"].shape
        depth2_dim = env.observation_space["observation"]["depth2"].shape
        gimg1_dim = env.observation_space["observation"]["goal_camera1"].shape
        gimg2_dim = env.observation_space["observation"]["goal_camera2"].shape
        gdepth1_dim = env.observation_space["observation"]["goal_depth1"].shape
        gdepth2_dim = env.observation_space["observation"]["goal_depth2"].shape
        lin_dim = env.observation_space["observation"]["robot_state"].shape
        full_dim = env.observation_space["observation"]["full_observation"].shape
        act_dim = env.action_space.shape
        self.file = h5py.File(DST_HDF, "w")
        self.img1_dset = self.file.create_dataset(
            "camera1", combined_shape(N, img1_dim), dtype="uint8"
        )
        self.img2_dset = self.file.create_dataset(
            "camera2", combined_shape(N, img2_dim), dtype="uint8"
        )
        self.depth1_dset = self.file.create_dataset(
            "depth1", combined_shape(N, depth1_dim), dtype="float32"
        )
        self.depth2_dset = self.file.create_dataset(
            "depth2", combined_shape(N, depth2_dim), dtype="float32"
        )
        self.gimg1_dset = self.file.create_dataset(
            "goal_camera1", combined_shape(N, gimg1_dim), dtype="uint8"
        )
        self.gimg2_dset = self.file.create_dataset(
            "goal_camera2", combined_shape(N, gimg2_dim), dtype="uint8"
        )
        self.gdepth1_dset = self.file.create_dataset(
            "goal_depth1", combined_shape(N, gdepth1_dim), dtype="float32"
        )
        self.gdepth2_dset = self.file.create_dataset(
            "goal_depth2", combined_shape(N, gdepth2_dim), dtype="float32"
        )
        self.lin_dset = self.file.create_dataset(
            "robot_state", combined_shape(N, lin_dim), dtype="f"
        )
        self.full_dset = self.file.create_dataset(
            "full_observation", combined_shape(N, full_dim), dtype="f"
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
        lin = obs["observation"]["robot_state"]
        full = obs["observation"]["full_observation"]
        img1 = obs["observation"]["camera1"]
        img2 = obs["observation"]["camera2"]
        gimg1 = obs["observation"]["goal_camera1"]
        gimg2 = obs["observation"]["goal_camera2"]
        depth1 = obs["observation"]["depth1"]
        depth2 = obs["observation"]["depth2"]
        gdepth1 = obs["observation"]["goal_depth1"]
        gdepth2 = obs["observation"]["goal_depth2"]

        if self.dset_ptr < N: # Prevent saving last observation unnecesarily
            self.img1_dset[self.dset_ptr] = img1
            self.img2_dset[self.dset_ptr] = img2
            self.gimg1_dset[self.dset_ptr] = gimg1
            self.gimg2_dset[self.dset_ptr] = gimg2
            self.depth1_dset[self.dset_ptr] = depth1
            self.depth2_dset[self.dset_ptr] = depth2
            self.gdepth1_dset[self.dset_ptr] = gdepth1
            self.gdepth2_dset[self.dset_ptr] = gdepth2
            self.lin_dset[self.dset_ptr] = lin
            self.full_dset[self.dset_ptr] = full

        obs["observation"] = obs["observation"]["full_observation"]
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

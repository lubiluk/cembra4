import os

import gym
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

import panda_gym
from algos import SAC
from algos.common import replay_buffer_her_cam
from algos.sac import core_her_cam_goal
from wrappers import DoneOnSuccessWrapper

torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(enabled=False)

save_path = "./data/push_cam_su"
exp_name = "push_cam_su"
demonstrations_path = "S:/collect_push_sb.hdf5"


class PreprocessingWrapper(gym.ObservationWrapper):
    """
    A wrapper that normalizes camera observations
    """

    def __init__(self, env):
        super(PreprocessingWrapper, self).__init__(env)

        self.img_size = (2, 100, 100)

        obs_spaces = dict(
            camera=gym.spaces.Box(
                -1.0,
                1.0,
                shape=self.img_size,
                dtype=np.float32,
            ),
            robot_state=env.observation_space.spaces["observation"]["robot_state"],
        )

        self.observation_space = gym.spaces.Dict(
            dict(
                desired_goal=env.observation_space["desired_goal"],
                achieved_goal=env.observation_space["achieved_goal"],
                observation=gym.spaces.Dict(obs_spaces),
            )
        )

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

    def observation(self, obs):
        """what happens to each observation"""

        # Convert image to grayscale
        img = obs["observation"]["camera"]
        gimg = obs["observation"]["goal_camera"]

        obs["observation"]["camera"] = torch.cat(
            (self.transform(img), self.transform(gimg)), dim=0
        )

        return obs


class Extractor(nn.Module):
    def __init__(self):
        super(Extractor, self).__init__()

        obs_space = env.observation_space.spaces["observation"]["camera"]

        self.cnn = nn.Sequential(
            nn.Conv2d(obs_space.shape[0], 8, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            obs = torch.as_tensor(obs_space.sample()[None]).float()
            n_flatten = self.cnn(obs).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, 64), nn.ReLU())

    def forward(self, x):
        x = self.linear(self.cnn(x))
        return x


if __name__ == "__main__":

    def preload():
        # Load demonstrations
        with h5py.File(demonstrations_path, "r") as f:
            with torch.no_grad():
                t = transforms.Compose(
                    [
                        transforms.ToPILImage(),
                        transforms.Grayscale(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,)),
                    ]
                )

                N = len(f["action"])

                for i in range(N):
                    robot_state = f["observation"][i][:6]
                    img = f["camera"][i]
                    gimg = f["goal_camera"][i]
                    camera = torch.cat((t(img), t(gimg)), dim=0)
                    action = f["action"][i]
                    reward = f["reward"][i]
                    done = f["done"][i]
                    desired_goal = (0, 0, 0)
                    achieved_goal = (0, 0, 0)

                    obs = {
                        "observation": {"camera": camera, "robot_state": robot_state},
                        "achieved_goal": achieved_goal,
                        "desired_goal": desired_goal,
                    }

                    if i == N - 1:
                        next_robot_state = robot_state.copy()
                        next_camera = torch.clone(camera)
                    else:
                        next_robot_state = robot_state = f["observation"][i+1][:6]
                        next_img = f["camera"][i+1]
                        next_gimg = f["goal_camera"][i+1]
                        next_camera = torch.cat((t(next_img), t(next_gimg)), dim=0)

                    next_obs = {
                        "observation": {"camera": next_camera, "robot_state": next_robot_state},
                        "achieved_goal": achieved_goal,
                        "desired_goal": desired_goal,
                    }

                    info = {}

                    yield (obs, action, reward, next_obs, done, info)

    os.makedirs(save_path, exist_ok=True)

    env = PreprocessingWrapper(DoneOnSuccessWrapper(gym.make("PandaPushCam-v1")))

    ac_kwargs = dict(
        hidden_sizes=[256, 256], activation=nn.ReLU, extractor_module=Extractor
    )
    rb_kwargs = dict(
        size=5_000,
        n_sampled_goal=4,
        goal_selection_strategy="future",
        preloader=preload,
    )

    logger_kwargs = dict(output_dir=save_path, exp_name=exp_name)

    model = SAC(
        env=env,
        actor_critic=core_her_cam_goal.MLPActorCritic,
        ac_kwargs=ac_kwargs,
        replay_buffer=replay_buffer_her_cam.ReplayBuffer,
        rb_kwargs=rb_kwargs,
        max_ep_len=100,
        batch_size=256,
        gamma=0.95,
        lr=0.0003,
        update_after=512,
        update_every=512,
        logger_kwargs=logger_kwargs,
        use_gpu_buffer=True,
    )

    model.train(steps_per_epoch=1024, epochs=5000)

    from algos.test_policy import load_policy_and_env, run_policy

    _, get_action = load_policy_and_env(save_path)

    run_policy(env, get_action)

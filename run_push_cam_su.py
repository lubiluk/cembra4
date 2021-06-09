import os

import gym
import panda_gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from wrappers import DoneOnSuccessWrapper
from train_push_cam_su import Extractor, PreprocessingWrapper, save_path


if __name__ == "__main__":
    from algos.test_policy import load_policy_and_env, run_policy
    env = PreprocessingWrapper(DoneOnSuccessWrapper(gym.make("PandaPushCam-v1", render=True)))
    _, get_action = load_policy_and_env(save_path)
    run_policy(env, get_action)

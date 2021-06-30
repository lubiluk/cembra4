from operator import mod
import gym
import panda_gym
from wrappers import DoneOnSuccessWrapper
import cv2

env = DoneOnSuccessWrapper(gym.make("PandaPushCam-v1", render=True))

obs = env.reset()
done = False

i = 0
while True:
    action = env.action_space.sample()  # random action
    obs, reward, done, info = env.step(action)
    env.render()

    cv2.imshow('camera1', obs["observation"]["camera1"])
    cv2.imshow('camera2', obs["observation"]["camera2"])

    i += 1

    if i % 50 == 0:
        env.reset()

env.close()

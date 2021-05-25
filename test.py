import gym
import panda_gym
from wrappers import DoneOnSuccessWrapper
import cv2

env = DoneOnSuccessWrapper(gym.make("PandaReachCam-v1", render=True))

obs = env.reset()
done = False
while True:
    action = env.action_space.sample()  # random action
    obs, reward, done, info = env.step(action)

env.close()

from operator import mod
import gym
import panda_gym
from wrappers import DoneOnSuccessWrapper

env = DoneOnSuccessWrapper(gym.make("PandaPushCam-v1", render=True))

obs = env.reset()
done = False
while True:
    action = env.action_space.sample()  # random action
    obs, reward, done, info = env.step(action)
    env.render()

env.close()

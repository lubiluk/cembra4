from operator import mod
import gym
import panda_gym
from wrappers import DoneOnSuccessWrapper

env = DoneOnSuccessWrapper(gym.make("FreePandaPush-v1", render=True, object_shape="sphere"))

obs = env.reset()
done = False

i = 0
while True:
    action = env.action_space.sample()  # random action
    obs, reward, done, info = env.step(action)
    env.render()

    i += 1

    if i % 50 == 0:
        env.reset()

env.close()

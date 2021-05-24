import gym
import panda_gym
from wrappers import DoneOnSuccessWrapper

env = DoneOnSuccessWrapper(gym.make('PandaReach-v1', render=True))

obs = env.reset()
done = False
while not done:
    action = env.action_space.sample() # random action
    obs, reward, done, info = env.step(action)

env.close()
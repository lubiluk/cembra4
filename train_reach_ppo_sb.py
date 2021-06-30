import gym
import panda_gym
import tensorflow as tf

from stable_baselines.common.policies import MlpLnLstmPolicy, MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2

from wrappers import DoneOnSuccessWrapper
from gym.wrappers.flatten_observation import FlattenObservation
from gym.wrappers.filter_observation import FilterObservation
from stable_baselines.common.vec_env import VecNormalize

SAVE_PATH = "./data/reach_sb"


def wrap(env):
    return FlattenObservation(
            FilterObservation(
                DoneOnSuccessWrapper(env, reward_offset=0),
                filter_keys=["observation", "desired_goal"],
            )
        )
    


if __name__ == "__main__":
    # multiprocess environment
    env = VecNormalize(make_vec_env("PandaReachDense-v1", n_envs=8, wrapper_class=wrap))

    policy_kwargs = dict(
        act_fun=tf.nn.relu,
        layers=[64, 64],
    )

    model = PPO2(
        MlpPolicy,
        env,
        verbose=1,
        policy_kwargs=policy_kwargs,
        learning_rate=2.5e-4,
        n_steps=2048,
        noptepochs=10,
        cliprange=0.2,
        ent_coef=0.0,
        nminibatches=32,
    )
    model.learn(total_timesteps=10_000_000)
    model.save(SAVE_PATH)

    # Enjoy trained agent
    # model = PPO2.load(SAVE_PATH)
    
    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()

        if dones:
            env.reset()

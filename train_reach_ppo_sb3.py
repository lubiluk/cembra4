import gym
import panda_gym
import torch as th

from stable_baselines3.common.env_util  import make_vec_env
from stable_baselines3 import PPO

from wrappers import DoneOnSuccessWrapper
from gym.wrappers.flatten_observation import FlattenObservation
from gym.wrappers.filter_observation import FilterObservation
from stable_baselines3.common.vec_env import VecNormalize

SAVE_PATH = "./data/reach_sb3"


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
        activation_fn=th.nn.ReLU,
        net_arch=[64, 64],
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        policy_kwargs=policy_kwargs,
        learning_rate=2.5e-4,
        n_steps=2048,
        clip_range=0.2,
        ent_coef=0.0,
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

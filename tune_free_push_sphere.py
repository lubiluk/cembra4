import os

import gym
import numpy as np
import optuna
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from optuna.trial import TrialState
from stable_baselines3 import HER, SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.obs_dict_wrapper import ObsDictWrapper
from stable_baselines3.sac import MlpPolicy
from torchvision import datasets, transforms

import panda_gym
from wrappers import DoneOnSuccessWrapper

th.backends.cudnn.benchmark = True
th.autograd.set_detect_anomaly(False)
th.autograd.profiler.profile(enabled=False)

log_dir = "./data/free_push_sphere_sb_log"
save_path = "./data/free_push_sphere_sb"
best_save_path = "./data/free_push_sphere_sb_best"

os.makedirs(log_dir, exist_ok=True)

def make_env():
    return DoneOnSuccessWrapper(gym.make('FreePandaPush-v1', render=False, object_shape="sphere"))


def define_model(trial, env):
    n_layers = trial.suggest_int("n_layers", 1, 4)
    layers = [trial.suggest_int("n_units_l{}".format(i), 32, 512) for i in range(n_layers)]


    policy_kwargs = dict(
        activation_fn=th.nn.ReLU,
        net_arch=layers,
    )

    model = HER(
        MlpPolicy,
        env,
        SAC,
        verbose=1,
        max_episode_length=200,
        online_sampling=False,
        buffer_size=1_000_000,
        batch_size=trial.suggest_int("batch_size", 256, 2048),
        learning_rate=trial.suggest_float("lr", 0.0001, 0.1),
        learning_starts=1000,
        gamma=trial.suggest_float("gamma", 0.5, 0.99),
        ent_coef='auto',
        goal_selection_strategy='future',
        n_sampled_goal=trial.suggest_int("n_sampled_goal", 1, 10),
        policy_kwargs=policy_kwargs,
    )

    return model


class TuneCallback(BaseCallback):
    def __init__(self, trial, verbose=0, eval_freq=100):
        super().__init__(verbose=verbose)
        self.eval_freq = eval_freq
        self.trial = trial

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            accuracy = sum(self.model.ep_success_buffer) / len(self.model.ep_success_buffer)
            self.trial.report(accuracy,  self.n_calls / self.eval_freq)

            if self.trial.should_prune():
                return False

        return True


def objective(trial):
    env = make_env()
    eval_env = ObsDictWrapper(DummyVecEnv([make_env]))

    # Generate the model.
    model = define_model(trial, env)

    eval_callback = EvalCallback(eval_env,
                             best_model_save_path=best_save_path,
                             log_path=log_dir,
                             eval_freq=10_000,
                             deterministic=True,
                             render=False)

    model.learn(total_timesteps=3_000_000, callback=[eval_callback, TuneCallback(trial=trial)])

    if model.num_timesteps < 3_000_000:
        raise optuna.exceptions.TrialPruned()

    model.save(save_path)

    accuracy = sum(model.model.ep_success_buffer) / len(model.model.ep_success_buffer)
    return accuracy


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, catch=(ValueError,))

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

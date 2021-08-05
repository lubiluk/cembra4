import torch
import gym
import panda_gym
from neat.phenotype.feed_forward import FeedForwardNet
from gym.wrappers.flatten_observation import FlattenObservation
from gym.wrappers.filter_observation import FilterObservation
from wrappers import DoneOnSuccessWrapper

def wrap(env):
    return FlattenObservation(
            FilterObservation(
                DoneOnSuccessWrapper(env, reward_offset=0),
                filter_keys=["observation", "desired_goal"],
            )
        )

class PoleBalanceConfig:
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    VERBOSE = True

    NUM_INPUTS = 4
    NUM_OUTPUTS = 3
    USE_BIAS = True

    ACTIVATION = "tanh"
    SCALE_ACTIVATION = 1.0

    FITNESS_THRESHOLD = 990.0

    POPULATION_SIZE = 150
    NUMBER_OF_GENERATIONS = 1_000_000
    SPECIATION_THRESHOLD = 3.0

    CONNECTION_MUTATION_RATE = 0.80
    CONNECTION_PERTURBATION_RATE = 0.90
    ADD_NODE_MUTATION_RATE = 0.03
    ADD_CONNECTION_MUTATION_RATE = 0.5

    CROSSOVER_REENABLE_CONNECTION_GENE_RATE = 0.25

    # Top percentage of species to be saved before mating
    PERCENTAGE_TO_SAVE = 0.80


    def fitness_fn(self, genome):
        # OpenAI Gym
        env = wrap(gym.make("PandaReach-v1", render=False, reward_type="dense"))

        fitness = 1000
        phenotype = FeedForwardNet(genome, self)

        for i in range(10):
            done = False
            observation = env.reset()

            while not done:
                input = torch.Tensor([observation]).to(self.DEVICE)

                pred = phenotype(input).detach().cpu().numpy().squeeze()
                observation, reward, done, info = env.step(pred)
                fitness += reward

        env.close()

        return fitness


import logging

import gym
import torch

import neat.population as pop
from neat.visualize import draw_net
from neat.phenotype.feed_forward import FeedForwardNet

logger = logging.getLogger(__name__)

logger.info(PoleBalanceConfig.DEVICE)
neat = pop.Population(PoleBalanceConfig)
solution, generation = neat.run()

if solution is not None:
    logger.info("Found a Solution")
    draw_net(
        solution,
        view=True,
        filename="./images/reach-solution",
        show_disabled=True,
    )

    # OpenAI Gym
    env = wrap(gym.make("PandaReach-v1", render=True, reward_type="dense"))
    done = False
    observation = env.reset()

    phenotype = FeedForwardNet(solution, PoleBalanceConfig)

    while True:
        env.render()
        input = torch.Tensor([observation]).to(PoleBalanceConfig.DEVICE)

        pred = phenotype(input).detach().cpu().numpy().squeeze()
        observation, reward, done, info = env.step(pred)

        if done:
            observation = env.reset()

    env.close()

import time
from copy import deepcopy

import numpy as np
from numpy.random import random, randint

from simplegp.Selection import Selection
from simplegp.Variation import Variation


class SimpleGP:

    def __init__(
            self,
            tuner=None,
            fitness_function=None,
            linear_scale=False,
            functions=None,
            terminals=None,
            pop_size=500,
            crossover_rate=1.0,
            mutation_rate=0.0,
            max_evaluations=-1,
            max_generations=-1,
            max_time=-1,
            initialization_max_tree_height=4,
            max_tree_size=100,
            tournament_size=4,
            baldwin=False
    ):

        self.baldwin = baldwin
        self.tuner = tuner
        self.start_time = 0
        self.pop_size = pop_size
        self.fitness_function = fitness_function
        self.linear_scale = linear_scale
        self.functions = functions
        self.terminals = terminals
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

        self.max_evaluations = max_evaluations
        self.max_generations = max_generations
        self.max_time = max_time

        self.initialization_max_tree_height = initialization_max_tree_height
        self.max_tree_size = max_tree_size
        self.tournament_size = tournament_size

        self.generations = 0

    def __should_terminate(self):
        must_terminate = False
        elapsed_time = time.time() - self.start_time
        if 0 < self.max_evaluations <= self.fitness_function.evaluations:
            must_terminate = True
        elif 0 < self.max_generations <= self.generations:
            must_terminate = True
        elif 0 < self.max_time <= elapsed_time:
            must_terminate = True

        if must_terminate:
            print('Terminating at\n'
                  f'\t{self.generations} generations\n'
                  f'\t{self.fitness_function.evaluations} evaluations\n'
                  f'\t{np.round(elapsed_time, 2)} seconds')

        return must_terminate

    def _tune_generation(self, population, offspring):
        tuned_generation = []
        # If in Baldwin mode, tune both the population + offspring, otherwise only tune the offspring
        if self.baldwin:
            individuals_to_be_tuned = population + offspring
        else:
            individuals_to_be_tuned = offspring

        for indv in individuals_to_be_tuned:
            if indv.get_number_of_children() > 0 and self.generations in self.tuner.run_generations:
                if random() < self.tuner.population_fraction:
                    self.tuner.set_individual(indv)
                    indv = self.tuner.tune_weights()
                tuned_generation.append(indv)
            else:
                tuned_generation.append(indv)

        # If in Lamarckian mode, the parents still need to be added to the newly tuned offspring
        # In Baldwin mode, both parents and offspring are tuned each generation
        if not self.baldwin:
            tuned_generation.extend(population)

        return tuned_generation

    def run(self):
        # Reset the GA
        self.generations = 0
        self.start_time = time.time()

        population = []
        for i in range(self.pop_size):
            population.append(
                Variation.generate_random_tree(self.functions, self.terminals, self.initialization_max_tree_height))
            self.fitness_function.evaluate(population[i])

        while not self.__should_terminate():

            offspring = []

            for i in range(self.pop_size):

                o = deepcopy(population[i])
                if random() < self.crossover_rate:
                    o = Variation.subtree_crossover(o, population[randint(self.pop_size)])
                if random() < self.mutation_rate:
                    o = Variation.subtree_mutation(o, self.functions, self.terminals,
                                                   max_height=self.initialization_max_tree_height)

                if len(o.get_subtree()) > self.max_tree_size:
                    del o
                    o = deepcopy(population[i])
                else:
                    self.fitness_function.evaluate(o)

                offspring.append(o)

            population_and_offspring = self._tune_generation(population, offspring)

            population = Selection.tournament_select(population_and_offspring, self.pop_size, tournament_size=self.tournament_size)

            # Reset the weights of all individuals if in Baldwin mode
            if self.baldwin:
                for indv in population:
                    indv.reset_weights()

            self.generations = self.generations + 1
            print('GA '
                  f'{self.generations} '
                  f'{np.round(self.fitness_function.elite.fitness, 3)} '
                  f'{len(self.fitness_function.elite.get_subtree())}')


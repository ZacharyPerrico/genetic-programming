import random
import string

import numpy as np

from evolve import *
from node import *
from utils import *

# All functions relevant to non-genetic programming.

#
# Initialization
#

def gen_individual(target, **kwargs):
    return ''.join(random.choices(string.ascii_uppercase, k=len(target)))

#
# Evaluation
#

def fitness_func(pop, target, **kwargs):
    """Calculate the fitness value of all chromosomes in a population"""
    # goal = [letter_to_number[i] for i in function]
    fits = [
        sum([target[i] != letter for i, letter in enumerate(individual)])
        # sum([
        #     abs(goal[i]-letter_to_number[letter])
        #     for i,letter in enumerate(individual)
        # ])
        for individual in pop
    ]
    return fits

#
# Mutation
#

def mutation_func(individual, p_m, verbose=0, **kwargs):
    """Preform a mutation with a probability of p_m"""
    individual = list(individual)
    for i, letter in enumerate(individual):
        if random.random() < p_m:
            individual[i] = random.choice(string.ascii_uppercase)
    return ''.join(individual)

# Default kwargs
kwargs = {
    'seed': None,
    # 'function': lambda x: x**5 - 2*x**3 + x,
    'num_runs': 1,
    'num_gens': 100,
    'verbose': 1,
    'gen_individual': gen_individual,
    'init_tree_depth': 4,
    'fitness_func': fitness_func,
    'target': 'PROGRAMMING',
    'pop_size': 1,
    'keep_parents': 4, # Must be even
    'mutate_func': mutation_func,
    'p_m': 0.5, # Probability of a bit mutating
    'lambda': 20, # Lambda
}

if __name__ == '__main__':

    kwargs['label_title'] = 'Probability of Mutation'
    kwargs['labels'] = [0.3, 0.5]
    kwargs['key'] = 'p_m'
    kwargs['values'] = [0.3, 0.5]

    # Run simulation
    all_pops, all_fits = run_sims(**kwargs)
    # save_all(all_fits, all_pops, kwargs)
    plot_sims(all_pops, all_fits, **kwargs)
import numpy as np

from models.daggp.plot import plot_results
from src.evolve import run_tests
from src.models.daggp import *


kwargs = {
    'saves_path': '../../../saves/test/node',  # Save path relative to this file
    'verbose': True,
    'parallelize': True,
    'checkpoint_interval': 100,
    'update_timeout': 60,
    'save_formater_func': dag_to_save_str,  # Function to convert an individual into a savable string
    'load_formater_func': dag_from_save_str,  # Function to load an individual from a saved string
    ## Size ##
    'num_reps': 16,
    'num_gens': 100,
    'pop_size': 100,
    'max_height': 5,
    ## Initialization ##
    'init_individual_func': random_tree,  # Function used to generate a new organism
    'init_max_height': 4,
    'p_branch': 0.5,  # Probability of a node not being a terminal
    'ops': ['+', '-', '*', '/', '**'],
    'terminals': ['x'],
    ## Evaluation ##
    'eval_method': None,
    'fitness_func': mse,
    'timeout': 16,  # Number of evaluation iterations before forced termination
    'target_func': trig_sin,
    'domains': [list(np.linspace(-np.pi/2,np.pi/2,15))],
    ## Selection ##
    'minimize_fitness': True,
    'keep_parents': 2,  # Elitism, must be even
    'tournament_size': 2,  # Number of randomly chosen parents for each tournament
    ## Repopulation ##
    'subgraph_max_height': 2,
    'recombination_funcs': [subgraph_crossover],
    'recombination_probs': [0.2],
    'mutation_funcs': [subgraph_mutation, pointer_mutation],
    'mutation_probs': [0.4, 0.3],
    ## Tests ##
    'test_label': 'Field',  # Label to use when comparing all tests
    'test_keys': ['test_name', 'terminals'],  # Keys of each parameter to be changed for each test
    'test_values': [  # Tuple of tuples representing all values to change for each test
        ['Real', ['x']],
        ['Complex', ['x','i']],
        ['Irrational Complex', ['x','i','e']],
    ],
}


if __name__ == '__main__':
    run_tests(**kwargs)
    plot_results(**kwargs)
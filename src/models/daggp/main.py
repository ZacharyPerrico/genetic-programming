from src.evolve import run_tests
from src.models.daggp import *

# kwargs = {
#     'name': 'node_demo',  # Name of folder to contain all results
#     'seed': None,
#     'verbose': True,
#     'parallelize': True,
#     'saves_path': '../../../saves/dag/',  # Save path relative to this file
#     ## Size ##
#     'num_runs': 24,
#     'num_gens': 300,
#     'pop_size': 100,
#     'max_height': 10,  # The maximum height
#     ## Initialization ##
#     'init_individual_func': random_tree,  # Function used to generate the initial population
#     'ops': ['+', '-', '*', '/', '**'],
#     'terminals': ['x'],
#     'init_max_height': 4,
#     'p_branch': 0.5,  # Probability of a node branching
#     ## Evaluation ##
#     'eval_method': None,
#     'target_func': nate,
#     'fitness_func': mse,
#     'result_fitness_func': mse,  # Fitness to compare results
#     'domains': [[-4, 4, 50]],  # The domain of each variable expressed using np.linspace
#     ## Selection ##
#     'minimize_fitness': True,
#     'keep_parents': 2,  # Elitism, must be even
#     'k': 2,  # Number of randomly chosen parents for each tournament
#     ## Repopulation ##
#     'crossover_funcs': [
#         [subgraph_crossover, 0.2],
#     ],
#     'mutate_funcs': [
#         [subgraph_mutation, 0.3],
#         [pointer_mutation, 0.3],
#     ],
#     ## Tests ##
#     'test_kwargs': [
#         ['Initial Population', 'terminals', ],
#         ['Variable Only', ['x'], ],
#         # ['With Constants', ['x'] + list(range(-5, 6)), ],
#     ],
# }


kwargs = {
    # 'name': 'test/node',  # Name of folder to contain all results
    'saves_path': '../../../saves/test/node',  # Save path relative to this file
    'verbose': True,
    'parallelize': True,
    'checkpoint_interval': 10,
    'update_timeout': 60,
    ## Size ##
    'num_reps': 16,
    'num_gens': 100,
    'pop_size': 100,
    'max_height': 10,  # The maximum height
    ## Initialization ##
    'init_individual_func': random_tree,  # Function used to generate a new organism
    'ops': ['+', '-', '*', '/'],
    'terminals': ['x'],
    'init_max_height': 4,
    'p_branch': 0.5,  # Probability of a node branching
    ## Evaluation ##
    'eval_method': None,
    'fitness_func': mse,
    'timeout': 16,  # Number of evaluation iterations before forced termination
    'target_func': koza_3,
    'domains': [list(range(1,6))],
    ## Selection ##
    'minimize_fitness': True,
    'keep_parents': 2,  # Elitism, must be even
    'tournament_size': 2,  # Number of randomly chosen parents for each tournament
    ## Repopulation ##
    'subgraph_max_height': 2,
    'recombination_funcs': [subgraph_crossover],
    'recombination_probs': [0.2],
    'mutation_funcs': [subgraph_mutation],
    'mutation_probs': [0.7],
    ## Tests ##
    'test_label': 'Number Registers',  # Label to use when comparing all tests
    'test_keys': ['test_name', 'terminals'],  # Keys of each parameter to be changed for each test
    'test_values': [  # Tuple of tuples representing all values to change for each test
        ['Variable Only', ['x']],
        ['With Constants', ['x'] + list(range(-5, 6))],
    ],
}


if __name__ == '__main__':
    run_tests(**kwargs)
    # plot_results(fits, **kwargs)
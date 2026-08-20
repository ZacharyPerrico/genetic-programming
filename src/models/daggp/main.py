import numpy as np

from models.abstract.methods import polynomial, koza_2, trig_sin
from models.daggp.plot import plot_results
from models.daggp.methods import *
from src.evolve import run_tests


# kwargs = {
#     'saves_path': '../../../saves/daggp/test',  # Save path relative to this file
#     'verbose': True,
#     'parallelize': True,
#     'checkpoint_interval': 500,
#     'update_timeout': 60,  # Time before a replicate fails if it cannot update the database
#     'save_formater_func': dag_to_save_str,  # Function to convert an individual into a savable string
#     'load_formater_func': dag_from_save_str,  # Function to load an individual from a saved string
#     ## Size ##
#     'num_reps': 10,
#     'num_gens': 100,
#     'pop_size': 100,
#     'max_height': 6,
#     ## Initialization ##
#     'init_individual_func': random_tree,  # Function used to generate a new organism
#     'init_max_height': 2,
#     'p_branch': 0.75,  # Probability of a node not being a terminal
#     'ops': ['+', '-', '*', '/', 'real', 'imag', 'exp'],
#     'terminals': ['x','i'],
#     ## Evaluation ##
#     'eval_method': None,
#     'fitness_func': dag_mse,
#     'target_func': koza_2,
#     'domains': [list(np.linspace(-1,1,15))],
#     ## Selection ##
#     'minimize_fitness': True,
#     'keep_parents': 2,  # Elitism, must be even
#     'tournament_size': 2,  # Number of randomly chosen parents for each tournament
#     ## Repopulation ##
#     'subgraph_max_height': 2,
#     'recombination_funcs': [subgraph_crossover],
#     'recombination_probs': [0.25],
#     'mutation_funcs': [subgraph_mutation, pointer_mutation],
#     'mutation_probs': [0.25, 0.5],
#     ## Tests ##
#     'test_label': 'Field',  # Label to use when comparing all tests
#     'test_keys': ['test', 'terminals', 'ops'],  # Keys of each parameter to be changed for each test
#     'test_values': [  # Lists representing all values to change for each test
#         ['Real',               ['x'],     ['+', '-', '*', '/']],
#         ['Complex',            ['x','i'], ['+', '-', '*', '/', 'real', 'imag']],
#         ['Irrational Complex', ['x','i'], ['+', '-', '*', '/', 'real', 'imag', 'exp']],
#     ],
# }

kwargs = {
    'saves_path': '../../../saves/daggp/polynomial',  # Save path relative to this file
    'verbose': True,
    'parallelize': True,
    'checkpoint_interval': 500,
    'update_timeout': 60,  # Time before a replicate fails if it cannot update the database
    'save_formater_func': dag_to_save_str,  # Function to convert an individual into a savable string
    'load_formater_func': dag_from_save_str,  # Function to load an individual from a saved string
    ## Size ##
    'num_reps': 10,
    'num_gens': 1000,
    'pop_size': 100,
    'max_height': 10,
    ## Initialization ##
    'init_individual_func': random_tree,  # Function used to generate a new organism
    'init_max_height': 3,
    'p_branch': 0.75,  # Probability of a node not being a terminal
    'ops': ['+', '-', '*', '/', 'real', 'imag', 'exp'],
    'terminals': ['x','i'],
    ## Evaluation ##
    'eval_method': None,
    'fitness_func': dag_mse,
    # 'target_func': trig_sin,
    'target_func': polynomial,
    'coefficients': [0] + [1]*7,
    'domains': [list(np.linspace(-1,1,11))],
    ## Selection ##
    'minimize_fitness': True,
    'keep_parents': 2,  # Elitism, must be even
    'tournament_size': 2,  # Number of randomly chosen parents for each tournament
    ## Repopulation ##
    'subgraph_max_height': 2,
    'recombination_funcs': [subgraph_crossover],
    'recombination_probs': [0.25],
    'mutation_funcs': [subgraph_mutation, pointer_mutation],
    'mutation_probs': [0.25, 0.5],
    ## Tests ##
    'test_label': 'Field',  # Label to use when comparing all tests
    'test_keys': ['test', 'terminals', 'ops'],  # Keys of each parameter to be changed for each test
    'test_values': [  # Tuple of tuples representing all values to change for each test
        ['Real',                   ['x'],      ['+', '-', '*', '/']],
        ['Real Pow',               ['x', -1],      ['+', '-', '*', '/', '**']],
        # ['Complex',                ['x', 'i'], ['+', '-', '*', '/']],
        # ['Complex Pow',            ['x', 'i'], ['+', '-', '*', '/', '**']],
        ['Irrational Complex',     ['x', 'i', 'pi'], ['+', '-', '*', '/', 'exp']],
        # ['Irrational Complex Pow', ['x', 'i'], ['+', '-', '*', '/', '**', 'exp']],
        # ['Irrational', [0,1,2], ['+', '-', '*', '/', 'ln', 'exp']],
    ],
}



# kwargs = {
#     'saves_path': '../../../saves/daggp/pi',  # Save path relative to this file
#     'verbose': True,
#     'parallelize': True,
#     'checkpoint_interval': 500,
#     'update_timeout': 60,  # Time before a replicate fails if it cannot update the database
#     'save_formater_func': dag_to_save_str,  # Function to convert an individual into a savable string
#     'load_formater_func': dag_from_save_str,  # Function to load an individual from a saved string
#     ## Size ##
#     'num_reps': 30,
#     'num_gens': 200,
#     'pop_size': 100,
#     'max_height': 10,
#     ## Initialization ##
#     'init_individual_func': random_tree,  # Function used to generate a new organism
#     'init_max_height': 3,
#     'p_branch': 0.75,  # Probability of a node not being a terminal
#     'ops': ['+', '-', '*', '/', 'real', 'imag', 'exp'],
#     'terminals': ['x','i'],
#     ## Evaluation ##
#     'eval_method': None,
#     'fitness_func': dag_mse,
#     # 'target_func': trig_sin,
#     'target_func': polynomial,
#     'coefficients': [np.pi],
#     'domains': [list(np.linspace(-4*np.pi, 4*np.pi,31))],
#     ## Selection ##
#     'minimize_fitness': True,
#     'keep_parents': 2,  # Elitism, must be even
#     'tournament_size': 2,  # Number of randomly chosen parents for each tournament
#     ## Repopulation ##
#     'subgraph_max_height': 2,
#     'recombination_funcs': [subgraph_crossover],
#     'recombination_probs': [0.25],
#     'mutation_funcs': [subgraph_mutation, pointer_mutation],
#     'mutation_probs': [0.25, 0.5],
#     ## Tests ##
#     'test_label': 'Field',  # Label to use when comparing all tests
#     'test_keys': ['test', 'terminals', 'ops'],  # Keys of each parameter to be changed for each test
#     'test_values': [  # Tuple of tuples representing all values to change for each test
#         # ['Real',                   ['x'],      ['+', '-', '*', '/']],
#         # ['Real Pow',               ['x'],      ['+', '-', '*', '/', '**']],
#         # ['Complex',                ['x', 'i'], ['+', '-', '*', '/']],
#         # ['Complex Pow',            ['x', 'i'], ['+', '-', '*', '/', '**']],
#         # ['Irrational Complex',     ['x', 'i'], ['+', '-', '*', '/', 'exp']],
#         # ['Irrational Complex Pow', ['x', 'i'], ['+', '-', '*', '/', '**', 'exp']],
#         ['Irrational', ['x'], ['+', '-', '*', '/', 'ln', 'exp']],
#     ],
# }










# kwargs = {
#     'saves_path': '../../../saves/daggp/fix_pole',  # Save path relative to this file
#     'verbose': True,
#     'parallelize': True,
#     'checkpoint_interval': 500,
#     'update_timeout': 60,  # Time before a replicate fails if it cannot update the database
#     'save_formater_func': dag_to_save_str,  # Function to convert an individual into a savable string
#     'load_formater_func': dag_from_save_str,  # Function to load an individual from a saved string
#     ## Size ##
#     'num_reps': 10,
#     'num_gens': 50,
#     'pop_size': 20,
#     'max_height': 6,
#     ## Initialization ##
#     'init_individual_func': random_tree,  # Function used to generate a new organism
#     'init_max_height': 2,
#     'p_branch': 0.75,  # Probability of a node not being a terminal
#     'ops': ['+', '-', '*', '/', 'real', 'imag', 'exp'],
#     'terminals': ['x0','x1','x2','x3','i'],
#     ## Evaluation ##
#     'fitness_threshold': 12000,
#     'fitness_func': fix_cart_pole_fitness,
#     'eval_method': None,
#     'timeout': 12000,
#     ## Selection ##
#     'minimize_fitness': False,
#     'keep_parents': 2,  # Elitism, must be even
#     'tournament_size': 2,  # Number of randomly chosen parents for each tournament
#     ## Repopulation ##
#     'subgraph_max_height': 2,
#     'recombination_funcs': [subgraph_crossover],
#     'recombination_probs': [0.25],
#     'mutation_funcs': [subgraph_mutation, pointer_mutation],
#     'mutation_probs': [0.25, 0.5],
#     ## Tests ##
#     'test_label': 'Field',  # Label to use when comparing all tests
#     'test_keys': ['test', 'terminals', 'ops'],  # Keys of each parameter to be changed for each test
#     'test_values': [  # Lists representing all values to change for each test
#         ['Real',               ['x0','x1','x2','x3'],     ['+', '-', '*', '/']],
#         ['Complex',            ['x0','x1','x2','x3','i'], ['+', '-', '*', '/', 'real', 'imag']],
#         ['Irrational Complex', ['x0','x1','x2','x3','i'], ['+', '-', '*', '/', 'real', 'imag', 'exp']],
#     ],
# }




# kwargs = {
#     'saves_path': '../../../saves/daggp/fix_pole',  # Save path relative to this file
#     'verbose': True,
#     'parallelize': True,
#     'checkpoint_interval': 500,
#     'update_timeout': 60,  # Time before a replicate fails if it cannot update the database
#     'save_formater_func': dag_to_save_str,  # Function to convert an individual into a savable string
#     'load_formater_func': dag_from_save_str,  # Function to load an individual from a saved string
#     ## Size ##
#     'num_reps': 10,
#     'num_gens': 10,
#     'pop_size': 10,
#     'max_height': 6,
#     ## Initialization ##
#     'init_individual_func': random_tree,  # Function used to generate a new organism
#     'init_max_height': 2,
#     'p_branch': 0.75,  # Probability of a node not being a terminal
#     'ops': ['+', '-', '*', '/', 'real', 'imag', 'exp'],
#     'terminals': ['x0','x1','x2','x3','i'],
#     ## Evaluation ##
#     # 'fitness_threshold': 300,
#     'fitness_func': cart_pole_fitness,
#     'gradual_fitness': True,
#     'timeout': 200,
#     'init_angle': np.pi,
#     'angle_boundary': 120 * np.pi/180,
#     'position_boundary': 10,
#     'eval_method': None,
#     ## Selection ##
#     'minimize_fitness': False,
#     'keep_parents': 2,  # Elitism, must be even
#     'tournament_size': 2,  # Number of randomly chosen parents for each tournament
#     ## Repopulation ##
#     'subgraph_max_height': 2,
#     'recombination_funcs': [subgraph_crossover],
#     'recombination_probs': [0.25],
#     'mutation_funcs': [subgraph_mutation, pointer_mutation],
#     'mutation_probs': [0.25, 0.5],
#     ## Tests ##
#     'test_label': 'Field',  # Label to use when comparing all tests
#     'test_keys': ['test', 'terminals', 'ops'],  # Keys of each parameter to be changed for each test
#     'test_values': [  # Lists representing all values to change for each test
#         ['Real',               ['x0','x1','x2','x3'],     ['+', '-', '*', '/']],
#         ['Complex',            ['x0','x1','x2','x3','i'], ['+', '-', '*', '/', 'real', 'imag']],
#         ['Irrational Complex', ['x0','x1','x2','x3','i'], ['+', '-', '*', '/', 'real', 'imag', 'exp']],
#     ],
# }



# x1+(x0*x3)


if __name__ == '__main__':
    run_tests(**kwargs)
    plot_results(**kwargs)
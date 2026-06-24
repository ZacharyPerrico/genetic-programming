from models.daggp.plot import plot_results
from models.daggp.methods import *
from src.evolve import run_tests

# kwargs = {
#     'saves_path': '../../../saves/daggp/test',  # Save path relative to this file
#     'verbose': True,
#     'parallelize': True,
#     'checkpoint_interval': 1000,
#     'update_timeout': 60,  # Time before a replicate fails if it cannot update the database
#     'save_formater_func': dag_to_save_str,  # Function to convert an individual into a savable string
#     'load_formater_func': dag_from_save_str,  # Function to load an individual from a saved string
#     ## Size ##
#     'num_reps': 10,
#     'num_gens': 300,
#     'pop_size': 100,
#     'max_height': 5,
#     ## Initialization ##
#     'init_individual_func': random_tree,  # Function used to generate a new organism
#     'init_max_height': 4,
#     'p_branch': 0.5,  # Probability of a node not being a terminal
#     'ops': ['+', '-', '*', '/', '**'],
#     'terminals': ['x'],
#     ## Evaluation ##
#     'eval_method': None,
#     'fitness_func': dag_mse,
#     'timeout': 16,  # Number of evaluation iterations before forced termination
#     'target_func': trig_sin,
#     'domains': [list(np.linspace(-np.pi,np.pi,15))],
#     ## Selection ##
#     'minimize_fitness': True,
#     'keep_parents': 2,  # Elitism, must be even
#     'tournament_size': 2,  # Number of randomly chosen parents for each tournament
#     ## Repopulation ##
#     'subgraph_max_height': 2,
#     'recombination_funcs': [subgraph_crossover],
#     'recombination_probs': [0.2],
#     'mutation_funcs': [subgraph_mutation, pointer_mutation],
#     'mutation_probs': [0.4, 0.3],
#     ## Tests ##
#     'test_label': 'Field',  # Label to use when comparing all tests
#     'test_keys': ['test_name', 'terminals', 'ops'],  # Keys of each parameter to be changed for each test
#     'test_values': [  # Tuple of tuples representing all values to change for each test
#         ['Real',               ['x'],         ['+', '-', '*', '/']],
#         ['Complex',            ['x','i'],     ['+', '-', '*', '/', 'real', 'imag']],
#         ['Irrational Complex', ['x','i'],     ['+', '-', '*', '/', 'real', 'imag', 'exp']],
#     ],
# }


# kwargs = {
#     'saves_path': '../../../saves/daggp/tuning_height',  # Save path relative to this file
#     'verbose': True,
#     'parallelize': True,
#     'checkpoint_interval': 1000,
#     'update_timeout': 60,  # Time before a replicate fails if it cannot update the database
#     'save_formater_func': dag_to_save_str,  # Function to convert an individual into a savable string
#     'load_formater_func': dag_from_save_str,  # Function to load an individual from a saved string
#     ## Size ##
#     'num_reps': 10,
#     'num_gens': 300,
#     'pop_size': 100,
#     'max_height': 5,
#     ## Initialization ##
#     'init_individual_func': random_tree,  # Function used to generate a new organism
#     'init_max_height': 4,
#     'p_branch': 0.5,  # Probability of a node not being a terminal
#     'ops': ['+', '-', '*', '/', 'real', 'imag', 'exp'],
#     'terminals': ['x','i'],
#     ## Evaluation ##
#     'eval_method': None,
#     'fitness_func': dag_mse,
#     'timeout': 16,  # Number of evaluation iterations before forced termination
#     'target_func': trig_sin,
#     'domains': [list(np.linspace(-np.pi,np.pi,15))],
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
#     'test_keys': ['test_name', 'recombination_probs', 'mutation_probs'],  # Keys of each parameter to be changed for each test
#     'test_values': [  # Tuple of tuples representing all values to change for each test
#         [f'{c} {sm}+{pm}', [c], [sm, pm]]
#         for c in [0.25, 0.5, 0.75]
#         for sm in [0.25, 0.5, 0.75]
#         for pm in [0.25, 0.5, 0.75]
#         if sm + pm <= 1
#     ],
# }




# kwargs = {
#     'saves_path': '../../../saves/daggp/koza_2_extended_domain',  # Save path relative to this file
#     'verbose': True,
#     'parallelize': True,
#     'checkpoint_interval': 500,
#     'update_timeout': 60,  # Time before a replicate fails if it cannot update the database
#     'save_formater_func': dag_to_save_str,  # Function to convert an individual into a savable string
#     'load_formater_func': dag_from_save_str,  # Function to load an individual from a saved string
#     ## Size ##
#     'num_reps': 10,
#     'num_gens': 1000,
#     'pop_size': 100,
#     'max_height': 10,
#     ## Initialization ##
#     'init_individual_func': random_tree,  # Function used to generate a new organism
#     'init_max_height': 6,
#     'p_branch': 0.75,  # Probability of a node not being a terminal
#     'ops': ['+', '-', '*', '/', 'real', 'imag', 'exp'],
#     'terminals': ['x','i'],
#     ## Evaluation ##
#     'eval_method': None,
#     'fitness_func': dag_mse,
#     'target_func': koza_2,
#     'domains': [list(np.linspace(-2,2,15))],
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
#     'test_keys': ['test_name', 'terminals', 'ops'],  # Keys of each parameter to be changed for each test
#     'test_values': [  # Tuple of tuples representing all values to change for each test
#         ['Real',               ['x'],     ['+', '-', '*', '/']],
#         ['Complex',            ['x','i'], ['+', '-', '*', '/', 'real', 'imag']],
#         ['Irrational Complex', ['x','i'], ['+', '-', '*', '/', 'real', 'imag', 'exp']],
#     ],
# }



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
#     'num_gens': 10,
#     'pop_size': 10,
#     'max_height': 10,
#     ## Initialization ##
#     'init_individual_func': random_tree,  # Function used to generate a new organism
#     'init_max_height': 6,
#     'p_branch': 0.75,  # Probability of a node not being a terminal
#     'ops': ['+', '-', '*', '/', 'real', 'imag', 'exp'],
#     'terminals': ['x','i'],
#     ## Evaluation ##
#     'eval_method': None,
#     'fitness_func': dag_mse,
#     'target_func': koza_2,
#     'domains': [list(np.linspace(-2,2,15))],
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
    'saves_path': '../../../saves/daggp/pole_test',  # Save path relative to this file
    'verbose': True,
    'parallelize': True,
    'checkpoint_interval': 500,
    'update_timeout': 60,  # Time before a replicate fails if it cannot update the database
    'save_formater_func': dag_to_save_str,  # Function to convert an individual into a savable string
    'load_formater_func': dag_from_save_str,  # Function to load an individual from a saved string
    ## Size ##
    'num_reps': 10,
    'num_gens': 50,
    'pop_size': 10,
    'max_height': 6,
    ## Initialization ##
    'init_individual_func': random_tree,  # Function used to generate a new organism
    'init_max_height': 2,
    'p_branch': 0.75,  # Probability of a node not being a terminal
    'ops': ['+', '-', '*', '/', 'real', 'imag', 'exp'],
    'terminals': ['x0','x1','x2','x3','i'],
    ## Evaluation ##
    'eval_method': None,
    'fitness_func': dag_pole_fitness,
    ## Selection ##
    'minimize_fitness': False,
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
    'test_values': [  # Lists representing all values to change for each test
        ['Real',               ['x0','x1','x2','x3'],     ['+', '-', '*', '/']],
        ['Complex',            ['x0','x1','x2','x3','i'], ['+', '-', '*', '/', 'real', 'imag']],
        ['Irrational Complex', ['x0','x1','x2','x3','i'], ['+', '-', '*', '/', 'real', 'imag', 'exp']],
    ],
}


if __name__ == '__main__':
    print('Starting...')
    run_tests(**kwargs)
    plot_results(**kwargs)
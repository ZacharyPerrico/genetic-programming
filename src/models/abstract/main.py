"""
    Standard format for main files
    Indented values correspond to the function above
"""
from evolve import run_tests


kwargs = {
    'saves_path': '../../../saves/daggp/imag_roots',  # Save path relative to this file
    'verbose': True,
    'parallelize': True,
    'checkpoint_interval': 500,
    'update_timeout': 60,  # Time before a replicate fails if it cannot update the database
    'save_formater_func': None,  # Function to convert an individual into a savable string
    'load_formater_func': None,  # Function to load an individual from a saved string
    ## Size ##
    'num_reps': 10,  # Number of times that each test will be repeated with a different seed
    'num_gens': 1000,
    'pop_size': 100,
    'max_height': 10,
    ## Initialization ##
    'init_individual_func': None,  # Function used to generate a new organism
        'init_max_height': 6,  # The max_height for only the initial generation
        'p_branch': 0.75,  # Probability of a node not being a terminal
        'ops': ['+', '-', '*', '/', 'real', 'imag', 'exp'],
        'terminals': ['x','i'],
    ## Evaluation ##
    'fitness_func': None,
        'target_func': None,  # Used for symbolic regression
        'domains': [],  # A list of lists containing all tested values in the domain of each variable
    ## Selection ##
    'minimize_fitness': True,
    'keep_parents': 2,  # Elitism value, must be even
    'tournament_size': 2,  # Number of randomly chosen parents for each tournament
    ## Repopulation ##
    'recombination_probs': [0.25],
    'recombination_funcs': [None],
        'subgraph_max_height': 2,
    'mutation_probs': [0.25],
    'mutation_funcs': [None],
    ## Tests ##
    'test_label': 'Field',  # Label indicating the meaning behind each test_name
    'test_keys': ['test_name', 'terminals', 'ops'],  # Keys of each parameter to be changed for each test
    'test_values': [  # List of lists representing all values to change for each test
        ['Real',               ['x'],     ['+', '-', '*', '/', '**']],
        ['Complex',            ['x','i'], ['+', '-', '*', '/', '**']],
        ['Irrational Complex', ['x','i'], ['+', '-', '*', '/', '**', 'exp']],
    ],
}



if __name__ == '__main__':
    run_tests(**kwargs)
    # plot_results(**kwargs)
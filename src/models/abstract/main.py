
kwargs = {
    'saves_path': '../../../saves/daggp/imag_roots',  # Save path relative to this file
    'verbose': True,
    'parallelize': True,
    'checkpoint_interval': 500,
    'update_timeout': 60,  # Time before a replicate fails if it cannot update the database
    'save_formater_func': None,  # Function to convert an individual into a savable string
    'load_formater_func': None,  # Function to load an individual from a saved string
    ## Size ##
    'num_reps': 10,
    'num_gens': 1000,
    'pop_size': 100,
    'max_height': 10,
    ## Initialization ##
    'init_individual_func': None,  # Function used to generate a new organism
        'init_max_height': 6,
        'p_branch': 0.75,  # Probability of a node not being a terminal
        'ops': ['+', '-', '*', '/', 'real', 'imag', 'exp'],
        'terminals': ['x','i'],
    ## Evaluation ##
    'fitness_func': None,
        'target_func': None,
        'domains': [],
    ## Selection ##
    'minimize_fitness': True,
    'keep_parents': 2,  # Elitism, must be even
    'tournament_size': 2,  # Number of randomly chosen parents for each tournament
    ## Repopulation ##
    'subgraph_max_height': 2,
    'recombination_funcs': [None],
    'recombination_probs': [0.25],
    'mutation_funcs': [None],
    'mutation_probs': [0.25],
    ## Tests ##
    'test_label': 'Field',  # Label to use when comparing all tests
    'test_keys': ['test_name', 'terminals', 'ops'],  # Keys of each parameter to be changed for each test
    'test_values': [  # Tuple of tuples representing all values to change for each test
        ['Real',               ['x'],     ['+', '-', '*', '/', '**']],
        ['Complex',            ['x','i'], ['+', '-', '*', '/', '**']],
        ['Irrational Complex', ['x','i'], ['+', '-', '*', '/', '**', 'exp']],
    ],
}
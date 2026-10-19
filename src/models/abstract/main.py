"""
    Standard format for main files.
    Indented values correspond to the function above.
    Parenthesis indicate parameters only used by select models.
"""
from evolve import run_tests


kwargs = {
    'saves_path': '../../../saves/abstact/example',  # Save path relative to this file
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
    'max_height': 10,  # (DAGGP)
    'min_lens': [16],  # (SMLGP) Length of each memory bank excluding regs
    'max_lens': [64],  # (SMLGP) Length of each memory bank excluding regs
    'num_regs': 2,  # (SMLGP) Number of regs to be created
    'value_lim': 512,  # (SMLGP) One more than the largest value that can be used by the model
    ## Initialization ##
    'init_individual_func': None,  # Function used to generate a new organism
        'init_max_height': 6,  # (DAGGP) The max_height for only the initial generation
        'p_branch': 0.75,  # (DAGGP) Probability of a node not being a terminal
        'init_min_lens': [16],  # (SMLGP) Length of each memory bank excluding regs
        'init_max_lens': [32],  # (SMLGP) Length of each memory bank excluding regs
        'ops': ['+', '-', '*', '/', 'real', 'imag', 'exp'],  # List of all valid operations
        'terminals': ['x','i'],  # (SMLGP)
    ## Evaluation ##
    'fitness_func': None,
        'timeout': 16,  # (SMLGP) Number of evaluation iterations before forced termination
        'target_func': None,  # Used for symbolic regression
            'fitness_cases': [],  #
            'domains': [],  # A list of lists containing all tested values in the domain of each variable
        'coefficients': [],  # List of coefficients used instead of target_func
    ## Selection ##
    'minimize_fitness': True,
    'keep_parents': 2,  # Elitism, must be even
    'tournament_size': 2,  # Number of randomly chosen parents for each tournament
    ## Repopulation ##
    'recombination_probs': [0.25],
    'recombination_funcs': [None],
        'subgraph_max_height': 2,  # (DAGGP)
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
from evolve import run_sims
from basic import *
from utils import *

# Default kwargs
kwargs = {
    # The random seed to use for random and Numpy.random
    'seed': None,

    'function': lambda x: x**5 - 2*x**3 + x,
    'fitness_func': fitness_func,

    'target': 'PROGRAMMING',

    'num_runs': 10,

    'gen_individual': gen_individual,

    'init_tree_depth': 4,

    'pop_size': 600,

    # Number of randomly chosen parents for each tournament
    'k': 4,

    # Probability of crossover
    'p_c': 0.9,

    # Must be even
    'keep_parents': 4,

    'max_subtree_depth': 4,
    'max_tree_depth': 400,

    # Function used to create next generation
    'crossover_func': crossover,

    # Probability of a bit mutating
    'p_m': 0.5,

    # Function used to create next generation
    'mutate_func': subtree_mutation,

    # Generations before timeout
    'max_gen': 2,

    # Print changes and updates
    'verbose': 1,

    # Simplify algebraicly before evaluating
    'algebraic': False,

    # The domain of the problem expressed using np.linspace
    'x_linspace': (-1,1,21),

    # The valid leaves of the tree
    'terminals': ['x'],
}

if __name__ == '__main__':

    # kwargs['label_title'] = 'Types of Terminals'
    # kwargs['key'] = 'terminals'
    # kwargs['labels'] = ['$x$ and -5 to 5', '$x$ only']
    # kwargs['values'] = [['x',-5,-4,-3,-2,-1,0,1,2,3,4,5], ['x']]

    kwargs['label_title'] = 'Types of Terminals'
    kwargs['labels'] = ['Basic', 'Advanced']
    kwargs['key'] = 'ops'
    kwargs['values'] = [
        {
            '+': 2,
            '-': 2,
            '*': 2,
            '/': 2,
        },{
            '+': 2,
            '-': 2,
            '*': 2,
            '/': 2,
            # '**': 2,
            'min': 2,
            'max': 2,
            'abs': 1,
            'if_then_else': 3,
            '&': 2,
            '|': 2,
        }
    ]

    # Run simulation
    all_pops, all_fits = run_sims(**kwargs)
    # save_all(all_fits, all_pops, kwargs)
    plot_sims(all_pops, all_fits, **kwargs)
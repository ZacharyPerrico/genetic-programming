import random
import numpy as np
from node import *
from evolve import run_sims
from utils import *


# All functions relevant to genetic programming.

#
# Initialization
#

def gen_individual(init_tree_depth, ops, terminals, p_branch=.5,  init_call=True, **kwargs):
    """Generate a random tree"""
    # Create a branch with an operator value
    if init_call or random.random() < p_branch and init_tree_depth > 0:
        op = random.choice(list(ops.keys()))
        children = [gen_individual(init_tree_depth-1, ops, terminals, p_branch,False) for _ in range(ops[op])]
        return Node(op, children)
    # Create a leaf
    else:
        return Node(random.choice(terminals))

#
# Evaluation
#

# def function(x):
#     """The objective function"""
#     return x ** 5 - 2 * x ** 3 + x

def fitness_func(pop, function, x_linspace, algebraic, **kwargs):
    """Calculate the fitness value of all chromosomes in a population"""

    xs = np.linspace(*x_linspace)

    y_true = function(xs)

    fits = []

    for node in pop:

        y_node = [node(x, algebraic) for x in xs]

        # fit = (sum((abs(y_true - y_node)) ** 2) / len(xs)) ** (1/2)

        fit = sum(abs(y_true - y_node))

        fits.append(fit)

    return fits

#
# Mutation
#

def subtree_mutation(x, p_m, verbose, **kwargs):
    """Preform a mutation with a probability of p_m"""

    # Create a copy of x
    x = x.copy()

    # List of all nodes
    x_nodes = x.nodes()

    # List of all nodes with no children
    x_nodes = [xn for xn in x.nodes() if len(xn) == 0]

    node = random.choice(x_nodes)

    # Probability of mutation
    if random.random() < p_m:

        if verbose > 1:
            old_x = x.copy()

        new_branch = kwargs['gen_individual'](**kwargs)
        new_x = node.replace(new_branch)

        if verbose > 1:
            print(f'Mutation: {old_x} replaces {node} with {new_branch} returns {new_x}')

        x = new_x

    return x

#
# Reproduction
#

def crossover(a, b, max_subtree_depth, max_tree_depth, verbose, **kwargs):

    # Copy original trees
    a_new = a.copy()
    b_new = b.copy()

    a_depth = a.depth()
    b_depth = b.depth()

    # List of all nodes with children
    a_parent_nodes = [an for an in a_new.nodes() if an.depth() <= max_subtree_depth]

    # Select the first random node (branch)
    a_parent_node = random.choice(a_parent_nodes)
    a_parent_node_depth = a_parent_node.depth()

    # List of all nodes that could swap with a without being too long in the worse case
    # TODO implement a more accurate assessment of length
    b_parent_nodes = [bn for bn in b_new.nodes() if bn.depth() <= max_subtree_depth
                      and b_depth - bn.depth() + a_parent_node_depth <= max_tree_depth
                      and a_depth + bn.depth() - a_parent_node_depth <= max_tree_depth
                      ]

    # Select a random node with children
    b_parent_node = random.choice(b_parent_nodes)

    # Swap the two nodes
    a_parent_node.replace(b_parent_node.copy())
    b_parent_node.replace(a_parent_node.copy())

    if verbose > 1:
        print(f'Crossover: {a}  &  {b}  ->  {a_new}  &  {b_new}')

    return a_new, b_new



# Default kwargs
kwargs = {
    # The random seed to use for random and Numpy.random
    'seed': None,
    'num_runs': 10,
    'num_gens': 2,
    'pop_size': 600,
    'max_tree_depth': 400,
    'max_subtree_depth': 4,
    'verbose': 1,
    'algebraic': False, # Simplify algebraicly before evaluating
    'terminals': ['x'], # The valid leaves of the tree
    'fitness_func': fitness_func,
    'function': lambda x: x**5 - 2*x**3 + x,
    'x_linspace': (-1,1,21), # The domain of the problem expressed using np.linspace
    'gen_individual': gen_individual,
    'init_tree_depth': 4,
    'crossover_func': crossover, # Function used to create next generation
    'k': 4, # Number of randomly chosen parents for each tournament
    'p_c': 0.9, # Probability of crossover
    'keep_parents': 4, # Must be even
    'mutate_func': subtree_mutation, # Function used to create next generation
    'p_m': 0.5, # Probability of a bit mutating
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

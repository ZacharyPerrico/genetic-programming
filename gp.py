import random
from tabnanny import verbose

import numpy as np
from node import Node

#
# Initialization
#

def gen_chromosome(tree_depth, p_branch=.5, leaves=['x'], init_call=True, **kwargs):
    """Generate a random tree"""
    # Create a branch with an operator value
    if init_call or random.random() < p_branch and tree_depth > 0:
        # Pick operation
        op = random.choice(Node.ops)
        # Generate children
        c0 = gen_chromosome(tree_depth-1, p_branch, leaves, False)
        c1 = gen_chromosome(tree_depth-1, p_branch, leaves, False)
        # Return new Node
        return Node(op, [c0, c1])
    # Create a leaf
    else:
        return Node(random.choice(leaves))

def gen_pop(M, **kwargs):
    """Generate a random population"""
    return [gen_chromosome(**kwargs) for _ in range(M)]

#
# Evaluation
#

def function(x):
    """The objective function"""
    return x ** 5 - 2 * x ** 3 + x
    # return x * 0 + 4923

def fitness(pop, **kwargs):
    """Calculate the fitness value of all chromosomes in a population"""

    xs = np.linspace(*kwargs['x_linspace'])

    y_true = function(xs)

    fits = []

    for node in pop:

        y_node = [node(x, kwargs['algebraic']) for x in xs]

        # fit = (sum((abs(y_true - y_node)) ** 2) / len(xs)) ** (1/2)

        fit = sum(abs(y_true - y_node))

        fits.append(fit)

    return fits

#
# Mutation
#

def subtree_mutation(x, p_m, **kwargs):
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

        if kwargs['verbose'] > 1:
            old_x = x.copy()

        new_branch = gen_chromosome(**kwargs)
        new_x = node.replace(new_branch)

        if kwargs['verbose'] > 1:
            print(f'Mutation: {old_x} replaces {node} with {new_branch} returns {new_x}')

        x = new_x

    return x

#
# Reproduction
#

def select_parent(pop, fits, k, **kwargs):
    """Select a single parent from a tournament of k"""

    # Select the random tournament
    tourn = random.choices(pop, k=k)

    # Created a zipped list of fitness and chromosomes
    parent = [(fits[i], i) for i in range(len(tourn))]

    # Sort all parents by fitness
    parent = sorted(parent)
    # Get the chromosome of the first element
    parent = tourn[parent[0][1]]

    return parent

def crossover(a, b, **kwargs):

    # Copy original trees
    a_new = a.copy()
    b_new = b.copy()

    a_depth = a.depth()
    b_depth = b.depth()

    # List of all nodes with children
    a_parent_nodes = [an for an in a_new.nodes() if an.depth() <= kwargs['max_crossover_depth']]

    # Select the first random node (branch)
    a_parent_node = random.choice(a_parent_nodes)
    a_parent_node_depth = a_parent_node.depth()

    # List of all nodes that could swap with a without being too long in the worse case
    # TODO implement a more accurate assessment of length
    b_parent_nodes = [bn for bn in b_new.nodes() if bn.depth() <= kwargs['max_crossover_depth']
                      and b_depth - bn.depth() + a_parent_node_depth <= kwargs['max_depth']
                      and a_depth + bn.depth() - a_parent_node_depth <= kwargs['max_depth']
    ]

    # Select a random node with children
    b_parent_node = random.choice(b_parent_nodes)

    # Swap the two nodes
    a_parent_node.replace(b_parent_node.copy())
    b_parent_node.replace(a_parent_node.copy())

    # # Select a random child index
    # a_child_node_index = random.randint(0, len(a_parent_node)-1)
    # b_child_node_index = random.randint(0, len(b_parent_node)-1)
    # # Copy the child
    # a_child_node_copy = a_parent_node[a_child_node_index].copy()
    # b_child_node_copy = b_parent_node[b_child_node_index].copy()
    # # Assign the new children to the opposite parent nodes
    # a_parent_node[a_child_node_index] = a_child_node_copy
    # b_parent_node[b_child_node_index] = b_child_node_copy

    if kwargs['verbose'] > 1:
        print(f'Crossover: {a}  &  {b}  ->  {a_new}  &  {b_new}')

    return a_new, b_new

#
# Simulation and Iteration
#

def next_pop(**kwargs):
    """Generate the next population"""

    new_pop = []

    # Add the fitness values to the kwargs to pass to other functions
    kwargs['fits'] = fitness(**kwargs)

    # Repeat until the new population is the same size as the old
    while len(new_pop) < len(kwargs['pop']):

        # Select two parents
        c0 = select_parent(**kwargs)
        c1 = select_parent(**kwargs)

        # Crossover
        if random.random() < kwargs['p_c']:
            # Call the provided crossover function
            c0, c1 = kwargs['crossover_func'](c0, c1, **kwargs)

        # Mutate children
        c0 = kwargs['mutate_func'](c0, **kwargs)
        c1 = kwargs['mutate_func'](c1, **kwargs)

        new_pop.append(c0)
        new_pop.append(c1)

    return new_pop, kwargs['fits']


def run_sim(**kwargs):

    # Set random seed
    if 'seed' in kwargs:
        random.seed(kwargs['seed'])
        np.random.seed(kwargs['seed'])

    # Initial population
    pop = gen_pop(**kwargs)

    # Initial history
    pop_history = [pop]

    # Initial fitness values
    fit_history = []

    for generation in range(kwargs['T_max']):

        if kwargs['verbose'] > 0:
            print(f'Generation {generation} of {kwargs["T_max"]}')

        # Next generation
        pop, fit = next_pop(pop=pop, **kwargs)

        # Save previous fitnesses
        fit_history.append(fit)

        # Save new population
        pop_history.append(pop)

    # Final fitness values
    fit_history.append(fitness(pop_history[-1], **kwargs))

    if kwargs['verbose'] > 0:
        print('Timeout reached')

    return pop_history, fit_history

def run_sims(label_title, key, labels, values, **kwargs):
    # All values of all chromosomes of all generations of all runs
    # This can be saved as a 4D array for easy manipulation and access
    all_pops = []
    all_fits = []

    for value in values:
        kwargs[key] = value
        # Append all values of all chromosomes of all generations
        pops, fits = run_sim(**kwargs)
        all_pops.append(pops)
        all_fits.append(fits)

    # Convert to NumPy arrays
    all_fits = np.array(all_fits)
    all_pops_new = np.empty(all_fits.shape, dtype=object)
    all_pops_new[:] = all_pops
    all_pops = all_pops_new

    return all_pops, all_fits

#
# Testing
#

if __name__ == '__main__':
    x = Node('x')
    a = gen_chromosome(2)
    b = gen_chromosome(2)
    a = (x + x) - x
    b = x * (x / x)
    c = crossover(a, b, verbose=2)
    print(a)
    print(b)
    print(c)
    print(b.depth())

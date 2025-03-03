from node import *
from evolve import *
from utils import *
import multiprocessing
from multiprocessing import Process

# All functions relevant to genetic programming.

#
# Initialization
#

def init_individual(init_tree_depth, ops, terminals, p_branch=0.5, init_call=True, **kwargs):
    """Generate a random tree"""
    # Create a branch with an operator value
    if init_call or random.random() < p_branch and init_tree_depth > 0:
        op = random.choice(ops)
        children = [init_individual(init_tree_depth - 1, ops, terminals, p_branch, False) for _ in range(Node.valid_ops[op])]
        return Node(op, children)
    # Create a leaf
    else:
        return Node(random.choice(terminals))

#
# Evaluation
#

def fitness_helper(node, xs, y_true):
    # node = pop[i]
    # y_node = [node(x) for x in xs]
    # fit = (sum((abs(y_true - y_node)) ** 2) / len(xs)) ** (1/2)
    # fit = sum(abs(y_true - y_node))
    # fits.append(fit)
    fit = node
    return fit

def fitness_func(pop, target_func, x_linspace, **kwargs):
    """Calculate the fitness value of all chromosomes in a population"""
    xs = np.linspace(*x_linspace)
    y_true = target_func(xs)
    fits = np.empty(len(pop))
    for i,node in enumerate(pop):
        y_node = [node(x) for x in xs]
        fit = (sum((abs(y_true - y_node)) ** 2) / len(xs)) ** (1/2)
        # fit = sum(abs(y_true - y_node))
        # fits.append(fit)
        fits[i] = fit

    # # args = [(node, xs, y_true) for node in pop]
    # args = [(node, xs, y_true) for node in range(len(pop))]
    #
    # with multiprocessing.Pool(processes=4) as pool:
    #     fits = pool.starmap(fitness_helper, args)

    # print(fits)

    fits = np.nan_to_num(fits, nan=1000000, posinf=1000000)

    return fits

#
# Mutation
#

def subtree_mutation(a, p_m, verbose, **kwargs):
    """Preform a mutation with a probability of p_m"""

    # Probability of mutation
    if random.random() < p_m:
        a = a.copy()
        # List of all nodes with no children
        a_nodes = [n for n in a.nodes() if len(n) == 0]
        old_brach = random.choice(a_nodes)

        if verbose > 1:
            old_a = a.copy()

        new_branch = kwargs['init_individual_func'](**kwargs)
        new_a = old_brach.replace(new_branch)

        if verbose > 1:
            print(f'Mutation: {old_a} replaces {old_brach} with {new_branch} returns {new_a}')

        a = new_a

    return a

#
# Reproduction
#

def crossover(a, b, max_subtree_depth, max_tree_depth, verbose, **kwargs):

    # Copy original trees
    a_new = a.copy()
    b_new = b.copy()

    a_depth = a.height()
    b_depth = b.height()

    # List of all nodes with children
    a_parent_nodes = [an for an in a_new.nodes() if an.height() <= max_subtree_depth]

    # Select the first random node (branch)
    a_parent_node = random.choice(a_parent_nodes)
    a_parent_node_depth = a_parent_node.height()

    # List of all nodes that could swap with a without being too long in the worse case
    # TODO implement a more accurate assessment of length
    b_parent_nodes = [bn for bn in b_new.nodes() if bn.height() <= max_subtree_depth
                      and b_depth - bn.height() + a_parent_node_depth <= max_tree_depth
                      and a_depth + bn.height() - a_parent_node_depth <= max_tree_depth
                      ]

    # Select a random node with children
    b_parent_node = random.choice(b_parent_nodes)

    # Swap the two nodes
    a_parent_node.replace(b_parent_node.copy())
    b_parent_node.replace(a_parent_node.copy())

    if verbose > 1:
        print(f'Crossover: {a}  &  {b}  ->  {a_new}  &  {b_new}')

    return a_new, b_new

#
# Problems
#

def target_func(x): return 10 * x**2 + x

#
# Default kwargs
#

kwargs = {
    'seed': None,
    'num_reps': 4,
    'num_gens': 100,
    'pop_size': 600, #600,
    'max_tree_depth': 200, #400,
    'max_subtree_depth': 4,
    'verbose': 1, # 0: no updates, 1: generation updates, 2: all updates
    'algebraic': False, # Simplify before evaluating
    'terminals': ('x',),
    'ops': ('+','-','*','/','**'),
    'init_individual_func': init_individual,
    'p_branch': 0.5,
    'init_tree_depth': 4,
    'fitness_func': fitness_func,
    'target_func': target_func,
    'x_linspace': (0, 15, 16),  # The domain of the problem expressed using np.linspace
    'crossover_func': crossover,
    'k': 4, # Number of randomly chosen parents for each tournament
    'p_c': 0.9, # Probability of crossover
    'keep_parents': 4, # Must be even
    'mutate_func': subtree_mutation,
    'p_m': 0.5, # Probability of a bit mutating
}

if __name__ == '__main__':

    kwargs['name'] = 'const2'
    kwargs['label_title'] = 'Types of Operations'
    kwargs['labels'] = [
        '4-ops',
        '5-ops',
        'all-ops'
    ]
    kwargs['key'] = 'ops'
    kwargs['values'] = [
        ['+', '-', '*', '/'],
        ['+', '-', '*', '/', '**'],
        ['+', '-', '*', '/', '**', 'min', 'max', 'abs', 'if_then_else', '&', '|']
    ]

    # Run simulation
    all_pops, all_fits = run_sims(**kwargs)
    plot_sims(all_pops, all_fits, **kwargs)
    save_all(all_pops, all_fits, kwargs)

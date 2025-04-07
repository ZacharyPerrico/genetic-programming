import multiprocessing

import numpy as np
from scipy.optimize import minimize

from node import *
from evolve import *
from plot import *
from utils import save_all

from math import sin, cos

"""Functions relevant to implementing genetic programming"""

#
# Initialization Functions
#

def random_tree(init_tree_depth, ops=Node.valid_ops, terminals=['x'], p_branch=0.5, init_call=True, **kwargs):
    """Generate a random tree"""
    # Create a branch with an operator value
    if init_call or random.random() < p_branch and init_tree_depth > 0:
        op = random.choice(ops)
        children = [random_tree(init_tree_depth - 1, ops, terminals, p_branch, False) for _ in range(Node.valid_ops[op])]
        return Node(op, children)
    # Create a leaf
    else:
        return Node(random.choice(terminals))

def random_noop_tree(init_tree_depth, ops=Node.valid_ops, terminals=['x'], p_branch=0.5, **kwargs):
        c = [random_tree(init_tree_depth, ops, terminals, p_branch=0.5, init_call=True, **kwargs) for _ in range(5)]
        return Node('noop', c)

#
# Fitness Functions
#

def fitness_helper(id, node, xs, y_target):
    """Used for parallel computing"""
    y_actual = [node(*x) for x in xs]
    fit = (sum((abs(y_target - y_actual)) ** 2) / len(xs)) ** (1 / 2)
    return fit


def mse(pop, target_func, domains, **kwargs):
    """Calculate the fitness value of all individuals in a population against the target function for the provided domain"""
    xs = [np.linspace(*domain) for domain in domains]
    xs = np.array(np.meshgrid(*xs)).reshape((len(xs), -1)).T
    y_target = np.array([target_func(*list(x)) for x in xs])
    xs = xs.swapaxes(0, 1)
    fits = np.empty(len(pop))
    for i,node in enumerate(pop):
        # y_actual = [node(*x, eval_method=kwargs['eval_method']) for x in xs]
        y_actual = node(*xs, eval_method=kwargs['eval_method'])
        fit = (sum((abs(y_target - y_actual)) ** 2) / len(xs)) ** (1/2)
        fits[i] = fit
    # args = [(id, node, xs, y_target) for id,node in enumerate(pop)]
    # with multiprocessing.Pool(processes=4) as pool:
    #     fits = pool.starmap(fitness_helper, args)
    fits = np.nan_to_num(fits, nan=np.inf, posinf=np.inf)
    return fits


def correlation(pop, target_func, domains, is_final=False, **kwargs):
    """Calculate the fitness value of all individuals in a population against the target function for the provided domain"""
    xs = [np.linspace(*domain) for domain in domains]
    xs = np.array(np.meshgrid(*xs)).reshape((len(xs), -1)).T
    y_target = np.array([target_func(*list(x)) for x in xs])
    xs = xs.swapaxes(0,1)
    y_target_mean = np.mean(y_target)
    fits = np.empty(len(pop))
    for i,node in enumerate(pop):
        # y_actual = np.array([node(*x, eval_method=kwargs['eval_method']) for x in xs])
        y_actual = node(*xs, eval_method=kwargs['eval_method'])
        y_actual_mean = np.mean(y_actual)
        sum_target_actual = sum((y_target - y_target_mean) * np.conjugate(y_actual - y_actual_mean))
        sum_target_2 = sum(abs((y_target - y_target_mean))**2)
        sum_actual_2 = sum(abs((y_actual - y_actual_mean))**2)
        R = sum_target_actual / (sum_target_2 * sum_actual_2) ** (1/2)
        fit = 1 - R * np.conjugate(R)
        fits[i] = fit

        # Post-processing
        if is_final:
            def min_f(a): return np.sum(np.abs(y_target - (a[1] * y_actual + a[0])))
            res = minimize(min_f, [0,0], method='Nelder-Mead', tol=1e-6)
            new_node = (node * float(res.x[1])) + float(res.x[0])
            pop[i] = new_node
    # Replace inf and nan to arbitrary large values
    fits = np.nan_to_num(fits, nan=np.inf, posinf=np.inf)
    return fits


# def final_correlation(pop, target_func, domains, **kwargs):
#     """Calculate the fitness value of all individuals in a population against the target function for the provided domain"""
#     xs = [np.linspace(*domain) for domain in domains]
#     xs = np.array(np.meshgrid(*xs)).reshape((len(xs), -1)).T
#     y_target = np.array([target_func(*list(x)) for x in xs])
#     xs = xs.swapaxes(0, 1)
#     y_target_mean = np.mean(y_target)
#     fits = np.empty(len(pop))
#     for i,node in enumerate(pop):
#         # y_actual = np.array([node(*x) for x in xs])
#         y_actual = node(*xs, eval_method=kwargs['eval_method'])
#         y_actual_mean = np.mean(y_actual)
#         sum_target_actual = sum((y_target - y_target_mean) * np.conjugate(y_actual - y_actual_mean))
#         sum_target_2 = sum(abs((y_target - y_target_mean))**2)
#         sum_actual_2 = sum(abs((y_actual - y_actual_mean))**2)
#         R = sum_target_actual / (sum_target_2 * sum_actual_2) ** (1/2)
#         fit = 1 - R * np.conjugate(R)
#         fits[i] = fit
#
#
#     # Replace inf and nan to arbitrary large values
#     fits = np.nan_to_num(fits, nan=np.inf, posinf=np.inf)
#     return fits

#
# Mutation Functions
#

def randomize_mutation(a, **kwargs):
    """Preform a mutation with a probability of p_m"""
    return kwargs['init_individual_func'](**kwargs)

def subtree_mutation(a, **kwargs):
    """Preform a mutation with a probability of p_m"""
    # Probability of mutation
    # if random.random() < p_m:
    a = a.copy()
    # List of all nodes with no children
    a_nodes = [n for n in a.nodes() if len(n) == 0]
    old_branch = random.choice(a_nodes)
    if kwargs['verbose'] > 1:
        old_a = a.copy()
    new_branch = kwargs['new_individual_func'](**kwargs)
    new_a = old_branch.replace(new_branch)
    if kwargs['verbose'] > 1:
        print(f'\tsubtree_mutation: {old_a} replaces {old_branch} with {new_branch} returns {new_a}')
    a = new_a
    return a

# def ancestor_split_mutation(a, p_m, verbose, **kwargs):
#     """Only the direct parent may """
#     # Probability of mutation
#     if random.random() < p_m:
#         a = a.copy()
#         # List of all nodes with multiple parents
#         a_nodes = [n for n in a.nodes() if len(n.parents) > 1]
#         old_branch = random.choice(a_nodes)
#         if verbose > 1:
#             old_a = a.copy()
#         new_branch = old_branch.copy()
#         old_branch.replace(new_branch)
#         a.reset_parents()
#         a.set_parents()
#         if verbose > 1:
#             print(f'\t\tMutation: {old_a} replaces {old_branch} with {new_branch} returns {a}')
#     return a

def split_mutation(a, **kwargs):
    """Only the direct parent may """
    # Probability of mutation
    # if random.random() < p_m:
    a = a.copy()
    # List of all nodes with multiple parents
    a_nodes = [n for n in a.nodes() if len(n.parents) > 1]
    # Mutation failed
    if len(a_nodes) == 0:
        print(f'\tsplit_mutation: failed for {a}')
        return a

    node = random.choice(a_nodes)
    if kwargs['verbose'] > 1:
        old_a = a.copy()
    for parent in node.parents:
        index = node.index_in(parent)
        new_node = Node(node.value, node.children)
        parent[index] = new_node
    a.reset_parents()
    a.set_parents()

    if kwargs['verbose'] > 1:
        print(f'\tsplit_mutation: {old_a} splits {node} returns {a}')
    return a

def pointer_mutation(root, **kwargs):
    old_root = root
    root = root.copy()
    # List of all nodes with multiple parents
    possible_parents = [n for n in root.nodes() if len(n.parents) > 0 and len(n.children) > 0]

    if len(possible_parents) == 0:
        if kwargs['verbose'] > 1:
            print(f'\tpointer_mutation: failed for {root}')
        return root

    parent = random.choice(possible_parents)
    child_index = np.random.randint(len(parent))
    child = parent[child_index]

    # descendants = parent.nodes()

    # possible_new_child = [n for n in root.nodes() if n.index_in(descendants) == -1]

    possible_new_child = [n for n in root.nodes() if parent.index_in(n.nodes()) == -1]

    new_child = random.choice(possible_new_child)
    parent[child_index] = new_child

    root.reset_parents()
    root.set_parents()

    if kwargs['verbose'] > 1:
        print(f'\tpointer_mutation: {old_root} replaces {child} with {new_child} returns {root}')
    return root

#
# Crossover Functions
#

def subtree_crossover(a, b, **kwargs):
    # Copy original trees
    a_new = a.copy()
    b_new = b.copy()
    a_height = a.height()
    b_height = b.height()
    # List of all nodes
    a_parent_nodes = [an for an in a_new.nodes() if an.height() <= kwargs['max_subtree_depth']]
    # Select the first random node (branch)
    a_parent_node = random.choice(a_parent_nodes)
    a_parent_node_height = a_parent_node.height()
    # List of all nodes that could swap with a without being too long in the worse case
    # TODO implement a more accurate assessment of length
    b_parent_nodes = [bn for bn in b_new.nodes() if bn.height() <= kwargs['max_subtree_depth']
                      and b_height - bn.height() + a_parent_node_height <= kwargs['max_tree_depth']
                      and a_height + bn.height() - a_parent_node_height <= kwargs['max_tree_depth']
                      ]
    # Select a random node with children
    b_parent_node = random.choice(b_parent_nodes)
    # Swap the two nodes
    a_parent_node.replace(b_parent_node.copy())
    b_parent_node.replace(a_parent_node.copy())

    if kwargs['verbose'] > 1:
        print(f'\tsubtree_crossover: {a} and {b} produce {a_new} and {b_new}')
    return a_new, b_new

#
# Target Functions
#

def logical_or(*x): return bool(x[0]) or bool(x[1])
def f(x): return x**5 - 2*x**3 + x
def mod2k(*x): return x[0] % (2 ** x[1])
def xor_and_xor(*x): return (int(x[0]) ^ int(x[1])) & (int(x[2]) ^ int(x[3]))
# def const_32(x): return x**5 + 32*x**3 + x
def const_32(x): return 32*x**2 + x

#
# Initial pops
#

def init_indiv(**kwargs):
    x_0 = Node('x_0')
    x_1 = Node('x_1')
    f = x_0 >> 2
    f = f.limited()
    return f

def init_sin(**kwargs): return Node.sin(Node('x')).limited()

#
# Debug
#

if __name__ == '__main__':
    e = Node('e')
    i = Node('i')
    x = Node('x')
    # f = e ** (i * x)
    # # f = Node.sin(x)
    # # f = 3j * (x * i + 3j)
    #
    # f1 = Node.cos(x)
    # f2 = i * Node.sin(x)
    #
    # def ff(x):
    #     return cos(x)
    #     # return x * 1j
    #
    # pop = [f, f1, f2]
    #
    # # fits = correlation(pop, ff, domains=[[0, 2*np.pi, 1]])
    #
    # plot_nodes(pop,
    #            labels=['$e^{ix}$', '$\\cos(x)$', '$\\sin(x)$'],
    #            target_func=ff,
    #            result_fitness_func=mse, #correlation,
    #            domains=[[0, 2*np.pi, 31]])



    # x = Node('x')
    #
    # # f1 = 1 + x
    # # f2 = x - f1
    # # f3 = f1 * f2
    # # f4 = f2 / f3
    # # f = f4
    #
    # # f1 = 1 + x
    # # f2 = f1 - x
    # # f3 = f2 * f1
    # # f4 = f3 / f2
    # # f = f4
    #
    # f = (x**5 - 2*x**3 + x).to_tree()
    #
    # # g1 = x / 1
    #
    # # (f1).replace(g1)
    #
    # plot_graph(f)
    # f = pointer_mutation(f)
    # plot_graph(f)



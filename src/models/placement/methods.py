"""
Genetic programming functions specifically for the evolution of linear models.
"""
import math

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from src.utils.utils import cartesian_prod

#
# Setup Functions
#

def setup_uniform_clients(**kwargs):
    """Uses the seed to generate the client positions and returns the updated kwargs"""
    rng = np.random.default_rng(kwargs['seed'])
    clients = rng.uniform(kwargs['min_value'], kwargs['max_value'], size=(kwargs['num_clients'], 2))
    kwargs['clients'] = clients
    return kwargs


#
# Initialization Functions
#

def random_uniform_router_coords(**kwargs):
    """Generate a random list of coords"""
    return kwargs['rng'].uniform(kwargs['min_value'], kwargs['max_value'], size=(kwargs['num_routers'],2))



#
# Fitness Functions
#

# def coverage(routers, **kwargs):
#     """Returns an array for each client indicating if they are covered or not"""
#     cov_arr = [False] * len(kwargs['clients'])
#     for i,client in enumerate(kwargs['clients']):
#         for j,router in enumerate(routers):
#             dist = np.linalg.norm(client - router)
#             if dist < kwargs['radius']:
#                 cov_arr[i] = True
#                 break
#     return cov_arr

def covered(routers, **kwargs):
    """Returns an array for each client indicating if they are covered or not"""
    cov_arr = [-1] * len(kwargs['clients'])
    for i,client in enumerate(kwargs['clients']):
        for j,router in enumerate(routers):
            dist = np.linalg.norm(client - router)
            if dist < kwargs['radius']:
                cov_arr[i] = j
                break
    return cov_arr

def calc_adj_mat(routers, **kwargs):
    adj_mat = np.zeros((len(routers),len(routers)), int)
    for i,r0 in enumerate(routers):
        for j,router in enumerate(routers):
            if i != j:
                dist = np.linalg.norm(r0 - router)
                if dist <= 2 * kwargs['radius']:
                    adj_mat[i,j] = 1
                    adj_mat[j,i] = 1
    return adj_mat

def connected(adj_mat, **kwargs):
    return connected_components(
        csgraph=csr_matrix(adj_mat),
        directed=False,
        return_labels=False
    )

def num_cov(pop, **kwargs):
    """Fitness function based on the total interference"""
    fits = np.empty(len(pop))
    for i,org in enumerate(pop):
        N_cov = sum(np.array(covered(org, **kwargs)) != -1)
        N_con = connected(calc_adj_mat(org, **kwargs))
        fits[i] = N_cov - N_con
    fits = np.array(fits)
    return fits




#
# Crossover Functions
#

def coords_one_point_crossover(a, b, **kwargs):
    cut_a = kwargs['rng'].integers(0, len(a) + 1)
    cut_b_min = max(cut_a + len(b) - kwargs['max_len'], cut_a - len(a) + kwargs['min_len'])
    cut_b_max = min(cut_a + len(b) - kwargs['min_len'], cut_a - len(a) + kwargs['max_len'])
    cut_b = kwargs['rng'].integers(cut_b_min, cut_b_max + 1)
    new_a = a[:cut_a] + b[cut_b:]
    new_b = b[:cut_b] + a[cut_a:]

    return new_a, new_b


def coords_two_point_crossover(a, b, **kwargs):
    """Modified crossover to work for fixed lengths"""
    min_len = len(a)
    max_len = len(a)
    a = list(a)
    b = list(b)
    # Difference in lengths of the sections to be swapped
    # diff_diff_cuts = len(a) - len(b)
    # kwargs['min_len'] <= len(a) + diff_diff_cuts <= kwargs['max_len']
    # kwargs['min_len'] <= len(b) - diff_diff_cuts <= kwargs['max_len']
    diff_diff_cuts_min = max(min_len - len(a), len(b) - max_len)
    diff_diff_cuts_max = min(max_len - len(a), len(b) - min_len)
    diff_diff_cuts = kwargs['rng'].integers(diff_diff_cuts_min, diff_diff_cuts_max + 1)
    # The length of a cut cannot be negative
    # 0 <= diff_cuts_a <= len(a)
    # 0 <= diff_cuts_a + diff_diff_cuts <= len(b)
    diff_cuts_a = kwargs['rng'].integers(max(0, -diff_diff_cuts), min(len(a), len(b) - diff_diff_cuts) + 1)
    diff_cuts_b = diff_cuts_a + diff_diff_cuts
    cut_a_0 = kwargs['rng'].integers(0, len(a) - diff_cuts_a + 1)
    cut_b_0 = kwargs['rng'].integers(0, len(b) - diff_cuts_b + 1)
    cut_a_1 = cut_a_0 + diff_cuts_a
    cut_b_1 = cut_b_0 + diff_cuts_b
    # Swap the two sections
    new_a = a[:cut_a_0] + b[cut_b_0:cut_b_1] + a[cut_a_1:]
    new_b = b[:cut_b_0] + a[cut_a_0:cut_a_1] + b[cut_b_1:]
    # assert kwargs['min_len'] <= len(new_a) <= kwargs['max_len']
    # assert kwargs['min_len'] <= len(new_b) <= kwargs['max_len']
    new_a = np.array(new_a)
    new_b = np.array(new_b)
    return new_a, new_b

#
# Mutation Functions
#

def coords_point_mutation(org, **kwargs):
    """Randomly change a value in a random line"""
    # Select a random value
    index = kwargs['rng'].integers(len(org))
    # Replace the argument
    org[index] = kwargs['rng'].uniform(kwargs['min_value'], kwargs['max_value'])
    return org

def coords_point_mutation_2d(org, **kwargs):
    """Randomly change a value in a random line"""
    # Select a random value
    index = kwargs['rng'].integers(len(org))
    # Replace the argument
    # org[index] = point_mutation(org[index], **kwargs)
    coords_point_mutation(org[index], **kwargs)
    return org

#
# Debug
#

if __name__ == '__main__':
    # pass

#     org =  [8,  4,  3,  0 ,
# 11,  0,  0, 13 ,
#  4, 13, 10,  1 ,
#  4, 15,  1,  8 ,
#  9,  1, 12, 13 ,
#  3, 13,  9,  1 ,
# 14, 13,  3, 13 ,
# 11, 11,  6,  3 ,
#  4, 13,  4,  0 ,
#  9, 14,  1, 10 ,
#  4,  3, 244,  7,
# 13,  0, 13,  5 ,
#  7,  4, 13, 10 ,
#  8,  6, 13,  0 ,
#  7, 10,  1,  1 ,
# 35, 13,  0, 11]
#
#
#     a = _check_sylver_coinage(16, [7])
#
#     print(a)

    # from main import kwargs
    #
    # kwargs['rng'] = np.random.default_rng()
    #
    # l = random_mems(**kwargs)
    # l = Linear(l)
    #
    # print(l)

    # prevs = [
    #     [5],
    #     [5,4],
    #     [5,4,11],
    #     [5,4,11,6],
    #     [5,4,11,6,7],
    #     [5,4,11,6,7,2],
    #     [5,4,11,6,7,2,3],
    # ]
    #
    # ns = range(15)
    #
    # vss = []
    # for prev in prevs:
    #     vs = []
    #     for n in ns:
    #         s = _check_sylver_coinage(n, prev)
    #         vs.append(n * s)
    #     vss.append(vs)
    #
    # # prev = [5]
    # # n = 10
    # # a = _check_sylver_coinage(n, prev)
    #
    # vss = np.array(vss, int)
    #
    # print(vss)

    a = [
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 0],
    ]

    b = n_con(a)
    print(b)
"""
Genetic programming functions usable by most models.
"""
import math

import numpy as np

from utils.utils import cartesian_prod


# from utils.utils import cartesian_prod


#
# Saving Functions (save_formater_func)
#

#
# Loading Functions (load_formater_func)
#

#
# Initialization Functions (init_individual_func)
#

#
# Target Functions (target_func)
#

def real_dist(x0,x1): return (x0**2 + x1**2)**.5
def trig_sin(x): return np.sin(x)

def multiply(x0,x1): return x0 * x1
def power(x0,x1): return x0 ** x1
def factorial(x): return math.factorial(x)
def triangular(x): return x*(x + 1)//2
def sum_squares(x): return x*(x+1)*(2*x+1)/6

def logical_or(*x): return bool(x[0]) or bool(x[1])
def mod2k(*x): return x[0] % (2 ** x[1])
def xor_and_xor(*x): return (int(x[0]) ^ int(x[1])) & (int(x[2]) ^ int(x[3]))
def bit_sum(x): return sum(int(i) for i in f'{int(x):04b}')

def koza_1(x): return x**4 + x**3 + x**2 + x
def koza_2(x): return x**5 - 2*x**3 + x
def koza_3(x): return x**6 - 2*x**4 + x**2

def nguyen_1(x): return                      x**3 + x**2 + x
def nguyen_2(x): return               x**4 + x**3 + x**2 + x
def nguyen_3(x): return        x**5 + x**4 + x**3 + x**2 + x
def nguyen_4(x): return x**6 + x**5 + x**4 + x**3 + x**2 + x
def nguyen_5(x): return np.sin(x**2) * np.cos(x) - 1
def nguyen_6(x): return np.sin(x) + np.sin(x + x**2)
def nguyen_7(x): return np.log(x + 1) + np.log(x**2 + 1)
def nguyen_8(x): return np.sqrt(x)
def nguyen_9(x,y): return np.sin(x) + np.sin(y**2)
def nguyen_10(x,y): return 2 * np.sin(x) * np.cos(y)
def nguyen_11(x,y): return x ** y
def nguyen_12(x,y): return x**4 - x**3 + y**2/2 - y


#
# Fitness Functions (fitness_func)
#

def mse(pop, rmse=False, **kwargs):
    """Calculate the fitness value of all individuals in a population against the target function for the provided domain"""
    # 2D array of input variables for each test case
    cases = cartesian_prod(*kwargs['domains'])
    y_target = np.array([kwargs['target_func'](*list(case)) for case in cases])
    fits = np.empty(len(pop))
    for i, org in enumerate(pop):
        y_actual = org(*cases, eval_method=kwargs['eval_method'])
        # y_actual = []
        # for case in cases:
        #     # l = Linear([[0]*kwargs['num_regs'], org[0].copy()], ops=kwargs['ops'], value_lim=kwargs['value_lim'])
        #     for j,x in enumerate(case):
        #         l.mem[0][1+j] = x
        #     # Evaluate the organism
        #     l.run(kwargs['timeout'])
        #     y_actual.append(l.mem[0][-1])
        # Calculate MSE
        fits[i] = sum((abs(y_target - y_actual)) ** 2) / len(cases)
        # Calculate RMSE
        if rmse:
            fits[i] **= 0.5
    return fits

#
# Recombination Functions (recombination_funcs)
#

def one_point_crossover(a, b, **kwargs):
    cut_a = kwargs['rng'].integers(0, len(a) + 1)
    cut_b_min = max(cut_a + len(b) - kwargs['max_len'], cut_a - len(a) + kwargs['min_len'])
    cut_b_max = min(cut_a + len(b) - kwargs['min_len'], cut_a - len(a) + kwargs['max_len'])
    cut_b = kwargs['rng'].integers(cut_b_min, cut_b_max + 1)
    new_a = a[:cut_a] + b[cut_b:]
    new_b = b[:cut_b] + a[cut_a:]
    return new_a, new_b

def two_point_crossover(a, b, **kwargs):
    # Difference in lengths of the sections to be swapped
    # diff_diff_cuts = len(a) - len(b)
    # kwargs['min_len'] <= len(a) + diff_diff_cuts <= kwargs['max_len']
    # kwargs['min_len'] <= len(b) - diff_diff_cuts <= kwargs['max_len']
    diff_diff_cuts_min = max(kwargs['min_len'] - len(a), len(b) - kwargs['max_len'])
    diff_diff_cuts_max = min(kwargs['max_len'] - len(a), len(b) - kwargs['min_len'])
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
    assert kwargs['min_len'] <= len(new_a) <= kwargs['max_len']
    assert kwargs['min_len'] <= len(new_b) <= kwargs['max_len']
    return new_a, new_b

#
# Mutation Functions (mutation_funcs)
#


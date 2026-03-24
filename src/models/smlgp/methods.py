"""
Genetic programming functions specifically for the evolution of linear models.
Linear org is represented as a 2D arrays and converted to a Linear objects when evaluating.
"""
import copy
import math

import numpy as np
from fontTools.unicodedata import block

from src.models.smlgp.model import Linear
from src.utils.utils import cartesian_prod


#
# Initialization Functions
#

def random_mem(**kwargs):
    """Generate a random list of operations"""
    init_len = kwargs['rng'].integers(kwargs['init_min_len'], kwargs['init_max_len']+1)
    init_len -= init_len % 4
    code = kwargs['rng'].integers(0, kwargs['value_lim'], init_len)
    code = list(code)
    return code

def random_mems(**kwargs):
    """Generate a random list of mems"""
    code = []
    for i in range(len(kwargs['init_min_lens'])):
        init_min_len = kwargs['init_min_lens'][i]
        init_max_len = kwargs['init_max_lens'][i]
        code.append(random_mem(init_min_len=init_min_len, init_max_len=init_max_len, **kwargs))
    return code



def _random_contextual_line(**kwargs):
    """Helper function to generating a single line of org"""
    return [
        kwargs['rng'].integers(len(kwargs['ops']), dtype=int),
        kwargs['rng'].integers(kwargs['num_regs'], dtype=int),
        kwargs['rng'].integers(kwargs['value_lim'], dtype=int),
        # kwargs['rng'].integers(16, dtype=int),
        kwargs['rng'].integers(1+2*(1+len(kwargs['max_lens'])), dtype=int),
        # kwargs['rng'].integers(7, dtype=int),
    ]

def _random_contextual_mem(**kwargs):
    """Generate a random list of transitions"""
    init_len = kwargs['rng'].integers(kwargs['init_min_len']//Linear.LINE_LENGTH, (kwargs['init_max_len']//Linear.LINE_LENGTH)+1)
    code = [_random_contextual_line(**kwargs) for _ in range(init_len)]
    code = sum(code, [])
    return code

def random_contextual_mems(**kwargs):
    code = []
    for i in range(len(kwargs['init_min_lens'])):
        init_min_len = kwargs['init_min_lens'][i]
        init_max_len = kwargs['init_max_lens'][i]
        code.append(_random_contextual_mem(init_min_len=init_min_len, init_max_len=init_max_len, **kwargs))
    return code


#
# Fitness Functions
#

def setup_self_rep(code, **kwargs):
    """Create a Linear object with blank registers, MEM, -1"""
    l = Linear([[0]*kwargs['num_regs'], code[0], [kwargs['value_lim']]*len(code[0])], ops=kwargs['ops'], value_lim=kwargs['value_lim'])
    return l

def self_rep_fitness(pop, **kwargs):
    """Calculate the fitness value of all individuals in a population"""
    fits = np.empty(len(pop))
    for i, code in enumerate(pop):
        l = setup_self_rep(code.copy(), **kwargs)
        l.run(kwargs['timeout'])
        # Fitness is number of values equal
        fit = sum([code[0][j] != l.mem[2][j] for j in range(len(code[0]))])
        fits[i] = fit
    return fits


def self_match_fitness(pop, **kwargs):
    """Calculate the fitness value of all individuals in a population"""
    fits = np.empty(len(pop))
    for i, code in enumerate(pop):
        # code = copy.deepcopy(code)
        v = kwargs['value_lim']
        # v = 0
        l = Linear([[0] * kwargs['num_regs'], code[0].copy(), [v] * len(code[0])], ops=kwargs['ops'],
                   value_lim=kwargs['value_lim'])
        l.run(kwargs['timeout'])
        # Fitness is number of values equal
        # fit = sum([l.mem[1][j] != l.mem[2][j] for j in range(len(code[0]))])
        fit_0 = sum([l.mem[1][j] != l.mem[2][j] for j in range(len(code[0]))])
        # fit_1 = (sum(l.mem[1]) - (kwargs['value_lim']//2*len(code[0]))) ** 2
        fit_1 = 0
        fits[i] = fit_0 + fit_1
    return fits





# def self_rep_fitness(pop, **kwargs):
#     """Calculate the fitness value of all individuals in a population"""
#     fits = np.empty(len(pop))
#
#     for i, org in enumerate(pop):
#         # Random org to be copied
#         c = [kwargs['rng'].integers(0, kwargs['value_lim'], len(org[0]))]
#         # Execute the self replication crossover method to get children
#         copied = self_rep_crossover(org, c.copy(), two_way=False, **kwargs)[0][0]
#
#         # Fitness is number of values equal
#         fit = sum(np.array(c[0]) != copied)
#         fits[i] = fit
#     return fits


def setup_self_crossover_overwrite(self_rep_code, a, b, **kwargs):
    """Use the self_rep_code to copy a onto b"""
    return Linear([[0]*4, self_rep_code.copy(), a.copy(), b.copy()], ops=kwargs['ops'], value_lim=kwargs['value_lim'])


def self_crossover_overwrite_fitness(pop, **kwargs):
    """Calculate the fitness value of all individuals in a population"""
    fits = np.empty(len(pop))

    for i, org in enumerate(pop):
        # Code used to replicate
        rep_code = org[0]
        # Random org to be copied
        a = kwargs['rng'].integers(0, kwargs['value_lim'], len(org[0]))
        b = kwargs['rng'].integers(0, kwargs['value_lim'], len(org[0]))
        # Prevent a == b for any values
        b = [(b[i] + 1) % kwargs['value_lim'] if a[i] == b[i] else b[i] for i in range(len(a))]
        # Execute the self replication crossover method to get children
        child = setup_self_crossover_overwrite(rep_code, a, b, **kwargs)
        child.run(kwargs['timeout'])
        child = child.mem[-1]
        child = np.array(child)

        # Values in c must be equal to a value in a or b
        a_matches = sum(child == a)
        b_matches = sum(child == b)

        fit = ((len(a)-a_matches)**2 + (len(b)-b_matches)**2) / (2*(len(a)/2)**2) - 1
        fits[i] = fit

    return fits


# def self_crossover_fitness(pop, **kwargs):
#     """Calculate the fitness value of all individuals in a population"""
#     fits = np.empty(len(pop))
#
#     for i, org in enumerate(pop):
#         # Random org to be copied
#         c = [kwargs['rng'].integers(0, kwargs['value_lim']+1, len(org[0]))]
#         # Execute the self replication crossover method to get children
#         result = self_crossover([org[0]], c.copy(), two_way=False, **kwargs)
#         # Fitness is number of values equal
#         fit = sum(np.array(c[0]) != result[0][0])
#         fits[i] = fit
#
#     return fits


def setup_self_crossover(self_rep_code, a, b, **kwargs):
    return Linear([[0]*4, self_rep_code.copy(), a.copy(), b.copy(), [-1]*len(a)], ops=kwargs['ops'], value_lim=kwargs['value_lim'])


def self_crossover_fitness(pop, **kwargs):
    """Calculate the fitness value of all individuals in a population"""
    fits = np.empty(len(pop))

    for i, org in enumerate(pop):
        # Code used to replicate
        rep_code = org[0]
        # Random org to be copied
        a = [[kwargs['value_lim']] * len(org[0])]
        # Prevent a == b for any values
        b = [(b[i] + 1) % kwargs['value_lim'] if a[i] == b[i] else b[i] for i in range(len(a))]
        # Execute the self replication crossover method to get children
        child = setup_self_crossover(rep_code, a, b, **kwargs)
        child.run(kwargs['timeout'])
        child = child.mem[-1]
        child = np.array(child)

        # Values in c must be equal to a value in a or b
        fit = sum(child != a)
        fits[i] = fit

    return fits


def setup_mutual_rep(self_rep_code, a, **kwargs):
    return Linear([[0]*4, self_rep_code.copy(), a.copy(), [kwargs['value_lim']]*len(a)], ops=kwargs['ops'], value_lim=kwargs['value_lim'])


def mutual_rep_fitness(pop, **kwargs):
    """Calculate the fitness value of all individuals in a population"""
    fits = np.empty(len(pop))

    for i, org in enumerate(pop):
        # Code used to replicate
        rep_code = org[0]
        # Execute the self replication crossover method to get children
        a = kwargs['rng'].integers(0, kwargs['value_lim'], len(org[0]))
        child = setup_mutual_rep(rep_code, a, **kwargs)
        child.run(kwargs['timeout'])
        child = child.mem[-1]
        child = np.array(child)

        # Values in c must be equal to a value in a or b
        fit = sum(child != a)
        fits[i] = fit

    return fits




def lgp_mse(pop, rmse=False, **kwargs):
    """Calculate the fitness value of all individuals in a population against the target function for the provided domain"""
    # 2D array of input variables for each test case
    cases = cartesian_prod(*kwargs['domains'])
    y_target = np.array([kwargs['target_func'](*list(case)) for case in cases])
    fits = np.empty(len(pop))
    for i, org in enumerate(pop):
        y_actual = []
        for case in cases:
            l = Linear([[0]*kwargs['num_regs'], org[0].copy()], ops=kwargs['ops'], value_lim=kwargs['value_lim'])
            for j,x in enumerate(case):
                l.mem[0][1+j] = x
            # Evaluate the organism
            l.run(kwargs['timeout'])
            y_actual.append(l.mem[0][-1])
        # Calculate MSE
        fits[i] = sum((abs(y_target - y_actual)) ** 2) / len(cases)
        # Calculate RMSE
        if rmse:
            fits[i] **= 0.5
    return fits

def lgp_rmse(pop, **kwargs):
    """Calculate the fitness value of all individuals in a population against the target function for the provided domain"""
    return lgp_mse(pop, rmse=True, **kwargs)







def repeated_lgp_mse(pop, rmse=False, **kwargs):
    """Calculate the fitness value of all individuals in a population against the target function for the provided domain"""
    # 2D array of input variables for each test case
    cases = cartesian_prod(*kwargs['domains'])
    y_target = np.array([kwargs['target_func'](*list(case)) for case in cases])
    fits = np.empty(len(pop))
    for i, org in enumerate(pop):
        y_actual = []
        l = Linear([[0]*kwargs['num_regs'], org[0].copy()], ops=kwargs['ops'], value_lim=kwargs['value_lim'])
        for case in cases:
            for j,x in enumerate(case):
                l.mem[0][1+j] = x
            # Evaluate the organism
            l.run(kwargs['timeout'])
            y_actual.append(l.mem[0][-1])
        # Calculate MSE
        fits[i] = sum((abs(y_target - y_actual)) ** 2) / len(cases)
        # Calculate RMSE
        if rmse:
            fits[i] **= 0.5
    return fits

def repeated_lgp_rmse(pop, **kwargs):
    """Calculate the fitness value of all individuals in a population against the target function for the provided domain"""
    return repeated_lgp_mse(pop, rmse=True, **kwargs)



# def dynamic_fitness(pop, gen, **kwargs):
#     if gen < kwargs['dynamic_gen']:
#         return self_crossover_overwrite_fitness(pop, **kwargs)
#     else:
#         return lgp_rmse(pop, **kwargs) + (100 * self_crossover_overwrite_fitness(pop, **kwargs))


def dynamic_fitness(pop, gen, **kwargs):
    if gen < kwargs['dynamic_gen']:
        return mutual_rep_fitness(pop, **kwargs)
    else:
        return lgp_rmse(pop, **kwargs) + (1 * mutual_rep_fitness(pop, **kwargs))




def _check_sylver_coinage(n , played_values):
    # Invalid by the definition of the game
    if n <= 1:
        return False
    # No previous moves
    elif not played_values:
        return True
    # Dynamic programming to test representability
    reachable = [False] * (n + 1)
    reachable[0] = True
    for i in range(1, n + 1):
        for a in played_values:
            if i - a >= 0 and reachable[i - a]:
                reachable[i] = True
                break
    # Move is invalid if n is in the semigroup
    return not reachable[n]


def _smlgp_sylver_coinage(org0, org1, **kwargs):
    num_turns = kwargs['num_turns']
    played_numbers = [0] * (num_turns * 2)
    # played_numbers = [0] * (num_turn
    # Initialize organism a
    a = Linear([[0] + played_numbers, org0], ops=kwargs['ops'])
    # Initialize organism b
    b = Linear([[0] + played_numbers, org1], ops=kwargs['ops'])
    for turn in range(0, num_turns, 2):
        # Update memory of played values
        a.mem[0] = [0] + [0] + played_numbers
        a.regs = a.mem[0]
        # Run until timeout
        a.run(kwargs['timeout'])
        # Extract final value played by a
        a_played = a.mem[0][1]
        # print(a_played)
        # Check if the value is valid and save it
        if _check_sylver_coinage(a_played, played_numbers):
            # played_numbers.append(a_played)
            played_numbers[turn] = a_played
        else:
            return turn-1, turn
        turn += 1
        # Update memory of played values
        b.mem[0] = [0] + [0] + played_numbers
        b.regs = b.mem[0]
        # Run until timeout
        b.run(kwargs['timeout'])
        # Extract final value played by a
        b_played = b.mem[0][1]
        # print(b_played)
        # Check if the value is valid and save it
        if _check_sylver_coinage(b_played, played_numbers):
            # played_numbers.append(b_played)
            played_numbers[turn] = b_played
        else:
            return turn, turn-1
    return num_turns+1, num_turns+1


def smlgp_compete(pop, **kwargs):
    """Randomly compete each organism against another organism"""
    shuffle_map = np.arange(len(pop))
    kwargs['rng'].shuffle(shuffle_map)
    fits = np.empty(len(pop))
    for i in range(0, len(pop), 2):
        index0 = shuffle_map[i]
        index1 = shuffle_map[i+1]
        org0 = pop[index0]
        org1 = pop[index1]
        fit0, fit1 = _smlgp_sylver_coinage(org0, org1, **kwargs)
        fits[index0] = fit0
        fits[index1] = fit1
    return fits


#
# Target Functions
#

def x2(x): return 2 * x
def multiply(x0,x1): return x0 * x1
def power(x0,x1): return x0 ** x1
def factorial(x): return math.factorial(x)
def triangular(x): return x*(x + 1)//2
def sum_squares(x): return x*(x+1)*(2*x+1)/6
def koza_1(x): return x**4 + x**3 + x**2 + x
def koza_2(x): return x**5 - 2*x**3 + x
def koza_3(x): return x**6 - 2*x**4 + x**2


#
# Crossover Functions
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

def two_point_crossover_2d(a, b, **kwargs):
    new_a = [None] * len(a)
    new_b = [None] * len(b)
    for i in range(len(a)):
         new_a[i], new_b[i] = two_point_crossover(a[i], b[i], min_len=kwargs['min_lens'][i], max_len=kwargs['max_lens'][i], **kwargs)
    return new_a, new_b

def two_point_block_crossover(a, b, **kwargs):
    block_len = Linear.LINE_LENGTH
    aa = np.array(a).reshape((-1, block_len)).tolist()
    bb = np.array(b).reshape((-1, block_len)).tolist()
    new_a, new_b = two_point_crossover(aa, bb, min_len=kwargs['min_len']//block_len, max_len=kwargs['max_len']//block_len, rng=kwargs['rng'])
    new_a = np.array(new_a).ravel().tolist()
    new_b = np.array(new_b).ravel().tolist()
    return new_a, new_b

def two_point_block_crossover_2d(a, b, **kwargs):
    new_a = [None] * len(a)
    new_b = [None] * len(b)
    for i in range(len(a)):
         new_a[i], new_b[i] = two_point_block_crossover(a[i], b[i], min_len=kwargs['min_lens'][i], max_len=kwargs['max_lens'][i], **kwargs)
    return new_a, new_b




def fixed_two_point_crossover(a, b, **kwargs):
    cut_0 = kwargs['rng'].integers(0, len(a)-1)
    cut_1 = kwargs['rng'].integers(0, len(a))
    if cut_0 > cut_1: cut_0, cut_1 = cut_1, cut_0
    # Swap the two sections
    new_a = a[:cut_0] + b[cut_0:cut_1] + a[cut_1:]
    new_b = b[:cut_0] + a[cut_0:cut_1] + b[cut_1:]
    return new_a, new_b

def fixed_two_point_crossover_2d(a, b, **kwargs):
    new_a = [None] * len(a)
    new_b = [None] * len(b)
    for i in range(len(a)):
         new_a[i], new_b[i] = fixed_two_point_crossover(a[i], b[i], **kwargs)
    return new_a, new_b

def fixed_two_point_block_crossover(a, b, **kwargs):
    block_len = 4
    aa = np.array(a).reshape((-1, block_len)).tolist()
    bb = np.array(b).reshape((-1, block_len)).tolist()
    new_a, new_b = fixed_two_point_crossover(aa, bb, rng=kwargs['rng'])
    new_a = np.array(new_a).ravel().tolist()
    new_b = np.array(new_b).ravel().tolist()
    return new_a, new_b

def fixed_two_point_block_crossover_2d(a, b, **kwargs):
    new_a = [None] * len(a)
    new_b = [None] * len(b)
    for i in range(len(a)):
         new_a[i], new_b[i] = fixed_two_point_block_crossover(a[i], b[i], min_len=kwargs['min_lens'][i], max_len=kwargs['max_lens'][i], **kwargs)
    return new_a, new_b





def setup_self_rep_crossover(self_rep_code, code_to_rep, **kwargs):
    return Linear([[0]*4, self_rep_code.copy(), code_to_rep.copy(), [-1]*len(code_to_rep)], ops=kwargs['ops'], value_lim=kwargs['value_lim'])


def self_rep_crossover(a, b, two_way=True, **kwargs):
    """Uses MEM1 in a to copy each MEM of b from MEM2 to MEM3. Then repeats with reversed roles"""
    children = []
    orderings = ((a,b),(b,a)) if two_way else ((a,b),)
    # Create and run a Linear runtime object
    for u,v in orderings:
        children.append([])
        for j, mem in enumerate(v):
            # l = Linear([[0]*4, u[0], mem, [-1]*len(mem)], ops=kwargs['ops'])
            l = setup_self_rep_crossover(u[0], mem, **kwargs)
            l.run(kwargs['timeout'])
            children[-1].append(l.mem[3])
    return children


def self_crossover_overwrite(a, b, two_way=True, **kwargs):
    """Uses a[0] to copy each MEM from a onto b. Then repeats with reversed roles"""
    children = []
    orderings = ((a,b),(b,a)) if two_way else ((a,b),)
    # Create and run a Linear runtime object
    for u,v in orderings:
        children.append([])
        for j in range(len(u)):
            l = setup_self_crossover(u[0], u[j], v[j], **kwargs)
            l.run(kwargs['timeout'])
            children[-1].append(l.mem[-1])
    return children



def self_crossover(a, b, two_way=True, **kwargs):
    """Uses a.MEM1 as MEM1 and b.MEM1 as MEM2 to create a new parent in MEM3. Then repeats with reversed roles"""
    children = []
    orderings = ((a,b),(b,a)) if two_way else ((a,b),)
    # Create and run a Linear runtime object
    for u,v in orderings:
        children.append([])
        for j in range(len(u)):
            l = setup_self_crossover(u[0], v[j], u[j], **kwargs)
            l.run(kwargs['timeout'])
            children[-1].append(l.mem[3])
    return children


def mutual_rep(a, b, two_way=True, **kwargs):
    """Uses a.MEM1 as MEM1 and b.MEM1 as MEM2 to create a new parent in MEM3. Then repeats with reversed roles"""
    children = []
    orderings = ((a,b),(b,a)) if two_way else ((a,b),)
    # Create and run a Linear runtime object
    for u,v in orderings:
        children.append([])
        for j in range(len(u)):
            l = setup_mutual_rep(u[0], v[j], **kwargs)
            l.run(kwargs['timeout'])
            children[-1].append(l.mem[-1])
    return children


def dynamic_crossover(a, b, gen, **kwargs):
    if gen < kwargs['dynamic_gen']:
        return a, b
    else:
        return mutual_rep(a, b, **kwargs)


#
# Mutation Functions
#

def point_mutation(org, **kwargs):
    """Randomly change a value in a random line"""
    # Select a random value
    index = kwargs['rng'].integers(len(org))
    # Replace the argument
    org[index] = kwargs['rng'].integers(kwargs['value_lim'])
    return org

def point_mutation_2d(org, **kwargs):
    """Randomly change a value in a random line"""
    # Select a random value
    index = kwargs['rng'].integers(len(org))
    point_mutation(org[index], **kwargs)
    return org


def contextual_point_mutation(org, **kwargs):
    """Randomly change a value in a random line"""
    # Select a random line and sub line
    index = kwargs['rng'].integers(len(org))
    sub_index = index % Linear.LINE_LENGTH
    # Replace the argument
    org[index] = _random_contextual_line(**kwargs)[sub_index]
    return org

def contextual_point_mutation_2d(org, **kwargs):
    """Randomly change a value in a random line"""
    # Select a random value
    index = kwargs['rng'].integers(len(org))
    contextual_point_mutation(org[index], **kwargs)
    return org



def dynamic_mutation(org, gen, **kwargs):
    if gen < kwargs['dynamic_gen']:
        return point_mutation_2d(org, **kwargs)
    else:
        return org

#
# Debug
#

if __name__ == '__main__':
    from main import kwargs
    kwargs['rng'] = np.random.default_rng()

    pop = [random_mems(**kwargs) for i in range(1000)]

    pop = [point_mutation_2d(i, **kwargs) for i in pop]

    pop = np.array(pop)

    print(np.max(pop))

    # a = [[
    #     'LOAD', 1, 0, 'IMMEDIATE',
    #     'STOP', 0,0,0,
    # ]]

    # kwargs['value_lim'] = 256
    #
    # a = [np.arange(16).tolist()]
    # b = [np.ones((32,),int).tolist()]
    #
    # a = random_contextual_mems(**kwargs)

    # contextual_point_mutation_2d(a, **kwargs)
    #
    # print(a)
    # # l = Linear([[0]*kwargs['num_regs'], *a], **kwargs)
    # # print(l)
    #
    # f = lgp_rmse([a], **kwargs)
    # print(f)
    # f = repeated_lgp_rmse([a], **kwargs)
    # print(f)


    # print(a)
    # print(b)
    #
    # na, nb = two_point_block_crossover_2d(a, b, **kwargs)
    #
    # print(na)
    # print(nb)
    #
    # print(len(na[0]))
    # print(len(nb[0]))

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
import random
import numpy as np


# All functions relevant to evolving in a general sense.

#
# Initialization
#

def gen_pop(pop_size, gen_individual, **kwargs):
    """Generate a random population"""
    return [gen_individual(**kwargs) for _ in range(pop_size)]

#
# Reproduction
#

def select_parent(pop, fits, k, **kwargs):
    """Select a single parent from a tournament of k"""
    # Select the random tournament
    tourn = random.choices(pop, k=k)
    # Created a zipped list of fitness and chromosomes
    parent = [(fits[i], i) for i in range(k)]
    # Sort all parents by fitness
    parent = sorted(parent)
    # Get the chromosome of the first element
    parent = tourn[parent[0][1]]
    return parent

#
# Simulation and Iteration
#

def next_pop(**kwargs):
    """Generate the next population"""

    new_pop = []

    # Truncate Selection
    if 'lambda' in kwargs:

        # Pool starts with all current parents
        pool = list(kwargs['pop'])

        # Create children for all parents and add to the pool
        for parent in kwargs['pop']:
            for i in range(kwargs['lambda']):
                child = kwargs['mutate_func'](parent, **kwargs)
                pool.append(child)

        # Evaluation
        pool_fits = kwargs['fitness_func'](pop=pool, **kwargs)

        # Sort and truncate the indices of the next generation
        pool_indices = [(pool_fits[i], i) for i in range(len(pool))]
        pool_indices = np.array(sorted(pool_indices))
        pool_indices = pool_indices[:kwargs['pop_size'], 1]
        pool_indices = list(pool_indices)

        # Reduce size
        new_pop = np.array(pool)[pool_indices]
        kwargs['fits'] = np.array(pool_fits)[pool_indices]

        # print(new_pop, kwargs['fits'])

        return new_pop, kwargs['fits']

    # Crossover
    else:

        # for parent in kwargs['pop_size']

        kwargs['fits'] = kwargs['fitness_func'](**kwargs)

        # mu = kwargs['pop_size']

        pool = []

        pool = [(kwargs['fits'][i], i) for i in range(kwargs['pop_size'])]

        pool = sorted(pool)

        # Evaluation
        # Add the fitness values to the kwargs to pass to other functions
        kwargs['fits'] = kwargs['fitness_func'](**kwargs)

        # Elitism
        new_pop = [kwargs['pop'][pool[i][1]] for i in range(kwargs['keep_parents'])]

        # Repeat until the new population is the same size as the old
        while len(new_pop) < len(kwargs['pop']):

            # Crossover
            if kwargs['crossover_func'] is not None:
                # Select two parents
                c0 = select_parent(**kwargs)
                c1 = select_parent(**kwargs)
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
    """Run a single simulation of a full set of generations"""

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

    for generation in range(kwargs['num_gens']):

        if kwargs['verbose'] > 0:
            print(f'Generation {generation} of {kwargs["num_gens"]}')

        # Next generation
        pop, fit = next_pop(pop=pop, **kwargs)

        # Save previous fitnesses
        fit_history.append(fit)

        # Save new population
        pop_history.append(pop)

        # print(pop)

    # Final fitness values
    fit_history.append(kwargs['fitness_func'](pop_history[-1], **kwargs))

    if kwargs['verbose'] > 0:
        print('Timeout reached')

    return pop_history, fit_history



def run_sims(key, values, num_runs, **kwargs):
    """Run multiple runs """

    # [test] [run] [gen] [individual]

    # All values of all chromosomes of all generations of all runs
    # This can be saved as a 4D array for easy manipulation and access
    all_pops = []
    all_fits = []

    all_pops = np.empty((len(values), num_runs, kwargs['num_gens']+1, kwargs['pop_size']), dtype=object)
    all_fits = np.empty((len(values), num_runs, kwargs['num_gens']+1, kwargs['pop_size']))

    # Tests
    for test, value in enumerate(values):
        kwargs[key] = value
        for run in range(num_runs):
            # Append all values of all chromosomes of all generations
            pops, fits = run_sim(**kwargs)
            all_pops[test, run] = pops
            all_fits[test, run] = fits

    return all_pops, all_fits



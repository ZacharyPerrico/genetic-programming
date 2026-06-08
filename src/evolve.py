"""
Core functions used in controlling evolution
All functions are independent of the subject of evolution
"""
import copy
import threading
import time
from multiprocessing import Pool, cpu_count

import numpy as np

from src.utils.save import save_kwargs, create_db, update_db


#
# Initialization
#

def init_pop(pop_size, init_individual_func, **kwargs):
    """Generate a random population"""
    return [init_individual_func(**kwargs) for _ in range(pop_size)]


#
# Selection
#

def tournament_selection(pop, fits, **kwargs):
    """Select a single parent from a tournament of k"""
    # Select the random tournament
    tourn_indices = kwargs['rng'].choice(len(pop), size=kwargs['tournament_size'], replace=False)
    tourn = [pop[i] for i in tourn_indices]
    # Created a zipped list of fitness and chromosomes
    parent = [(fits[i], i) for i in range(kwargs['tournament_size'])]
    # Sort all parents by fitness
    parent = sorted(parent)
    if not kwargs['minimize_fitness']:
        parent = parent[::-1]
    # Get the chromosome of the first element
    parent, fit = tourn[parent[0][1]], parent[0][0]
    return parent, fit


#
# Simulation and Iteration
#

def next_pop(pop, fits, gen, **kwargs):
    """Returns the next population from the given population and fitness values"""

    # Truncate Selection
    # This is not used
    if 'lambda' in kwargs:
        pass
        # Pool starts with all current parents
        # pool = list(pop) if kwargs['keep_parents'] else []
        # # Create children for all parents and add to the pool
        # for parent in pop:
        #     for i in range(kwargs['lambda']):
        #         child = kwargs['mutate_func'](parent, **kwargs)
        #         pool.append(child)
        # # Evaluation
        # pool_fits = kwargs['fitness_func'](pop=pool, **kwargs)
        # # Sort and truncate the indices of the next generation
        # pool_indices = [(pool_fits[i], i) for i in range(len(pool))]
        # pool_indices = np.array(sorted(pool_indices))
        # pool_indices = pool_indices[:kwargs['pop_size'], 1]
        # pool_indices = list(pool_indices)
        # # Reduce reps
        # new_pop = np.array(pool)[pool_indices]
        # kwargs['fits'] = np.array(pool_fits)[pool_indices]
        # if kwargs['verbose'] > 1: print(pop)
        # return new_pop, kwargs['fits']

    # Crossover
    else:

        # Elitism
        pool = [(fits[i], i) for i in range(kwargs['pop_size'])]
        pool = sorted(pool)
        pool = pool if kwargs['minimize_fitness'] else pool[::-1]
        new_pop = [pop[pool[i][1]] for i in range(kwargs['keep_parents'])]

        # Repeat until the new population is the same reps as the old
        while len(new_pop) < len(pop):

            # Selection
            org_0, fit_0 = tournament_selection(pop=pop, fits=fits, **kwargs)
            org_1, fit_1 = tournament_selection(pop=pop, fits=fits, **kwargs)
            org_0 = copy.deepcopy(org_0)
            org_1 = copy.deepcopy(org_1)

            # Crossover
            recombination_func = kwargs['rng'].choice(a=kwargs['recombination_funcs'], p=kwargs['recombination_probs'])
            if recombination_func is not None:
                org_0, org_1 = recombination_func(org_0, org_1, gen=gen, **kwargs)

            # Mutation
            mutation_func = kwargs['rng'].choice(a=kwargs['mutation_funcs'], p=kwargs['mutation_probs'])
            if mutation_func is not None:
                org_0 = mutation_func(org_0, gen=gen, **kwargs)
            mutation_func = kwargs['rng'].choice(a=kwargs['mutation_funcs'], p=kwargs['mutation_probs'])
            if mutation_func is not None:
                org_1 = mutation_func(org_1, gen=gen, **kwargs)

            new_pop.append(org_0)
            new_pop.append(org_1)

        return new_pop


def run_replicate(arg=None, **kwargs):
    """Run a single replicate with a full set of generations"""

    # A single arg may instead be passed if unpacking the dict is not possible
    if arg is not None:
        kwargs = arg

    # Initialization
    pop = init_pop(**kwargs)
    fits = kwargs['fitness_func'](pop=pop, gen=0, **kwargs)

    # Buffer holds values that have not yet been saved
    pop_buffer = [pop]
    fit_buffer = [fits]

    # Repeat for each generation
    for generation in range(1, kwargs['num_gens']):

        if kwargs['verbose']:
            print(f'Simulating Test {kwargs["test_name"]}, Run {kwargs["seed"]}, Generation {generation} of {kwargs["num_gens"]}')

        # Next generation and fitness
        pop = next_pop(pop=pop, fits=fits, gen=generation, **kwargs)
        fits = kwargs['fitness_func'](pop=pop, gen=generation, **kwargs)

        # Save results to buffer
        pop_buffer.append(pop)
        fit_buffer.append(fits)

        # Save to database and clear buffers when a checkpoint generation or the final generation is reached
        if generation % kwargs['checkpoint_interval'] == 0 or generation == kwargs['num_gens']-1:
            update_db(pop_buffer, fit_buffer, generation, **kwargs)
            pop_buffer = []
            fit_buffer = []


#
# Hyper-Parameter Generation and Control
#

def generate_reps(**kwargs):
    """
    Yields kwargs with unique seeds and rngs for each replicate
    """

    for _ in range(kwargs['num_reps']):

        # Assign seed and RNG
        kwargs['seed'] = (np.random.randint(0, 2**64, dtype='uint64'))
        kwargs['rng'] = np.random.default_rng(kwargs['seed'])

        # if 'setup_func' in kwargs:
        #     kwargs = kwargs['setup_func'](**kwargs)

        yield kwargs.copy()




def generate_tests(test_keys, test_values, **kwargs):
    """
    Convert simulation kwargs containing test_kwargs into a list of all the kwargs
    """

    kwargs['num_tests'] = len(test_keys)

    for test_num in range(kwargs['num_tests']):

        # Update with test-specific values
        rep_kwargs = copy.deepcopy(kwargs)
        for key, value in zip(test_keys, test_values[test_num]):
            rep_kwargs[key] = value

        # Add no-operation as a possible recombination
        prob_noop = 1 - sum(kwargs['recombination_probs'])
        if prob_noop > 0:
            rep_kwargs['recombination_funcs'].append(None)
            rep_kwargs['recombination_probs'].append(prob_noop)

        # Add no-operation as a possible mutation
        prob_noop = 1 - sum(kwargs['mutation_probs'])
        if prob_noop > 0:
            rep_kwargs['mutation_funcs'].append(None)
            rep_kwargs['mutation_probs'].append(prob_noop)

        yield rep_kwargs



def run_tests(**kwargs):
    """
    Simulate all runs for all tests with different hyperparameters.
    There are four levels: [test] [replicant] [generation/population] [organism/individual]
    """

    start_time = time.time()

    # Save kwargs first in case of failure
    save_kwargs(**kwargs)

    # Create the database used by all reps
    create_db(**kwargs)

    # Kwargs corresponding to each replicate
    jobs_kwargs = []
    for i in generate_tests(**kwargs):
        for j in generate_reps(**i):
            jobs_kwargs.append([j])

    # Parallelize processes
    if kwargs['parallelize']:

        if kwargs['verbose']:
            print(f'Dispatching {len(jobs_kwargs)} parallel jobs')

        # Submit jobs using multiprocessing
        with Pool(processes=min(cpu_count(), len(jobs_kwargs))) as pool:
            results = pool.starmap(run_replicate, jobs_kwargs)

        # threads = []
        # for j in jobs_args:
        #     t = threading.Thread(target=run_replicate, kwargs=j)
        #     threads.append(t)
        # for t in threads:
        #     t.start()
        # for t in threads:
        #     t.join()

    # Non parallelized
    else:
        for job_kwargs in jobs_kwargs:
            run_replicate(*job_kwargs)

    if kwargs['verbose']:
        print(f'\nTotal time elapsed {time.time() - start_time}')

# if __name__ == '__main__':
"""
Core functions used in controlling evolution
All functions are independent of the subject of evolution
"""
import copy
from multiprocessing import Pool, cpu_count

import numpy as np

from src.utils.save import save_kwargs, save_run, create_db, update_db


#
# Initialization
#

def init_pop(pop_size, init_individual_func, **kwargs):
    """Generate a random population"""
    return [init_individual_func(**kwargs) for _ in range(pop_size)]


#
# Selection
#

def tournament_selection(pop, fits, k, **kwargs):
    """Select a single parent from a tournament of k"""
    # Select the random tournament
    tourn_indices = kwargs['rng'].choice(len(pop), size=k, replace=False)
    tourn = [pop[i] for i in tourn_indices]
    # Created a zipped list of fitness and chromosomes
    parent = [(fits[i], i) for i in range(k)]
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
    """Returns the fitness values for the given population and the next population"""

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
        # kwargs['pop'] = pop

        # Evaluation
        # kwargs['fits'] = kwargs['fitness_func'](gen=gen, **kwargs)

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





def run_replicate(**kwargs):
    """Run a single simulation of a full set of generations"""

    # Initialization
    # shape = (kwargs['num_gens'] + 1, kwargs['pop_size'])
    # rep_pops = np.empty(shape, dtype=object)
    # rep_fits = np.empty(shape)
    # rep_pops[0] = init_pop(**kwargs)

    pop = init_pop(**kwargs)
    fits = kwargs['fitness_func'](pop=pop, gen=0, **kwargs)

    # Buffer holds values that have not been saved yet
    pop_buffer = [pop]
    fit_buffer = [fits]

    # Repeat for each generation
    for generation in range(1, kwargs['num_gens']):

        if kwargs['verbose'] and generation % 1 == 0:
            print(f'Simulating Test {kwargs["test_name"]}, Run {kwargs["seed"]}, Generation {generation} of {kwargs["num_gens"]}')

        # Next generation and fitness
        pop = next_pop(pop=pop, fits=fits, gen=generation, **kwargs)
        fits = kwargs['fitness_func'](pop=pop, gen=generation, **kwargs)

        # Save results
        pop_buffer.append(pop)
        fit_buffer.append(fits)

        if generation % kwargs['checkpoint_interval'] == 0 or generation == kwargs['num_gens']-1:

            update_db(pop_buffer, fit_buffer, generation, **kwargs)

            # Clear buffer
            pop_buffer = []
            fit_buffer = []

    # Final fitness values
    # rep_fits[-1] = kwargs['fitness_func'](pop, gen=generation, is_final=True, **kwargs)

    # pops, fits = run_replicate(**kwargs)
    # save_run(kwargs['test_path'], rep_pops, rep_fits, **kwargs)




def generate_reps(**kwargs):
    """
    Convert test kwargs
    """

    for _ in range(kwargs['num_reps']):

        # Assign seed and RNG
        # if kwargs['seed'] is None:
        kwargs['seed'] = np.random.randint(0, 2**64, dtype='uint64')
        kwargs['rng'] = np.random.default_rng(kwargs['seed'])

        # kwargs['rep_path'] = f'{kwargs['test_path']}{kwargs['seed']}'

        if 'setup_func' in kwargs:
            kwargs = kwargs['setup_func'](**kwargs)

        yield kwargs.copy()




def generate_tests(test_keys, test_values, **kwargs):
    """
    Convert simulation kwargs containing test_kwargs into a list of all the kwargs
    """

    kwargs['num_tests'] = len(test_keys)

    for test_num in range(kwargs['num_tests']):

        # Extract test-specific kwargs
        rep_kwargs = copy.deepcopy(kwargs)

        # Update with test-specific values
        for key, value in zip(test_keys, test_values[test_num]):
            rep_kwargs[key] = value

        # Set path
        # rep_kwargs['test_path'] = f'{kwargs['saves_path']}{kwargs['name']}/{rep_kwargs['test_name']}/'

        # print(kwargs[recombination_probs])

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
    There are four levels: [test] [run/replicant] [generation/population] [organism/individual]
    """

    # Save kwargs first in case of failure
    save_kwargs(**kwargs)

    # Set path
    kwargs['path'] = f'{kwargs['saves_path']}{kwargs['name']}/'

    create_db(**kwargs)

    # Number of tests must be inferred and is only used within this function
    # num_tests = len(test_kwargs) - 1

    # Build the job list: one job per (test, run)
    # jobs_args = [
    #     (test_num, run_num, test_kwargs, kwargs.copy())
    #     for test_num in range(num_tests)
    #     for run_num in range(num_runs)
    # ]

    jobs_args = []

    for i in generate_tests(**kwargs):
        for j in generate_reps(**i):
            jobs_args.append(j)

    # Parallelize processes
    if kwargs.get('parallelize', False):

        if kwargs.get('verbose', 0) > 0:
            print(f'Dispatching {len(jobs_args)} parallel jobs')

        # Change cpu_count() to adjust max core count use
        with Pool(processes=min(cpu_count(), len(jobs_args))) as pool:
            results = pool.starmap(run_replicate, jobs_args)

    # Non parallelized
    else:
        for job_args in jobs_args:
            run_replicate(**job_args)



# if __name__ == '__main__':
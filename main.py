from gp import *

#
# Default kwargs
#

kwargs = {
    'seed': None,
    'verbose': 1, # 0: no updates, 1: generation updates, 2: all updates

    'num_reps': 1,
    'num_gens': 100,
    'pop_size': 600, # Default: 600
    'max_tree_depth': 200, # Default: 400
    'max_subtree_depth': 4,

    'eval_method': None,

    'init_individual_func': random_tree,
    'terminals': ['x'],
    'ops': ['+','-','*','/','**'],
    'p_branch': 0.5,
    'init_tree_depth': 4,

    'fitness_func': correlation,
    'final_fitness_func': final_correlation, # Fitness function with post-processing
    'result_fitness_func': mse, # Fitness to compare results
    'domains': [[0, 1, 50]],  # The domain of the problem expressed using np.linspace

    'crossover_func': subtree_crossover,
    'k': 4, # Number of randomly chosen parents for each tournament
    'p_c': 0.9, # Probability of crossover
    'keep_parents': 4, # Elitism, must be even

    # 'mutate_func': subtree_mutation,
    'mutate_funcs': [
        [subtree_mutation, 0.3]
    ],
    'p_m': 0.5, # Probability of mutation
}

if __name__ == '__main__':

    # kwargs['name'] = 'const_32'
    # kwargs['target_func'] = const_32
    # # kwargs['fitness_func'] = correlation
    # # kwargs['result_fitness_func'] = mse_post
    # kwargs['terminals'] = ('x',)
    # kwargs['domains'] = ((-5,5,50),)
    # kwargs['num_gens'] = 10
    # kwargs['test_kwargs'] = [
    #     ['labels','p_c','p_m']] + [[f'{p_c}c {p_m}m', p_c, p_m]
    #         for p_m in np.arange(5,7,2) / 10
    #         for p_c in np.arange(5,7,2) / 10
    # ]

    # kwargs['name'] = 'cos'
    # kwargs['target_func'] = cos
    # # kwargs['fitness_func'] = correlation
    # kwargs['terminals'] = ('x','e','i',)
    # kwargs['domains'] = ((0, 2*math.pi, 31),)
    # kwargs['num_gens'] = 50
    # kwargs['pop_size'] = 400
    # kwargs['test_kwargs'] = [
    #     ['labels', 'init_individual_func', 'fitness_func'],
    #     ['random_cor', random_tree, correlation],
    #     # ['random_mse', random_tree, mse],
    # ]

    # kwargs['name'] = 'logical_or'
    # kwargs['target_func'] = logical_or
    # kwargs['terminals'] = ('x_0', 'x_1')
    # kwargs['domains'] = ((0,1,2), (0,1,2))
    # kwargs['num_gens'] = 10
    # kwargs['test_kwargs'] = [
    #     ['labels', 'ops'                      ],
    #     ['4-ops' , ['+', '-', '*', '/']       ],
    #     ['5-ops' , ['+', '-', '*', '/', '**'] ],
    # ]

    # kwargs['name'] = 'mod'
    # kwargs['target_func'] = mod2k
    # kwargs['fitness_func'] = correlation
    # kwargs['terminals'] = ('x_0', 'x_1')
    # kwargs['domains'] = ((0,15,16), (1,2,2))
    # kwargs['init_individual_func'] = init_indiv
    # # kwargs['num_gens'] = 100
    # kwargs['test_kwargs'] = [
    #     ['labels', 'ops'                      ],
    #     # ['4-ops' , ['+', '-', '*', '/']       ],
    #     ['5-ops' , ['+', '-', '*', '/', '**'] ],
    # ]

    # kwargs['name'] = 'logic'
    # kwargs['target_func'] = xor_and_xor
    # kwargs['fitness_func'] = correlation
    # kwargs['p_c'] = 0.5
    # kwargs['p_m'] = 0.5
    # kwargs['terminals'] = ('x_0', 'x_1', 'x_2', 'x_3')
    # kwargs['domains'] = ((0,1,2),(0,1,2),(0,1,2),(0,1,2))
    # kwargs['init_individual_func'] = random_tree
    # kwargs['num_gens'] = 50
    # kwargs['test_kwargs'] = [['labels','p_c','p_m']] + [[f'{p_m} {p_c}', p_c, p_m] for p_m in np.linspace(0.1,0.9,5) for p_c in np.linspace(0.1,0.9,5)]
    #
    # print(kwargs['test_kwargs'])
        # [0.3] * 2,
        # [0.5] * 2,
        # [0.7] * 2,


    # kwargs['name'] = 'cos'
    # kwargs['target_func'] = cos
    # kwargs['fitness_func'] = correlation
    # kwargs['terminals'] = ('x','e','i',)
    # kwargs['domains'] = ((0, 2*math.pi, 31),)
    # # kwargs['init_individual_func'] = init_sin
    # kwargs['num_gens'] = 1
    # kwargs['test_kwargs'] = [
    #     # ['labels', 'init_individual_func'],
    #     # ['random', random_tree],
    #     # ['sin', init_sin],
    #
    #     ['labels', 'init_individual_func', 'fitness_func'],
    #     ['random', random_tree, correlation],
    #     ['sin', init_sin, correlation],
    #     ['random_mse', random_tree, mse],
    #     ['sin_mse', init_sin, mse],
    # ]

    # kwargs['name'] = 'debug'
    # kwargs['verbose'] = 2
    # kwargs['target_func'] = f
    # kwargs['num_gens'] = 10
    # kwargs['fitness_func'] = mse
    # kwargs['legend_title'] = 'Types of Operations'
    # kwargs['test_kwargs'] = [
    #     ['labels', 'ops'                      ],
    #     ['4-ops' , ['+', '-', '*', '/']       ],
    #     # ['5-ops' , ['+', '-', '*', '/', '**'] ],
    # ]

    # kwargs |= {
    #     'name': 'debug',
    #     'verbose': 2,
    #     'target_func': f,
    #     'domains': [[0, 10, 50]],
    #     'num_gens': 50,
    #     'p_c': 0,
    #     'legend_title': 'Mutations',
    #     'test_kwargs': [
    #         ['labels', 'mutate_funcs'],
    #         # ['one', [[subtree_mutation, 0.5]]],
    #         ['split', [[split_mutation, 0], [pointer_mutation, 1.0]]],
    #         # ['two' , [[subtree_mutation, 0.5],[split_mutation, 0.5]]],
    #     ]
    # }

    # kwargs |= {
    #     'name': 'debug',
    #     'verbose': 2,
    #     'target_func': f,
    #     'domains': [[0, 10, 50]],
    #     'num_gens': 50,
    #     'p_c': 0,
    #     'legend_title': 'Mutations',
    #     'test_kwargs': [
    #         ['labels', 'mutate_funcs'],
    #         # ['one', [[subtree_mutation, 0.5]]],
    #         ['split', [[split_mutation, 0], [pointer_mutation, 1.0]]],
    #         # ['two' , [[subtree_mutation, 0.5],[split_mutation, 0.5]]],
    #     ],
    # }

    kwargs |= {
        'name': 'HA3.3.1',
        'verbose': 1,

        'target_func': sin,
        'terminals': ['x', 'e', 'i'],
        'domains': [[0, 2*np.pi, 31]],

        'fitness_func': mse,
        'final_fitness_func': mse,

        # 'target_func': f,
        # 'terminals': ['x'],
        # 'ops': ['+','-','*','/'],
        # 'domains': [[-1, 1, 50]],

        # 'init_tree_depth': 10,

        'num_gens': 10,
        'pop_size': 100,
        'keep_parents': 0,
        'k': 1,

        'legend_title': 'Mutations',
        'test_kwargs': [
            ['labels', 'p_c', 'mutate_funcs',],
            ['Full', 0, [[subtree_mutation, .7]]],
            ['Subtree Mutation', 0, [[subtree_mutation, .7]]],
            ['Crossover', .7, [[subtree_mutation, 0]]],
            ['Random', 0, [[randomize_mutation, 1]]],
            # ['Pointer Mutation', 0, [[pointer_mutation, .7]]],
        ],
    }

    # Run simulation, save, then plot
    all_pops, all_fits = run_sims(**kwargs)
    save_all(all_pops, all_fits, kwargs)
    plot_results(all_pops, all_fits, **kwargs)
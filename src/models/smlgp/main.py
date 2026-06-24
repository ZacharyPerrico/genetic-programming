from src.evolve import run_tests

# kwargs = {
#     'name': 'symb_reg_0',  # Name of folder to contain all results
#     'seed': None,
#     'verbose': True,
#     'parallelize': True,
#     'saves_path': '../../../saves/smlgp/',  # Save path relative to this file
#     ## Size ##
#     'num_runs': 3,
#     'num_gens': 600,
#     'pop_size': 128,
#     'min_lens': (64,),  # Length of each memory bank excluding regs
#     'max_lens': (128,),  # Length of each memory bank excluding regs
#     'num_regs': 4,
#     ## Initialization ##
#     'init_individual_func': random_mems,  # Function used to generate a new organism
#     'init_min_lens': (64,),  # Length of each memory bank excluding regs
#     'init_max_lens': (128,),  # Length of each memory bank excluding regs
#     'value_lim': 256,  # The largest value that can be used in the model plus one
#     'ops': ('STOP', 'LOAD', 'STORE', 'ADD', 'SUB', 'MUL', 'IFEQ'),
#     ## Evaluation ##
#     'fitness_func': lgp_rmse,
#     'timeout': 128,  # Number of evaluation iterations before forced termination
#     'target_func': factorial,
#     # 'domains': [[2,3,5],[4,5,6]],
#     'domains': [list(range(5))],
#     ## Selection ##
#     'minimize_fitness': True,
#     'keep_parents': 2,  # Elitism, must be even
#     'k': 2,  # Number of randomly chosen parents for each tournament
#     ## Repopulation ##
#     'crossover_funcs': [
#         [two_point_crossover_2d, 1.0],
#     ],
#     'mutate_funcs': [
#         [point_mutation_2d, 0.5],
#     ],
#     ## Tests ##
#     'test_kwargs': [
#         ['Crossover, Mutation', 'crossover_funcs', 'mutate_funcs'],
#         # ['0.0 0.0', [[two_point_crossover, 0.0]], [[point_mutation, 0.0]]],
#         *[
#             [f'{c}, {m}', [[two_point_crossover_2d, m]], [[point_mutation_2d, m]]]
#             for c in [.3,.5,.7]
#             for m in [.3,.5,.7]
#         ]
#     ]
# }


# kwargs = {
#     'name': 'mult_13',  # Name of folder to contain all results
#     'seed': None,
#     'verbose': True,
#     'parallelize': True,
#     'saves_path': '../../../saves/smlgp/',  # Save path relative to this file
#     ## Size ##
#     'num_runs': 3,
#     'num_gens': 2000,
#     'pop_size': 256,
#     'min_lens': (16,),  # Length of each memory bank excluding regs
#     'max_lens': (16,),  # Length of each memory bank excluding regs
#     'num_regs': 4,
#     ## Initialization ##
#     'init_individual_func': random_contextual_mems,  # Function used to generate a new organism
#     'init_min_lens': (16,),  # Length of each memory bank excluding regs
#     'init_max_lens': (16,),  # Length of each memory bank excluding regs
#     'value_lim': 64,  # One more than the largest value that can be used in the model
#     'ops': ('STOP', 'LOAD', 'STORE', 'ADD', 'SUB', 'IFEQ'),
#     # 'ops': ('STOP', 'ADD', 'SUB', 'IFEQ'),
#     ## Evaluation ##
#     'fitness_func': lgp_rmse,
#     'timeout': 32,  # Number of evaluation iterations before forced termination
#     'target_func': multiply,
#     # 'domains': [[0,1,2,3,4,5,6,7],[0,1,2,3,4,5,6,7]],
#     'domains': [[0,1,6,7],[0,1,6,7]],
#     # 'domains': [[4,5,6],[2,3,4]],
#     ## Selection ##
#     'minimize_fitness': True,
#     'keep_parents': 2,  # Elitism, must be even
#     'k': 2,  # Number of randomly chosen parents for each tournament
#     ## Repopulation ##
#     'crossover_funcs': [
#         [two_point_block_crossover_2d, 0.9],
#     ],
#     'mutate_funcs': [
#         [contextual_point_mutation_2d, 0.9],
#     ],
#     ## Tests ##
#     'test_kwargs': [
#         ['Crossover, Mutation', 'crossover_funcs', 'mutate_funcs'],
#         # ['0.0 0.0', [[two_point_crossover, 0.0]], [[point_mutation, 0.0]]],
#         *[
#             [f'{c}, {m}', [[two_point_block_crossover_2d, m]], [[contextual_point_mutation_2d, m]]]
#             for c in [.5,.7,.9]
#             for m in [.1,.3,.5]
#             # for c in [.1, .5, .9]
#             # for m in [.1, .5, .9]
#         ]
#     ]
#     # 'test_kwargs': [
#     #     ['Crossover', 'crossover_funcs'],
#     #     ['Fixed Block', [[two_point_block_crossover_2d, 0.9]]],
#     #     # ['Fixed', [[fixed_two_point_crossover_2d, 0.9]]],
#     #     # ['Regular', [[two_point_crossover_2d, 0.9]]],
#     # ]
#     # 'test_kwargs': [
#     #     ['Value Limit', 'value_lim'],
#     #     *[
#     #         [f'{t}', t]
#     #         for t in [64,128,256]
#     #     ]
#     # ]
# }



# kwargs = {
#     'name': 'self_rep_0',  # Name of folder to contain all results
#     'seed': None,
#     'verbose': True,
#     'parallelize': True,
#     'saves_path': '../../../saves/smlgp/',  # Save path relative to this file
#     ## Size ##
#     'num_runs': 20,
#     'num_gens': 1000,
#     'pop_size': 256,
#     'min_lens': (16,),  # Length of each memory bank excluding regs
#     'max_lens': (16,),  # Length of each memory bank excluding regs
#     'num_regs': 4,
#     ## Initialization ##
#     'init_individual_func': random_mems,  # Function used to generate a new organism
#     'init_min_lens': (16,),  # Length of each memory bank excluding regs
#     'init_max_lens': (16,),  # Length of each memory bank excluding regs
#     'value_lim': 16,  # One more than the largest value that can be used in the model
#     'ops': ('STOP', 'LOAD', 'STORE', 'ADD', 'SUB', 'IFEQ'),
#     ## Evaluation ##
#     'fitness_func': self_rep_fitness,
#     'timeout': 64,  # Number of evaluation iterations before forced termination
#     ## Selection ##
#     'minimize_fitness': True,
#     'keep_parents': 2,  # Elitism, must be even
#     'k': 2,  # Number of randomly chosen parents for each tournament
#     ## Repopulation ##
#     'crossover_funcs': [
#         [two_point_crossover_2d, 0.9],
#     ],
#     'mutate_funcs': [
#         [point_mutation_2d, 0.9],
#     ],
#     # Tests ##
#     'test_kwargs': [
#         ['Crossover, Mutation', 'crossover_funcs', 'mutate_funcs'],
#         # ['0.0 0.0', [[two_point_crossover, 0.0]], [[point_mutation, 0.0]]],
#         *[
#             [f'{c}, {m}', [[two_point_crossover_2d, m]], [[point_mutation_2d, m]]]
#             for c in [.7,]
#             for m in [.7,]
#         ]
#     ],
# }


# kwargs = {
#     'name': 'self_match_1',  # Name of folder to contain all results
#     'seed': None,
#     'verbose': True,
#     'parallelize': True,
#     'saves_path': '../../../saves/smlgp/',  # Save path relative to this file
#     ## Size ##
#     'num_runs': 30,
#     'num_gens': 200,
#     'pop_size': 256,
#     'min_lens': (12,),  # Length of each memory bank excluding regs
#     'max_lens': (12,),  # Length of each memory bank excluding regs
#     'num_regs': 4,
#     ## Initialization ##
#     'init_individual_func': random_mems,  # Function used to generate a new organism
#     'init_min_lens': (12,),  # Length of each memory bank excluding regs
#     'init_max_lens': (12,),  # Length of each memory bank excluding regs
#     'value_lim': 12,  # One more than the largest value that can be used in the model
#     'ops': ('STOP', 'LOAD', 'STORE', 'ADD', 'SUB', 'IFEQ'),
#     ## Evaluation ##
#     'fitness_func': self_match_fitness,
#     'timeout': 48,  # Number of evaluation iterations before forced termination
#     ## Selection ##
#     'minimize_fitness': True,
#     'keep_parents': 2,  # Elitism, must be even
#     'k': 2,  # Number of randomly chosen parents for each tournament
#     ## Repopulation ##
#     'crossover_funcs': [
#         [two_point_crossover_2d, 0.9],
#     ],
#     'mutate_funcs': [
#         [point_mutation_2d, 0.9],
#     ],
#     # Tests ##
#     'test_kwargs': [
#         ['Crossover, Mutation', 'crossover_funcs', 'mutate_funcs'],
#         # ['0.0 0.0', [[two_point_crossover, 0.0]], [[point_mutation, 0.0]]],
#         *[
#             [f'{c}, {m}', [[two_point_crossover_2d, m]], [[point_mutation_2d, m]]]
#             for c in [.9,]
#             for m in [.1,]
#         ]
#     ],
# }

# kwargs = {
#     'name': 'self_match_1',  # Name of folder to contain all results
#     'seed': None,
#     'verbose': True,
#     'parallelize': True,
#     'saves_path': '../../../saves/smlgp/',  # Save path relative to this file
#     ## Size ##
#     'num_runs': 30,
#     'num_gens': 200,
#     'pop_size': 256,
#     'min_lens': (16,),  # Length of each memory bank excluding regs
#     'max_lens': (16,),  # Length of each memory bank excluding regs
#     'num_regs': 4,
#     ## Initialization ##
#     'init_individual_func': random_mems,  # Function used to generate a new organism
#     'init_min_lens': (16,),  # Length of each memory bank excluding regs
#     'init_max_lens': (16,),  # Length of each memory bank excluding regs
#     'value_lim': 16,  # One more than the largest value that can be used in the model
#     'ops': ('STOP', 'LOAD', 'STORE', 'ADD', 'SUB', 'IFEQ'),
#     ## Evaluation ##
#     'fitness_func': self_match_fitness,
#     'timeout': 64,  # Number of evaluation iterations before forced termination
#     ## Selection ##
#     'minimize_fitness': True,
#     'keep_parents': 2,  # Elitism, must be even
#     'k': 2,  # Number of randomly chosen parents for each tournament
#     ## Repopulation ##
#     'crossover_funcs': [
#         [two_point_crossover_2d, 0.9],
#     ],
#     'mutate_funcs': [
#         [point_mutation_2d, 0.9],
#     ],
#     # Tests ##
#     'test_kwargs': [
#         ['Crossover, Mutation', 'crossover_funcs', 'mutate_funcs'],
#         # ['0.0 0.0', [[two_point_crossover, 0.0]], [[point_mutation, 0.0]]],
#         *[
#             [f'{c}, {m}', [[two_point_crossover_2d, m]], [[point_mutation_2d, m]]]
#             for c in [.9,]
#             for m in [.1,]
#         ]
#     ],
# }


# kwargs = {
#     'name': 'sum_squares_0',  # Name of folder to contain all results
#     'seed': None,
#     'verbose': True,
#     'parallelize': True,
#     'saves_path': '../../../saves/smlgp/',  # Save path relative to this file
#     ## Size ##
#     'num_runs': 15,
#     'num_gens': 100,
#     'pop_size': 512,
#     'min_lens': (16,),  # Length of each memory bank excluding regs
#     'max_lens': (64,),  # Length of each memory bank excluding regs
#     'num_regs': 6,
#     ## Initialization ##
#     'init_individual_func': random_contextual_mems,  # Function used to generate a new organism
#     'init_min_lens': (16,),  # Length of each memory bank excluding regs
#     'init_max_lens': (64,),  # Length of each memory bank excluding regs
#     'value_lim': 256,  # The largest value that can be used in the model plus one
#     'ops': ('STOP', 'LOAD', 'STORE', 'ADD', 'SUB', 'MUL', 'DIV', 'IFEQ'),
#     ## Evaluation ##
#     # 'fitness_func': repeated_lgp_rmse,
#     'timeout': 8,  # Number of evaluation iterations before forced termination
#     'target_func': sum_squares,
#     'domains': [list(range(1,8))],
#     ## Selection ##
#     'minimize_fitness': True,
#     'keep_parents': 2,  # Elitism, must be even
#     'k': 2,  # Number of randomly chosen parents for each tournament
#     ## Repopulation ##
#     'crossover_funcs': [
#         [two_point_block_crossover_2d, 0.9],
#     ],
#     'mutate_funcs': [
#         [contextual_point_mutation_2d, 0.1],
#     ],
#     ## Tests ##
#     # 'test_kwargs': [
#     #     ['Crossover', 'crossover_funcs'],
#     #     # ['Regular', [[two_point_crossover_2d, 0.9]]],
#     #     # ['Fixed', [[fixed_two_point_crossover_2d, 0.9]]],
#     #     ['Block', [[two_point_block_crossover_2d, 0.9]]],
#     #     # ['Fixed Block', [[fixed_two_point_crossover_2d, 0.9]]],
#     # ]
#     # 'test_kwargs': [
#     #     ['Ops', 'ops'],
#     #     ['Sub', ('STOP', 'LOAD', 'STORE', 'ADD', 'SUB', 'IFEQ')],
#     #     ['No Sub', ('STOP', 'LOAD', 'STORE', 'ADD', 'IFEQ')],
#     # ]
#     'test_kwargs': [
#         ['Fitness', 'fitness_func', 'timeout'],
#         ['Regular', lgp_rmse, 8*8],
#         ['Repeated', repeated_lgp_rmse, 8],
#     ]
# }

# kwargs = {
#     'name': 'lim_reg_sum_squares_1',  # Name of folder to contain all results
#     'seed': None,
#     'verbose': True,
#     'parallelize': True,
#     'saves_path': '../../../saves/smlgp/',  # Save path relative to this file
#     ## Size ##
#     'num_runs': 8,
#     'num_gens': 1000,
#     'pop_size': 512,
#     'min_lens': (16,),  # Length of each memory bank excluding regs
#     'max_lens': (32,),  # Length of each memory bank excluding regs
#     'num_regs': 2,
#     ## Initialization ##
#     'init_individual_func': random_contextual_mems,  # Function used to generate a new organism
#     'init_min_lens': (16,),  # Length of each memory bank excluding regs
#     'init_max_lens': (32,),  # Length of each memory bank excluding regs
#     'value_lim': 256,  # The largest value that can be used in the model plus one
#     'ops': ('STOP', 'LOAD', 'STORE', 'ADD', 'SUB', 'MUL', 'DIV', 'IFEQ'),
#     ## Evaluation ##
#     # 'fitness_func': repeated_lgp_rmse,
#     'timeout': 8,  # Number of evaluation iterations before forced termination
#     'target_func': triangular,
#     'domains': [list(range(1,8))],
#     ## Selection ##
#     'minimize_fitness': True,
#     'keep_parents': 2,  # Elitism, must be even
#     'k': 2,  # Number of randomly chosen parents for each tournament
#     ## Repopulation ##
#     'crossover_funcs': [
#         [two_point_block_crossover_2d, 0.9],
#     ],
#     'mutate_funcs': [
#         [contextual_point_mutation_2d, 0.1],
#     ],
#     ## Tests ##
#     'test_kwargs': [
#         ['Fitness', 'fitness_func', 'timeout', 'num_regs'],
#         ['Reset 6 Registers', lgp_rmse, 8*8, 6],
#         ['Reset 2 Registers', lgp_rmse, 8*8, 2],
#         ['Retained 6 Registers', repeated_lgp_rmse, 6],
#         ['Retained 2 Registers', repeated_lgp_rmse, 8, 2],
#     ]
# }


# kwargs = {
#     'name': 'lim_reg_sum_squares_3',  # Name of folder to contain all results
#     'seed': None,
#     'verbose': True,
#     'parallelize': True,
#     'saves_path': '../../../saves/smlgp/',  # Save path relative to this file
#     ## Size ##
#     'num_runs': 8,
#     'num_gens': 1000,
#     'pop_size': 512,
#     'min_lens': (16,),  # Length of each memory bank excluding regs
#     'max_lens': (32,),  # Length of each memory bank excluding regs
#     # 'num_regs': 2,
#     ## Initialization ##
#     'init_individual_func': random_contextual_mems,  # Function used to generate a new organism
#     'init_min_lens': (16,),  # Length of each memory bank excluding regs
#     'init_max_lens': (32,),  # Length of each memory bank excluding regs
#     'value_lim': 512,  # The largest value that can be used in the model plus one
#     'ops': ('STOP', 'LOAD', 'STORE', 'ADD', 'SUB', 'MUL', 'DIV', 'IFEQ'),
#     ## Evaluation ##
#     # 'fitness_func': repeated_lgp_rmse,
#     'timeout': 8,  # Number of evaluation iterations before forced termination
#     'target_func': sum_squares,
#     'domains': [list(range(1,6))],
#     ## Selection ##
#     'minimize_fitness': True,
#     'keep_parents': 2,  # Elitism, must be even
#     'k': 4,  # Number of randomly chosen parents for each tournament
#     ## Repopulation ##
#     'crossover_funcs': [
#         [two_point_block_crossover_2d, 0.5],
#     ],
#     'mutate_funcs': [
#         [contextual_point_mutation_2d, 0.5],
#     ],
#     ## Tests ##
#     'test_kwargs': [
#         ['Fitness', 'fitness_func', 'num_regs'],
#         ['Reset 4 Registers', lgp_rmse, 4],
#         ['Reset 2 Registers', lgp_rmse, 2],
#         ['Retained 4 Registers', repeated_lgp_rmse, 4],
#         ['Retained 2 Registers', repeated_lgp_rmse, 2],
#     ]
# }

# kwargs = {
#     'name': 'lim_reg_sum_squares_tuning_1',  # Name of folder to contain all results
#     'seed': None,
#     'verbose': True,
#     'parallelize': True,
#     'saves_path': '../../../saves/smlgp/',  # Save path relative to this file
#     ## Size ##
#     'num_runs': 3,
#     'num_gens': 1000,
#     'pop_size': 512,
#     'min_lens': (16,),  # Length of each memory bank excluding regs
#     'max_lens': (32,),  # Length of each memory bank excluding regs
#     'num_regs': 2,
#     ## Initialization ##
#     'init_individual_func': random_contextual_mems,  # Function used to generate a new organism
#     'init_min_lens': (16,),  # Length of each memory bank excluding regs
#     'init_max_lens': (32,),  # Length of each memory bank excluding regs
#     'value_lim': 512,  # The largest value that can be used in the model plus one
#     'ops': ('STOP', 'LOAD', 'STORE', 'ADD', 'SUB', 'MUL', 'DIV', 'IFEQ'),
#     ## Evaluation ##
#     'fitness_func': repeated_lgp_rmse,
#     'timeout': 8,  # Number of evaluation iterations before forced termination
#     'target_func': sum_squares,
#     'domains': [list(range(1,6))],
#     ## Selection ##
#     'minimize_fitness': True,
#     'keep_parents': 2,  # Elitism, must be even
#     'k': 4,  # Number of randomly chosen parents for each tournament
#     ## Repopulation ##
#     'crossover_funcs': [
#         [two_point_block_crossover_2d, 0.9],
#     ],
#     'mutate_funcs': [
#         [contextual_point_mutation_2d, 0.1],
#     ],
#     ## Tests ##
#     # 'test_kwargs': [
#     #     ['Fitness', 'fitness_func', 'num_regs', 'k'],
#     #     ['Reset 4 Registers', lgp_rmse, 4, 2],
#     #     ['Reset 2 Registers', lgp_rmse, 4, 4],
#     #     # ['Retained 6 Registers', repeated_lgp_rmse, 4],
#     #     # ['Retained 2 Registers', repeated_lgp_rmse, 2],
#     # ]
#     'test_kwargs': [
#         ['Crossover, Mutation', 'crossover_funcs', 'mutate_funcs'],
#         *[
#             [f'{c}, {m}', [[two_point_block_crossover_2d, m]], [[contextual_point_mutation_2d, m]]]
#             for c in [.1,.5,.9,]
#             for m in [.1,.5,.9,]
#         ]
#     ],
# }


# kwargs = {
#     'name': 'factorial_0',  # Name of folder to contain all results
#     'seed': None,
#     'verbose': True,
#     'parallelize': True,
#     'saves_path': '../../../saves/smlgp/',  # Save path relative to this file
#     ## Size ##
#     'num_runs': 15,
#     'num_gens': 100,
#     'pop_size': 512,
#     'min_lens': (16,),  # Length of each memory bank excluding regs
#     'max_lens': (64,),  # Length of each memory bank excluding regs
#     'num_regs': 6,
#     ## Initialization ##
#     'init_individual_func': random_contextual_mems,  # Function used to generate a new organism
#     'init_min_lens': (16,),  # Length of each memory bank excluding regs
#     'init_max_lens': (64,),  # Length of each memory bank excluding regs
#     'value_lim': 256,  # The largest value that can be used in the model plus one
#     'ops': ('STOP', 'LOAD', 'STORE', 'ADD', 'SUB', 'MUL', 'DIV', 'IFEQ'),
#     ## Evaluation ##
#     # 'fitness_func': repeated_lgp_rmse,
#     'timeout': 8,  # Number of evaluation iterations before forced termination
#     'target_func': factorial,
#     'domains': [list(range(1,8))],
#     ## Selection ##
#     'minimize_fitness': True,
#     'keep_parents': 2,  # Elitism, must be even
#     'k': 2,  # Number of randomly chosen parents for each tournament
#     ## Repopulation ##
#     'crossover_funcs': [
#         [two_point_block_crossover_2d, 0.9],
#     ],
#     'mutate_funcs': [
#         [contextual_point_mutation_2d, 0.1],
#     ],
#     ## Tests ##
#     'test_kwargs': [
#         ['Fitness', 'fitness_func', 'timeout'],
#         ['Regular', lgp_rmse, 8*8],
#         ['Repeated', repeated_lgp_rmse, 8],
#     ]
# }

# kwargs = {
#     'name': 'lim_reg_sum_squares_weighted',  # Name of folder to contain all results
#     'seed': None,
#     'verbose': True,
#     'parallelize': True,
#     'saves_path': '../../../saves/smlgp/',  # Save path relative to this file
#     ## Size ##
#     'num_runs': 8,
#     'num_gens': 4000,
#     'pop_size': 512,
#     'min_lens': (16,),  # Length of each memory bank excluding regs
#     'max_lens': (64,),  # Length of each memory bank excluding regs
#     # 'num_regs': 2,
#     ## Initialization ##
#     'init_individual_func': random_contextual_mems,  # Function used to generate a new organism
#     'init_min_lens': (16,),  # Length of each memory bank excluding regs
#     'init_max_lens': (32,),  # Length of each memory bank excluding regs
#     'value_lim': 512,  # The largest value that can be used in the model plus one
#     'ops': ('STOP', 'LOAD', 'STORE', 'ADD', 'SUB', 'MUL', 'DIV', 'IFEQ'),
#     ## Evaluation ##
#     # 'fitness_func': repeated_lgp_rmse,
#     'timeout': 16,  # Number of evaluation iterations before forced termination
#     'target_func': sum_squares,
#     'domains': [list(range(1,6))],
#     ## Selection ##
#     'minimize_fitness': True,
#     'keep_parents': 2,  # Elitism, must be even
#     'k': 2,  # Number of randomly chosen parents for each tournament
#     ## Repopulation ##
#     'crossover_funcs': [
#         [two_point_block_crossover_2d, 0.5],
#     ],
#     'mutate_funcs': [
#         [contextual_point_mutation_2d, 0.5],
#     ],
#     ## Tests ##
#     'test_kwargs': [
#         ['Fitness', 'fitness_func', 'num_regs'],
#         ['Reset 4 Registers', lgp_rmse, 4],
#         ['Reset 2 Registers', lgp_rmse, 2],
#         ['Retained 4 Registers', repeated_lgp_rmse, 4],
#         ['Retained 2 Registers', repeated_lgp_rmse, 2],
#     ]
# }

# kwargs = {
#     'name': 'weighted_lim_reg_sum_squares_4',  # Name of folder to contain all results
#     'seed': None,
#     'verbose': True,
#     'parallelize': True,
#     'saves_path': '../../../saves/smlgp/',  # Save path relative to this file
#     ## Size ##
#     'num_runs': 8,
#     'num_gens': 1000,
#     'pop_size': 512,
#     'min_lens': (16,),  # Length of each memory bank excluding regs
#     'max_lens': (64,),  # Length of each memory bank excluding regs
#     # 'num_regs': 2,
#     ## Initialization ##
#     'init_individual_func': random_weighted_contextual_mems,  # Function used to generate a new organism
#     'init_min_lens': (16,),  # Length of each memory bank excluding regs
#     'init_max_lens': (32,),  # Length of each memory bank excluding regs
#     'value_lim': 512,  # The largest value that can be used in the model plus one
#     # 'ops': ('STOP', 'LOAD', 'STORE', 'ADD', 'SUB', 'MUL', 'DIV', 'IFEQ'),
#     'ops': ('STOP', 'LOAD', 'ADD', 'SUB', 'MUL', 'DIV'),
#     'addr_weights': [1, 2, 0, 0, 0],
#     'pc_weight': 0,
#     ## Evaluation ##
#     # 'fitness_func': repeated_lgp_rmse,
#     'timeout': 16,  # Number of evaluation iterations before forced termination
#     'target_func': sum_squares,
#     'domains': [list(range(1,6))],
#     ## Selection ##
#     'minimize_fitness': True,
#     'keep_parents': 2,  # Elitism, must be even
#     'k': 2,  # Number of randomly chosen parents for each tournament
#     ## Repopulation ##
#     'crossover_funcs': [
#         [two_point_block_crossover_2d, 0.9],
#     ],
#     'mutate_funcs': [
#         [weighted_contextual_point_mutation_2d, 0.9],
#     ],
#     ## Tests ##
#     'test_kwargs': [
#         ['Fitness', 'fitness_func', 'num_regs'],
#         ['Reset 4 Registers', lgp_rmse, 4],
#         ['Reset 2 Registers', lgp_rmse, 2],
#         ['Retained 4 Registers', repeated_lgp_rmse, 4],
#         ['Retained 2 Registers', repeated_lgp_rmse, 2],
#     ]
# }



kwargs = {
    'name': 'test/test',  # Name of folder to contain all results
    'verbose': True,
    'parallelize': True,
    'saves_path': '../../../saves/',  # Save path relative to this file
    'checkpoint_interval': 1,
    'update_timeout': 60,
    ## Size ##
    'num_reps': 16,
    'num_gens': 100,
    'pop_size': 100,
    'min_lens': [16],  # Length of each memory bank excluding regs
    'max_lens': [64],  # Length of each memory bank excluding regs
    # 'num_regs': 2,
    ## Initialization ##
    'init_individual_func': random_contextual_mems,  # Function used to generate a new organism
    'init_min_lens': [16],  # Length of each memory bank excluding regs
    'init_max_lens': [32],  # Length of each memory bank excluding regs
    'value_lim': 512,  # The largest value that can be used in the model plus one
    'ops': ('STOP', 'LOAD', 'STORE', 'ADD', 'SUB', 'MUL', 'DIV', 'IFEQ'),
    ## Evaluation ##
    'fitness_func': lgp_rmse,
    'timeout': 16,  # Number of evaluation iterations before forced termination
    'target_func': sum_squares,
    'domains': [list(range(1,6))],
    ## Selection ##
    'minimize_fitness': True,
    'keep_parents': 2,  # Elitism, must be even
    'tournament_size': 2,  # Number of randomly chosen parents for each tournament
    ## Repopulation ##
    'recombination_funcs': [two_point_block_crossover_2d],
    'recombination_probs': [0.9],
    'mutation_funcs': [contextual_point_mutation_2d],
    'mutation_probs': [0.9],
    ## Tests ##
    'test_label': 'Number Registers',  # Label to use when comparing all tests
    'test_keys': ['test_name', 'num_regs'],  # Keys of each parameter to be changed for each test
    'test_values': [  # Tuple of tuples representing all values to change for each test
        ['2 Registers', 2],
        ['4 Registers', 4],
    ],
}

if __name__ == '__main__':

    # kwargs =

    # for i in generate_tests(**kwargs):
    #     for j in generate_reps(**i):
    #         print(j)

    # for k in kwargs:
    #     print(k)

    # print(list(i))

    run_tests(**kwargs)

    # fits = load_fits(**kwargs)
    # plot_results(fits, **kwargs)


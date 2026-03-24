from src.evolve import simulate_tests
from src.models.smlgp import *
from src.models.smlgp.plot import plot_results
from src.utils.save import load_runs, load_fits

# kwargs = {
#     'name': 'self_rep_mult_0',  # Name of folder to contain all results
#     'seed': None,
#     'verbose': True,
#     'parallelize': True,
#     'saves_path': '../saves/',  # Save path relative to this file
#     ## Size ##
#     'num_runs': 8,
#     'num_gens': 1000,
#     'pop_size': 1000,
#     'min_len': 4,
#     'max_len': 4,
#     ## Initialization ##
#     'init_individual_func': random_self_rep_code,  # Function used to generate a new organism
#     'init_min_len': 4,
#     'init_max_len': 4,
#     'value_lim': 16,
#     'ops': list(range(len(Linear.DEFAULT_OPS))),
#     'addr_modes': list(range(len(Linear.DEFAULT_ADDR_MODES))),
#     ## Evaluation ##
#     'fitness_func': lgp_self_rep_rmse,
#     'target_func': multiply,  # The function that the organism is attempting to replicate
#     'domains': [list(range(0, 4)), list(range(0, 4))],  # Cases are generated from the Cartesian product
#     'timeout': 64,  # Number of evaluation iterations before forced termination
#     ## Selection ##
#     'minimize_fitness': True,
#     'keep_parents': 2,  # Elitism, must be even
#     'k': 2,  # Number of randomly chosen parents for each tournament
#     ## Repopulation ##
#     'crossover_funcs': [
#         # [two_point_crossover, 0.9],
#         [self_crossover, 1.0],
#     ],
#     'mutate_funcs': [
#         [point_mutation, 0.0],
#     ],
#     ## Tests ##
#     'test_kwargs': [
#         ['Max Length', 'init_max_len', 'max_len'],
#         *[[str(i),i,i] for i in [4,8,16]],
#     ],
#     # 'test_kwargs': [
#     #     ['Crossover, Mutation', 'crossover_funcs', 'mutate_funcs'],
#     #     *[
#     #         [f'{pc} {pm}', [[two_point_crossover, pc]], [[point_mutation, pm]]]
#     #         for pc in [.3,.5,.7,.9]
#     #         for pm in [.3,.5,.7,.9]
#     #         # for pc in [.9]
#     #         # for pt in [.9]
#     #     ]
#     # ],
#     # 'test_kwargs': [
#     #     ['Crossover, Mutation', 'crossover_funcs', 'mutate_funcs'],
#     #     [f'0.5', [[two_point_crossover, .5]], [[point_mutation, .5]]],
#     #     [f'0.9', [[two_point_crossover, .9]], [[point_mutation, .9]]],
#     # ],
# }


# kwargs = {
#     'name': 'new_test_0',  # Name of folder to contain all results
#     'seed': None,
#     'verbose': True,
#     'parallelize': True,
#     'saves_path': '../../../saves/smlgp/',  # Save path relative to this file
#     ## Size ##
#     'num_runs': 1,
#     'num_gens': 100,
#     'pop_size': 100,
#     'min_lens': [64, 64],  # The length of each memory
#     'max_lens': [64, 64],  # The length of each memory
#     # 'min_len': 4,  # Used by predefined repopulation methods
#     # 'max_len': 4,  # Used by predefined repopulation methods
#     ## Initialization ##
#     'init_individual_func': random_code,  # Function used to generate a new organism
#     # 'init_min_len': 4,  # Used by predefined initialization methods
#     # 'init_max_len': 4,  # Used by predefined initialization methods
#     'init_min_lens': [64, 64],  # The length of each memory
#     'init_max_lens': [64, 64],  # The length of each memory
#     'value_lim': 16,
#     'ops': list(range(len(Linear.DEFAULT_OPS))),
#     'addr_modes': list(range(len(Linear.DEFAULT_ADDR_MODES))),
#     ## Evaluation ##
#     'fitness_func': smlgp_compete,
#     'target_func': multiply,  # The function that the organism is attempting to replicate across the domains
#     'domains': [list(range(0, 4)), list(range(0, 4))],  # Cases are generated from the Cartesian product
#     'timeout': 64,  # Number of evaluation iterations before forced termination
#     ## Selection ##
#     'minimize_fitness': True,
#     'keep_parents': 2,  # Elitism, must be even
#     'k': 2,  # Number of randomly chosen parents for each tournament
#     ## Repopulation ##
#     'crossover_funcs': [
#         [two_point_crossover, 0.9],
#         # [self_crossover, 1.0],
#     ],
#     'mutate_funcs': [
#         [point_mutation, 0.0],
#     ],
#     ## Tests ##
#     # 'test_kwargs': [
#     #     ['Crossover Rate', 'init_individual_func', 'fitness_func', 'crossover_funcs', ],
#     #     # ['1.0', random_non_self_rep_code, lgp_self_rep_rmse, [[self_crossover, 1.0]]],
#     #     # ['0.7', random_random_code, lgp_self_rep_rmse, [[self_crossover, 0.7]]],
#     #     # ['1.0', random_random_code, lgp_self_rep_rmse, [[self_crossover, 1.0]]],
#     #     # ['0.5', random_self_rep_code, lgp_self_rep_rmse, [[self_crossover, 0.5]]],
#     #     # ['1.0', random_self_rep_code, lgp_self_rep_rmse, [[self_crossover, 1.0]]],
#     #     ['p.5', random_code, lgp_rmse, [[two_point_crossover, 1.0]], ('STOP','LOAD','STORE','ADD','SUB','IFEQ','RAND',)],
#     #     # ['p.5', random_code, lgp_rmse, [[two_point_crossover, 1.0]]],
#     # ],
#     # 'test_kwargs': [
#     #     ['Ops', 'ops'],
#     #     ['Normal', ('STOP', 'LOAD', 'STORE', 'ADD', 'SUB', 'IFEQ',)],
#     #     ['DEL', ('STOP', 'LOAD', 'STORE', 'ADD', 'SUB', 'IFEQ', 'DEL',)],
#     # ],
#     # 'test_kwargs': [
#     #     ['Crossover, Mutation', 'crossover_funcs', 'mutate_funcs'],
#     #     *[
#     #         [f'{pc} {pm}', [[two_point_crossover, pc]], [[point_mutation, pm]]]
#     #         for pc in [.3,.5,.7,.9]
#     #         for pm in [.3,.5,.7,.9]
#     #         # for pc in [.9]
#     #         # for pt in [.9]
#     #     ]
#     # ],
#     # 'test_kwargs': [
#     #     ['Crossover, Mutation', 'crossover_funcs', 'mutate_funcs'],
#     #     [f'0.5', [[two_point_crossover, .5]], [[point_mutation, .5]]],
#     #     [f'0.9', [[two_point_crossover, .9]], [[point_mutation, .9]]],
#     # ],
#     'test_kwargs': [
#         ['Ops', 'ops'],
#         ['Normal', ('STOP', 'LOAD', 'STORE', 'ADD', 'SUB', 'IFEQ',)],
#         # ['DEL', ('STOP', 'LOAD', 'STORE', 'ADD', 'SUB', 'IFEQ', 'DEL',)],
#     ],
# }


# kwargs = {
#     'name': 'sylver_coinage',  # Name of folder to contain all results
#     'seed': None,
#     'verbose': True,
#     'parallelize': True,
#     'saves_path': '../../../saves/smlgp/',  # Save path relative to this file
#     ## Size ##
#     'num_runs': 2,
#     'num_gens': 2000,
#     'pop_size': 64,
#     # 'min_len': 64,  # Used by predefined repopulation methods
#     # 'max_len': 64,  # Used by predefined repopulation methods
#     'min_lens': [8,16],  # The length of each memory
#     'max_lens': [8,16],  # The length of each memory
#     'num_turns': 16,
#     ## Initialization ##
#     'init_individual_func': random_mem,  # Function used to generate a new organism
#     # 'init_min_len': 64,  # Used by predefined initialization methods
#     # 'init_max_len': 64,  # Used by predefined initialization methods
#     'init_min_lens': [8,16],  # The length of each memory
#     'init_max_lens': [8,16],  # The length of each memory
#     'value_lim': 16,
#     'ops': ('STOP', 'LOAD', 'STORE', 'ADD', 'SUB', 'IFEQ',),
#     'addr_modes': list(range(len(Linear.DEFAULT_ADDR_MODES))),
#     ## Evaluation ##
#     'fitness_func': smlgp_compete,
#     'timeout': 64,  # Number of evaluation iterations before forced termination
#     ## Selection ##
#     'minimize_fitness': False,
#     'keep_parents': 2,  # Elitism, must be even
#     'k': 2,  # Number of randomly chosen parents for each tournament
#     ## Repopulation ##
#     'crossover_funcs': [
#         [two_point_crossover, 0.1],
#     ],
#     'mutate_funcs': [
#         [point_mutation, 0.1],
#     ],
#     ## Tests ##
#     # 'test_kwargs': [
#     #     ['Ops', 'mutate_funcs'],
#     #     ['0.0', [[point_mutation, 0.0]]],
#     #     # ['0.3', [[point_mutation, 0.3]]],
#     #     # ['0.5', [[point_mutation, 0.5]]],
#     #     ['0.7', [[point_mutation, 0.7]]],
#     #     # ['0.9', [[point_mutation, 0.9]]],
#     # ]
#     # 'test_kwargs': [
#     #     ['Crossover, Mutation', 'crossover_funcs', 'mutate_funcs'],
#     #     # ['0.0 0.0', [[two_point_crossover, 0.0]], [[point_mutation, 0.0]]],
#     #     ['0.0, 0.7', [[two_point_crossover, 0.0]], [[point_mutation, 0.7]]],
#     #     ['0.7, 0.0', [[two_point_crossover, 0.7]], [[point_mutation, 0.0]]],
#     #     ['0.7, 0.7', [[two_point_crossover, 0.7]], [[point_mutation, 0.7]]],
#     # ]
#     'test_kwargs': [
#         ['Crossover, Mutation', 'crossover_funcs', 'mutate_funcs'],
#         # ['0.0 0.0', [[two_point_crossover, 0.0]], [[point_mutation, 0.0]]],
#         *[[f'{c}, {m}', [[two_point_crossover, c]], [[point_mutation, m]]]
#           for c in [0.05, 0.1, 0.15]
#           for m in [0.05, 0.1, 0.15]
#           ]
#     ]
#     # 'test_kwargs': [
#     #     ['Tournament Size', 'k'],
#     #     *[[f'{k}', k] for k in [2, 4]]
#     # ]
#     # 'test_kwargs': [
#     #     ['Operations', 'ops'],
#     #     ['Normal', ('STOP', 'LOAD', 'STORE', 'ADD', 'SUB', 'IFEQ',)],
#     #     ['Multiply', ('STOP', 'LOAD', 'STORE', 'ADD', 'SUB', 'IFEQ', 'MUL',)],
#     #     # ['Multiply', ('STOP', 'LOAD', 'STORE', 'ADD', 'SUB', 'IFEQ', 'MUL', 'DIV')],
#     # ]
#     # 'test_kwargs': [
#     #     ['Length', 'min_length', 'max_length', 'init_min_length', 'init_max_length'],
#     #     *[[f'{l}', l,l,l,l] for l in [32, 64, 128]]
#     # ]
# }




# kwargs = {
#     'name': 'project_test_1',  # Name of folder to contain all results
#     'seed': None,
#     'verbose': True,
#     'parallelize': True,
#     'saves_path': '../../../saves/smlgp/',  # Save path relative to this file
#     ## Size ##
#     'num_runs': 6,
#     'num_gens': 500,
#     'pop_size': 128,
#     'min_lens': [16,16],  # Length of each memory bank
#     'max_lens': [16,16],  # Length of each memory bank
#     'dynamic_gen': 400,
#     ## Initialization ##
#     'init_individual_func': random_mems,  # Function used to generate a new organism
#     'init_min_lens': [16,16],  # Length of each memory bank
#     'init_max_lens': [16,16],  # Length of each memory bank
#     'value_lim': 32,
#     'ops': ('STOP', 'LOAD', 'STORE', 'ADD', 'SUB', 'IFEQ',),
#     ## Evaluation ##
#     'fitness_func': swap_fit,
#     'timeout': 64,  # Number of evaluation iterations before forced termination
#     'target_func': multiply,
#     'domains': [[2,3,5],[2,3]],
#     ## Selection ##
#     'minimize_fitness': True,
#     'keep_parents': 2,  # Elitism, must be even
#     'k': 2,  # Number of randomly chosen parents for each tournament
#     ## Repopulation ##
#     'crossover_funcs': [
#         [swap_crossover, 1.0],
#     ],
#     'mutate_funcs': [
#         [point_mutation_2d, 0.5],
#     ],
#     ## Tests ##
#     'test_kwargs': [
#         ['Crossover, Mutation', 'mutate_funcs'],
#         # ['0.0 0.0', [[two_point_crossover, 0.0]], [[point_mutation, 0.0]]],
#         *[[f'{m}', [[point_mutation_2d, m]]]
#           for m in [.25,.3,.35]
#           ]
#     ]
# }

# kwargs = {
#     'name': 'project_test_7',  # Name of folder to contain all results
#     'seed': None,
#     'verbose': True,
#     'parallelize': True,
#     'saves_path': '../../../saves/smlgp/',  # Save path relative to this file
#     ## Size ##
#     'num_runs': 2,
#     'num_gens': 2100,
#     'pop_size': 128,
#     'min_lens': (32,16),  # Length of each memory bank
#     'max_lens': (32,16),  # Length of each memory bank
#     'dynamic_gen': 2000,
#     ## Initialization ##
#     'init_individual_func': random_mems,  # Function used to generate a new organism
#     'init_min_lens': (32,16),  # Length of each memory bank
#     'init_max_lens': (32,16),  # Length of each memory bank
#     'value_lim': 32,
#     'ops': ('STOP', 'LOAD', 'STORE', 'ADD', 'SUB', 'IFEQ', 'RAND'),
#     ## Evaluation ##
#     'fitness_func': dynamic_fitness,
#     'timeout': 128,  # Number of evaluation iterations before forced termination
#     'target_func': multiply,
#     'domains': [[2,3,5],[4,5,6]],
#     # 'domains': [[0,1,2,3,4]],
#     ## Selection ##
#     'minimize_fitness': True,
#     'keep_parents': 2,  # Elitism, must be even
#     'k': 2,  # Number of randomly chosen parents for each tournament
#     ## Repopulation ##
#     'crossover_funcs': [
#         [dynamic_crossover, 1.0],
#     ],
#     'mutate_funcs': [
#         [point_mutation_2d, 0.5],
#     ],
#     ## Tests ##
#     'test_kwargs': [
#         ['Crossover, Mutation', 'mutate_funcs', 'k', 'keep_parents'],
#         # ['0.0 0.0', [[two_point_crossover, 0.0]], [[point_mutation, 0.0]]],
#         *[
#             [f'{k}, {e}', [[dynamic_mutation, m]], k, e]
#             for m in [.4]
#             for k in [2, 4, 8, 16]
#             for e in [2, 4, 8, 16]
#             # for k in [4]
#             # for e in [2]
#         ]
#     ]
# }


# kwargs = {
#     'name': 'project_test_9',  # Name of folder to contain all results
#     'seed': None,
#     'verbose': True,
#     'parallelize': True,
#     'saves_path': '../../../saves/smlgp/',  # Save path relative to this file
#     ## Size ##
#     'num_runs': 2,
#     'num_gens': 2100,
#     'pop_size': 128,
#     'min_lens': (16,16),  # Length of each memory bank
#     'max_lens': (16,16),  # Length of each memory bank
#     'dynamic_gen': 2000,
#     ## Initialization ##
#     'init_individual_func': random_mems,  # Function used to generate a new organism
#     'init_min_lens': (16,16),  # Length of each memory bank
#     'init_max_lens': (16,16),  # Length of each memory bank
#     'value_lim': 32,
#     'ops': ('STOP', 'LOAD', 'STORE', 'ADD', 'SUB', 'IFEQ'),
#     ## Evaluation ##
#     'fitness_func': dynamic_fitness,
#     'timeout': 64,  # Number of evaluation iterations before forced termination
#     'target_func': multiply,
#     'domains': [[2,3,5],[4,5,6]],
#     # 'domains': [[0,1,2,3,4]],
#     ## Selection ##
#     'minimize_fitness': True,
#     'keep_parents': 2,  # Elitism, must be even
#     'k': 2,  # Number of randomly chosen parents for each tournament
#     ## Repopulation ##
#     'crossover_funcs': [
#         [dynamic_crossover, 1.0],
#     ],
#     'mutate_funcs': [
#         [point_mutation_2d, 0.5],
#     ],
#     ## Tests ##
#     'test_kwargs': [
#         ['Crossover, Mutation', 'mutate_funcs', 'k', 'keep_parents'],
#         # ['0.0 0.0', [[two_point_crossover, 0.0]], [[point_mutation, 0.0]]],
#         *[
#             [f'{k}, {e}', [[point_mutation_2d, m]], k, e]
#             for m in [.4]
#             for k in [2,4,8]
#             for e in [2,4,8]
#             # for k in [4]
#             # for e in [2]
#         ]
#     ]
# }

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

kwargs = {
    'name': 'self_match_1',  # Name of folder to contain all results
    'seed': None,
    'verbose': True,
    'parallelize': True,
    'saves_path': '../../../saves/smlgp/',  # Save path relative to this file
    ## Size ##
    'num_runs': 30,
    'num_gens': 200,
    'pop_size': 256,
    'min_lens': (16,),  # Length of each memory bank excluding regs
    'max_lens': (16,),  # Length of each memory bank excluding regs
    'num_regs': 4,
    ## Initialization ##
    'init_individual_func': random_mems,  # Function used to generate a new organism
    'init_min_lens': (16,),  # Length of each memory bank excluding regs
    'init_max_lens': (16,),  # Length of each memory bank excluding regs
    'value_lim': 16,  # One more than the largest value that can be used in the model
    'ops': ('STOP', 'LOAD', 'STORE', 'ADD', 'SUB', 'IFEQ'),
    ## Evaluation ##
    'fitness_func': self_match_fitness,
    'timeout': 64,  # Number of evaluation iterations before forced termination
    ## Selection ##
    'minimize_fitness': True,
    'keep_parents': 2,  # Elitism, must be even
    'k': 2,  # Number of randomly chosen parents for each tournament
    ## Repopulation ##
    'crossover_funcs': [
        [two_point_crossover_2d, 0.9],
    ],
    'mutate_funcs': [
        [point_mutation_2d, 0.9],
    ],
    # Tests ##
    'test_kwargs': [
        ['Crossover, Mutation', 'crossover_funcs', 'mutate_funcs'],
        # ['0.0 0.0', [[two_point_crossover, 0.0]], [[point_mutation, 0.0]]],
        *[
            [f'{c}, {m}', [[two_point_crossover_2d, m]], [[point_mutation_2d, m]]]
            for c in [.9,]
            for m in [.1,]
        ]
    ],
}


# kwargs = {
#     # 'name': 'triangular_4',  # Name of folder to contain all results
#     'name': 'sum_squares_0',  # Name of folder to contain all results
#     'seed': None,
#     'verbose': True,
#     'parallelize': True,
#     'saves_path': '../../../saves/smlgp/',  # Save path relative to this file
#     ## Size ##
#     'num_runs': 15,
#     'num_gens': 40,
#     'pop_size': 512,
#     'min_lens': (16,),  # Length of each memory bank excluding regs
#     'max_lens': (32,),  # Length of each memory bank excluding regs
#     'num_regs': 6,
#     ## Initialization ##
#     'init_individual_func': random_contextual_mems,  # Function used to generate a new organism
#     'init_min_lens': (16,),  # Length of each memory bank excluding regs
#     'init_max_lens': (32,),  # Length of each memory bank excluding regs
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
#     # 'test_kwargs': [
#     #     ['Fitness', 'fitness_func', 'timeout'],
#     #     ['Regular', lgp_rmse, 8*8],
#     #     ['Repeated', repeated_lgp_rmse, 8],
#     # ]
#     'test_kwargs': [
#         ['Generic Test', 'fitness_func', 'timeout'],
#         ['Test 0', lgp_rmse, 8],
#         ['Test 1', repeated_lgp_rmse, 8],
#     ]
# }



if __name__ == '__main__':
    simulate_tests(**kwargs)
    fits = load_fits(**kwargs)
    plot_results(fits, **kwargs)



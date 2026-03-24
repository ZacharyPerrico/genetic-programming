"""All functions relevant to plotting."""

import os

from src.models import *
from src.utils.save import load_kwargs, load_pop, load_fits, load_seed, load_pops
from src.utils.utils import cartesian_prod


#
# Data Based Plotting
#

def plot_fitness(all_fits, ax=None, save='Fitness', show=True, **kwargs):
    """Plot the average of the runs' minimum fitness for each test"""
    if ax is None:
        fig, ax = plt.subplots()
    x = np.array(range(all_fits.shape[2]))
    for test in range(all_fits.shape[0]):
        if kwargs['minimize_fitness']:
            y = np.mean(np.min(all_fits[test], axis=2), axis=0)
            ax.set_ylabel('Average Min Fitness Value')
        else:
            y = np.mean(np.max(all_fits[test], axis=2), axis=0)
            ax.set_ylabel('Average Max Fitness Value')

        plt.plot(x, y, label=kwargs['test_kwargs'][test + 1][0])

        # Error bands
        y_std = np.std(np.min(all_fits[test], axis=2), axis=0)
        ax.fill_between(x, y - y_std, y + y_std, alpha=0.2)

        # Plot every run
        # y = np.min(all_fits[test], axis=2)
        # for run in range(all_fits.shape[1]):
        #     plt.plot(x, y[run], label=kwargs['test_kwargs'][test + 1][0])

        # Min bands
        y = np.min(np.min(all_fits[test], axis=2), axis=0)
        plt.plot(x, y,  ':', label=kwargs['test_kwargs'][test + 1][0]+' (min-min)')

        # y = np.max(np.max(all_fits[test], axis=2), axis=0)
        # plt.plot(x, y, label=kwargs['test_kwargs'][test + 1][0]+' (max-max)')

        # y = np.max(np.min(all_fits[test], axis=2), axis=0)
        # plt.plot(x, y, label=kwargs['test_kwargs'][test + 1][0]+' (max-min)')

        # for run in range(all_fits[test].shape[0]):
        #     y = np.mean(all_fits[test,run], axis=1)
        #     plt.plot(x, y, label=kwargs['test_kwargs'][test + 1][0] + f' ({run})')

        # Scatter plot all points
        # xx = x.reshape((1,len(x),1)).repeat(all_fits.shape[1], axis=0).repeat(all_fits.shape[3], axis=2).ravel()
        # yy = all_fits[test].ravel()
        # plt.scatter(xx, yy, 0.1)

    # ax.set_yscale('log')
    plt.legend(title=kwargs['test_kwargs'][0][0])
    ax.set_xlabel('Generation')
    if save:
        plt.savefig(f'{kwargs["saves_path"]}{kwargs["name"]}/plots/{save}.pdf')
    if show:
        plt.show()
    plt.close()



def plot_mean_fitness(all_fits, ax=None, save=True, show=True, **kwargs):
    """Plot the average of the runs' minimum fitness for each test"""
    if ax is None:
        fig, ax = plt.subplots()

    x = np.array(range(all_fits.shape[2]))
    for test in range(all_fits.shape[0]):
        y = np.mean(np.mean(all_fits[test], axis=2), axis=0)
        ax.set_ylabel('Average Fitness Value')

        plt.plot(x, y, label=kwargs['test_kwargs'][test + 1][0])
        y_std = np.std(np.mean(all_fits[test], axis=2), axis=0)
        ax.fill_between(x, y - y_std, y + y_std, alpha=0.2)

        # Scatter plot all points
        # xx = x.reshape((1,len(x),1)).repeat(all_fits.shape[1], axis=0).repeat(all_fits.shape[3], axis=2).ravel()
        # yy = all_fits[test].ravel()
        # plt.scatter(xx, yy, 0.1)

    # ax.set_yscale('log')
    ax.set_xlabel('Generation')
    plt.legend(title=kwargs['test_kwargs'][0][0])
    if save:
        plt.savefig(f'{kwargs["saves_path"]}{kwargs["name"]}/plots/Fitness.png')
    if show:
        plt.show()
    plt.close()


def plot_means(values, ylabel, ax=None, save=True, show=True, **kwargs):
    """Plot the means of some values"""
    if ax is None:
        fig, ax = plt.subplots()
    for test in range(values.shape[0]):
        label = kwargs['test_kwargs'][test + 1][0]
        ys = np.mean(values[test], axis=(0,2))
        xs = np.array(range(values.shape[2]))
        plt.plot(xs, ys, label=label)
        # ys_std = ys.std()
        # ax.fill_between(xs, ys-ys_std, ys+ys_std, alpha=0.2)
    plt.xlabel('Generation')
    plt.ylabel(ylabel)
    plt.legend(title=kwargs['test_kwargs'][0][0])
    if save:
        plt.savefig(f'{kwargs["saves_path"]}{kwargs["name"]}/plots/{ylabel}.png')
    if show:
        plt.show()
    plt.close()


def plot_medians(values, ylabel):
    fig, ax = plt.subplots()
    for test in range(values.shape[0]):
        label = kwargs['test_kwargs'][test + 1][0]
        xs = np.array(range(values.shape[2]))
        ys = np.median(values[test], axis=(0,2))
        plt.plot(xs, ys, label=label)

        ys = np.mean(values[test], axis=(0,2))
        plt.plot(xs, ys, label=label)

        q1 = np.quantile(values[test], 0.25, axis=(0,2))
        q3 = np.quantile(values[test], 0.75, axis=(0,2))
        ax.fill_between(xs, q1, q3, alpha=0.2)
    plt.xlabel('Generation')
    plt.ylabel(ylabel)
    plt.legend(title=kwargs['test_kwargs'][0][0])
    plt.savefig(f'{kwargs["saves_path"]}{kwargs["name"]}/plots/{ylabel}.png')
    plt.show()


def plot_box(values, ylabel, ax=None, save=True, show=True, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    positions = range(len(kwargs['test_kwargs']) - 1)
    ys = [values[test].ravel() for test in range(len(kwargs['test_kwargs']) - 1)]
    # ax.boxplot(
    ax.violinplot(
        ys,
        positions=positions,
        # patch_artist=True,
        showmeans=False,
        showmedians=True,
    )
    ax.yaxis.grid(True)
    ax.set_xticks(ticks=positions, labels=[test[0] for test in kwargs['test_kwargs'][1:]])
    ax.set_xlabel(kwargs['test_kwargs'][0][0])
    ax.set_ylabel(ylabel)
    if save:
        plt.savefig(f'{kwargs["saves_path"]}{kwargs["name"]}/plots/{ylabel}.png')
    if show:
        plt.show()
    plt.close()


# def plot_hist(values, ylabel):
#     fig, ax = plt.subplots()
#     for test in range(values.shape[0]):
#         label = kwargs['test_kwargs'][test + 1][0]
#         xs = values[test, :, -1].ravel()
#         # ax.boxplot(xs,
#         #     positions=[test],
#         #     label=label,
#         #     patch_artist=True,
#         #     # showmeans=False,
#         #     # showfliers=False,
#         #     # medianprops={"color": "white", "linewidth": 0.5},
#         #     # boxprops={"facecolor": "C0", "edgecolor": "white", "linewidth": 0.5},
#         #     # whiskerprops={"color": "C0", "linewidth": 1.5},
#         #     # capprops={"color": "C0", "linewidth": 1.5}
#         # )
#     plt.xlabel('Generation')
#     plt.ylabel(ylabel)
#     plt.legend(title=kwargs['test_kwargs'][0][0])
#     plt.savefig(f'{kwargs["saves_path"]}{kwargs["name"]}/plots/{ylabel}.png')
#     plt.show()


#
# Tables
#

def table_best(obj, **kwargs):
    """Plot the best result of the given run and gen"""
    table = []
    cases = cartesian_prod(*kwargs['domains'])
    y_target = np.array([kwargs['target_func'](*list(xs)) for xs in cases])
    # Calculate values
    y_actual = []
    for case in cases:
        l = Linear([[0] + list(case) + [0], obj[0].copy()], ops=kwargs['ops'], value_lim=kwargs['value_lim'])
        l.run(kwargs['timeout'])
        y_actual = np.append(y_actual, l.regs[-1])
    # Append rows as columns to table
    table.append(range(len(cases)))
    table.append(cases)
    table.append(y_target)
    table.append(y_actual)
    table.append(abs(y_target - y_actual))
    table.append(abs(y_target - y_actual) ** 2)
    # Transpose the table and print each row
    for row in zip(*table):
        row = list(row)
        row[1] = str(row[1])
        row = '{:2} │ {:5} │ {:3} │ {:6.1f} │ {:6.1f} │ {:8.1f} │'.format(*row)
        print(row)
    # Print the total's row
    row = []
    row.append('')
    row.append('')
    row.append('')
    row.append('')
    row.append(sum(abs(y_target - y_actual)))
    row.append(sum((abs(y_target - y_actual)) ** 2))
    row.append(sum((abs(y_target - y_actual)) ** 2) / len(cases))
    row.append((sum((abs(y_target - y_actual)) ** 2) / len(cases)) ** (1 / 2))
    row = '{:2} │ {:5} │ {:3} │ {:6} │ {:6.1f} │ {:8.1f} │ {} │ {} │'.format(*row)
    print(row)

def repeated_table_best(obj, **kwargs):
    """Plot the best result of the given run and gen"""
    table = []
    cases = cartesian_prod(*kwargs['domains'])
    y_target = np.array([kwargs['target_func'](*list(xs)) for xs in cases])
    # Calculate values
    y_actual = []
    l = Linear([[0] * kwargs['num_regs'], obj[0].copy()], ops=kwargs['ops'], value_lim=kwargs['value_lim'])
    for case in cases:
        for j, x in enumerate(case):
            l.mem[0][1+j] = x
        # Evaluate the organism
        l.run(kwargs['timeout'])
        y_actual.append(l.mem[0][-1])
    # Append rows as columns to table
    table.append(range(len(cases)))
    table.append(cases)
    table.append(y_target)
    table.append(y_actual)
    table.append(abs(y_target - y_actual))
    table.append(abs(y_target - y_actual) ** 2)
    # Transpose the table and print each row
    for row in zip(*table):
        row = list(row)
        row[1] = str(row[1])
        row = '{:2} │ {:5} │ {:3} │ {:6.1f} │ {:6.1f} │ {:8.1f} │'.format(*row)
        print(row)
    # Print the total's row
    row = []
    row.append('')
    row.append('')
    row.append('')
    row.append('')
    row.append(sum(abs(y_target - y_actual)))
    row.append(sum((abs(y_target - y_actual)) ** 2))
    row.append(sum((abs(y_target - y_actual)) ** 2) / len(cases))
    row.append((sum((abs(y_target - y_actual)) ** 2) / len(cases)) ** (1 / 2))
    row = '{:2} │ {:5} │ {:3} │ {:6} │ {:6.1f} │ {:8.1f} │ {} │ {} │'.format(*row)
    print(row)


#
# Control
#

def get_best(all_fits, gen=slice(None), **kwargs):
    """Get the best result of the given run and gen"""
    best_orgs = []
    best_fits = []
    best_indices = []
    best_seeds = []
    for test in range(all_fits.shape[0]):
        if kwargs['minimize_fitness']:
            index = np.unravel_index(all_fits[test,:,gen,:].argmin(), all_fits[test,:,gen,:].shape)
        else:
            index = np.unravel_index(all_fits[test,:,gen,:].argmax(), all_fits[test,:,gen,:].shape)
        # Unpack the index as either including generation or not
        if len(index) == 3:
            run, gen_index, org = index
        else:
            run, org = index
            gen_index = gen
        # Store the best organisms and its fitness
        best_orgs.append(load_pop(test,run,**kwargs)[gen_index,org])
        best_fits.append(all_fits[test,run,gen_index,org])
        best_indices.append([test,run,gen_index,org])
        best_seeds.append(load_seed(test,run, **kwargs))
    return best_orgs, best_fits, best_indices, best_seeds


def plot_results(all_fits, **kwargs):
    """Plot all standard plots"""

    # Setup output to save images
    path = f'{kwargs["saves_path"]}{kwargs["name"]}/plots/'
    os.makedirs(path, exist_ok=True)
    print('Plotting results')

    plot_fitness(all_fits, show=True, **kwargs)

    # all_pops = load_pops(**kwargs)
    # lens = np.vectorize(lambda x: len(x[0]))(all_pops)
    # print(lens.shape)
    # plot_means(lens, 'Average Length', show=True, **kwargs)
    # plot_medians(np.vectorize(lambda x: len(x[0]))(all_pops), 'Average Number of Nodes')
    # plot_hist(np.vectorize(lambda x: len(x[0]))(all_pops), 'Average Number of Nodes')






    # Iterate over the best individuals of each test
    bests = zip(*get_best(all_fits, **kwargs))
    for i, best in enumerate(bests):
        best_org, best_fit, best_index, best_seed = best
        test_name = kwargs['test_kwargs'][i+1][0]
        kwargs['seed'] = int(best_seed)
        print(f'Best of {test_name}, Fitness {best_fit}')

        f = kwargs['fitness_func']([best_org.copy()], **kwargs)
        print(f'Fitness check {f}')

        print(best_org)
        l = setup_self_rep(best_org, **kwargs)
        print(l)
        l.run(kwargs['timeout'])
        print(l)



        if 'target_func' in kwargs:
            # kwargs['ops'] = ('STOP', 'LOAD', 'STORE', 'ADD', 'IFEQ')
            # table_best(best_org.copy(), **kwargs)
            l = Linear([[0]*3, best_org[0]], ops=kwargs['ops'], value_lim=kwargs['value_lim'])
            print(l)

            # kwargs['domains'] = [[0,1,2,3,4,5,6],[0,1,2,3,4,5,6]]
            # kwargs['domains'] = [[0, 1, 2, 3, 4, 5, 6, 7, 8]]
            # kwargs['timeout'] = 2**8
            # kwargs['value_lim'] = 2**16
            repeated_table_best(best_org.copy(), **kwargs)


        # l = setup_self_rep(best_org, **kwargs)
        # l.run(kwargs['timeout'])
        # # l = Linear([[0]+[1]+[2]+[0], best_org[0]], ops=kwargs['ops'], value_lim=kwargs['value_lim'])
        # # l.run(kwargs['timeout'])
        # print(l)







    # # Plot best results of each test
    # bests = zip(*get_best(all_fits, gen=kwargs['dynamic_gen']-1, **kwargs))
    # for i, best in enumerate(bests):
    #     best_org, best_fit = best
    #     test_name = kwargs['test_kwargs'][i+1][0]
    #     print(test_name)
    #     print('Fitness: ', best_fit)
    #     f = kwargs['fitness_func']([best_org], gen=kwargs['dynamic_gen']-1, rng=np.random.default_rng(), **kwargs)
    #     print('Recalculated fitness:', f)
    #     f = kwargs['fitness_func']([best_org], gen=kwargs['dynamic_gen']+1, rng=np.random.default_rng(), **kwargs)
    #     print('Recalculated fitness:', f)
    #
    #     # l = run_self_rep(best_org, **kwargs)
    #     # print(l)
    #
    #     # l = setup_self_crossover_overwrite(best_org[0], [-1]*len(best_org[0]), [-2]*len(best_org[0]), **kwargs)
    #     l = setup_mutual_rep(best_org[0], best_org[0], **kwargs)
    #     # print(l)
    #     l.run(kwargs['timeout'])
    #     print(l)
    #
    #
    #     # l = Linear(([0,2,3,0],best_org[1]), ops=kwargs['ops'], value_lim=kwargs['value_lim'])
    #     # l.run(kwargs['timeout'])
    #     # print(l)
    #
    #
    # # Plot best results of each test
    # bests = zip(*get_best(all_fits, gen=slice(kwargs['dynamic_gen'],None), **kwargs))
    # for i, best in enumerate(bests):
    #     best_org, best_fit = best
    #     test_name = kwargs['test_kwargs'][i + 1][0]
    #     print(test_name)
    #     print('Fitness: ', best_fit)
    #     f = kwargs['fitness_func']([best_org], gen=kwargs['dynamic_gen'] - 1, rng=np.random.default_rng(), **kwargs)
    #     print('Recalculated fitness:', f)
    #     f = kwargs['fitness_func']([best_org], gen=kwargs['dynamic_gen'] + 1, rng=np.random.default_rng(), **kwargs)
    #     print('Recalculated fitness:', f)
    #
    #     # l = run_self_rep(best_org, **kwargs)
    #     # print(l)
    #
    #     # l = setup_self_crossover_overwrite(best_org[0], [-1] * len(best_org[0]), [-2] * len(best_org[0]), **kwargs)
    #     # print(l)
    #     l = setup_mutual_rep(best_org[0], best_org[0], **kwargs)
    #     l.run(kwargs['timeout'])
    #     print(l)
    #
    #     l = Linear(([0, 2, 3, 0], best_org[1]), ops=kwargs['ops'], value_lim=kwargs['value_lim'])
    #     l.run(kwargs['timeout'])
    #     print(l)
    #
    #     table_best([best_org[1]], **kwargs)



if __name__ == '__main__':
    # name = 'factorial_0'
    # name = 'mult_9'
    name = 'self_match_1'
    # name = 'triangular_4'
    kwargs = load_kwargs(name, '../../../saves/smlgp/')
    fits = load_fits(**kwargs)
    plot_results(fits, **kwargs)



    # kwargs['rng'] = np.random.default_rng()
    #
    # a = [[
    #     'LOAD', 4, 1, 'REGS_DIRECT',
    #
    #     'LOAD', 2, 1, 'REGS_DIRECT',
    #     'ADD', 2, 1, 'IMMEDIATE',
    #     'MUL', 4, 2, 'REGS_DIRECT',
    #
    #     'LOAD', 3, 1, 'REGS_DIRECT',
    #     'MUL', 3, 2, 'IMMEDIATE',
    #     'ADD', 3, 1, 'IMMEDIATE',
    #     'MUL', 1, 3, 'REGS_DIRECT',
    #
    #     'DIV', 1,
    # ]]

    # a = [[
    #     'ADD', 2, 1, 'REGS_DIRECT',
    #     'STOP', 0,0,0,
    # ]]
    # a = [[
    #     'ADD', 2, 1, 'REGS_DIRECT',
    #     'STOP', 0,0,0,
    # ]]
    # l = Linear([[0]*kwargs['num_regs']]+a, **kwargs)
    # print(l)
    #
    # table_best(a, **kwargs)
    # repeated_table_best(a, **kwargs)
    # f = lgp_rmse([a], **kwargs)
    # print(f)
    # f = repeated_lgp_rmse([a], **kwargs)
    # print(f)



    # Almost works
#     self.mem[1](PROGRAM)
#     0 │ 15, 10, 4, 1 │ 'IFEQ', 2, 4, 'REGS_DIRECT',
#     4 │  3, 14, 3, 12 │ 'IFEQ', 2, 3, 'REGS_INDIRECT',
#     8 │  5, 15, 5, 6 │ 'ADD', 3, 5, 'REGS_DIRECT',
# 12 │  7, 10, 1, 14 │ 'IFEQ', 2, 1, 'CODE_INDIRECT',

    # c = [14,  0,  7, 10, 3, 3,  9,  1, 10, 14,  1,  5, 14,  0,  2,  4]
    # # 'domains': [[1, 2, 3, 4], [1, 2, 3]],
    # kwargs['domains'] = [[0,1,2,3,4,5,6],[0,1,2,3,4,5,6]]
    # kwargs['timeout'] = 32
    # kwargs['value_lim'] = 64
    # table_best([c], **kwargs)


  #  0 │ 130, 22,  3, 105 │ 'SUB',    1,  3, 'IMMEDIATE',
  #  4 │ 250, 81, 115, 186 │ 'MUL',    0, 115, 'REGS_DIRECT',
  #  8 │ 58, 167, 141, 84 │ 'STORE',  2, 141, 'CODE_INDIRECT',
  # 12 │ 62, 201, 77, 192 │ 'IFEQ',   0, 77, 'REGS_INDIRECT',
  # 16 │ 255, 59, 250, 226 │ 'ADD',    2, 250, 'REGS_DIRECT',
  # 20 │ 30, 159, 123, 62 │ 'STORE',  0, 123, 'REGS_INDIRECT',
  # 24 │ 17, 70, 113, 230 │ 'ADD',    1, 113, 'IMMEDIATE',
  # 28 │ 255, 40, 42, 106 │ 'ADD',    1, 42, 'REGS_DIRECT',
  # 32 │ 69, 177, 140, 154 │ 'IFEQ',   0, 140, 'CODE_INDIRECT',
  # 36 │ 81, 178, 81, 222 │ 'SUB',    1, 81, 'REGS_INDIRECT',
  # 40 │ 121, 206, 80, 29 │ 'STORE',  2, 80, 'CODE_INDIRECT',
  # 44 │ 78, 73, 192, 209 │ 'LOAD',   1, 192, 'CODE_INDIRECT',
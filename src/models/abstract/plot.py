"""General functions for plotting results."""
import numpy as np
from matplotlib import pyplot as plt

from models.smlgp.model import Linear
from src.utils.save import sql_query
from src.utils.utils import cartesian_prod

#
# Data Based Plotting
#

def plot_fitness(figsize=None, dpi=None, save=True, show=True, **kwargs):
    """Plot the average of the runs' minimum fitness for each test"""

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Iterate over all tests for each line in the plot
    for test_num, test_values in enumerate(kwargs['test_values']):

        test_name = test_values[0]

        if kwargs['minimize_fitness']:
            ordering_func = 'MIN'
            ax.set_ylabel('Average Min Fitness Value')
        else:
            ordering_func = 'MAX'
            ax.set_ylabel('Average Max Fitness Value')

        # Plot the avg of the min/max for each rep
        # query = f"""
        # SELECT gen, AVG(fit)
        # FROM (
        #     SELECT seed, gen, {ordering_func}(fit) AS fit
        #     FROM data
        #     WHERE test = '{test_name}'
        #     GROUP BY seed, gen
        #     )
        # GROUP BY gen
        # """
        # x, y = zip(*sql_query(query, **kwargs))
        # ax.plot(x, y, label=test_name)

        # Plot the error bands
        query = f"""
        WITH mfits AS (
            SELECT seed, gen, {ordering_func}(fit) AS mfit
            FROM data
            WHERE test = '{test_name}'
            GROUP BY seed, gen
        )
        SELECT mfits.gen, AVG(mfit), SQRT(AVG(POWER(mfit - mean, 2))) AS stdev
        FROM
            mfits
            INNER JOIN (
                SELECT gen, AVG(mfit) AS mean
                FROM mfits
                GROUP BY gen
            ) AS mmean ON mmean.gen = mfits.gen
        GROUP BY mfits.gen
        """
        r = sql_query(query, **kwargs)
        x, y, y_std = zip(*r)
        y = np.array(y)
        ax.plot(x, y, label=test_name)
        ax.fill_between(x, y - y_std, y + y_std, alpha=0.2)

        # query = f"""
        # SELECT seed, gen, {ordering_func}(fit) AS fit
        # FROM data
        # WHERE test = '{test_name}'
        # GROUP BY seed, gen
        # ORDER BY gen
        # """
        # yy = sql_query(query, **kwargs)
        # v = {}
        # for seed, gen, fit in yy:
        #     if seed not in v:
        #         v[seed] = []
        #     v[seed].append(fit)
        # v = list(v.values())
        # v = np.array(v, float)
        # # print(v)
        # s = np.std(v, axis=0)
        # m = np.mean(v, axis=0)
        # # print(s)
        # for i in range(len(s)):
        #     print(f'{i} {y[i]} {y_std[i]} {s[i]} {m[i]}')
        # print(s.shape)
        # y_std = np.std(np.min(all_fits[test], axis=2), axis=0)

    # Scatter plot all points
    # xx = x.reshape((1,len(x),1)).repeat(all_fits.shape[1], axis=0).repeat(all_fits.shape[3], axis=2).ravel()
    # yy = all_fits[test].ravel()
    # plt.scatter(xx, yy, 0.1)

    ax.set_xlabel('Generation')
    plt.legend(title=kwargs['test_label'])
    if save:
        plt.savefig(f'{kwargs['plot_path']}Fitness.svg')
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


def plot_medians(values, ylabel, **kwargs):
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

# def table_best(all_pops, all_fits, **kwargs):
#     """Plot the best result of the given run and gen"""
#     xs = [np.linspace(*domain) for domain in kwargs['domains']]
#     xs = np.array(np.meshgrid(*xs)).reshape((len(xs), -1)).T
#     y_true = np.array([[kwargs['target_func'](*list(x))] for x in xs])
#     table = np.concat((xs, y_true), axis=1)
#     # Iterate over all runs
#     for run in range(len(kwargs['test_kwargs']) - 1):
#         i = all_fits[run, :, :, :].argmin()
#         node = all_pops[run, :, :, :].flatten()[i]
#         y_node = [[node(*x)] for x in xs]
#         tab = np.concat((table, y_node), axis=1)
#         print('\n', node, sep='')
#         for row in tab:
#             print(('f(' + ', '.join(['{}'] * len(kwargs['domains'])) + ') = {} | {}').format(*row))


def table_best(obj, **kwargs):
    """Plot the best result of the given run and gen"""
    table = []
    cases = cartesian_prod(*kwargs['domains'])
    y_target = np.array([kwargs['target_func'](*list(xs)) for xs in cases])
    # Calculate values
    y_actual = []
    for case in cases:
        l = Linear([[0] + list(case) + [0], obj[0]])
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
        row = '{:2} │ {} │ {:2} │ {:4.1f} │ {:4.1f} │ {:5.1f} │'.format(*row)
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
    row = '{:2} │ {:5} │ {:2} │ {:4} │ {:4.1f} │ {:5.1f} │ {} │ {} │'.format(*row)
    print(row)



#
# Control
#

def get_best(**kwargs):
    """Get the best result of the given run and gen"""
    # https://www.sqlite.org/lang_select.html#bareagg
    if kwargs['minimize_fitness']:
        ordering_func = 'MIN'
    else:
        ordering_func = 'MAX'
    query = f"""
        SELECT test, seed, gen, id, {ordering_func}(fit), genotype
        FROM data
        GROUP BY test
    """
    bests = list(sql_query(query, **kwargs))
    for i in range(len(bests)):
        bests[i] = list(bests[i])
        bests[i][5] = kwargs['load_formater_func'](bests[i][5])
    return bests


def plot_grid(all_pops, all_fits, plot_func, title=None, save=True, show=True, **kwargs):
    """Plots a grid of plots over the test kwargs"""
    best = get_best(all_pops, all_fits, **kwargs)
    if len(kwargs['test_kwargs'][0]) < 2:
        print(f'Plotting failed for {plot_func.__name__}')
    else:
        # Values of first test_kwargs
        zipped0 = list(zip(*kwargs['test_kwargs'][1:]))[1]
        array0 = np.empty((len(zipped0),), 'object')
        array0[:] = zipped0
        values0, counts0 = np.unique(array0, return_counts=True)
        # Values of second test_kwargs
        zipped1 = list(zip(*kwargs['test_kwargs'][1:]))[2]
        array1 = np.empty((len(zipped1),), 'object')
        array1[:] = zipped1
        values1, counts1 = np.unique(array1, return_counts=True)
        # Setup grid
        nrows = len(counts0)
        ncols = len(counts1)
        if not ((counts0 == counts0[0]).all() and (counts1 == counts1[0]).all() and nrows * ncols == len(
                kwargs['test_kwargs'][1:])):
            print(f'Plotting failed for {plot_func.__name__}')
        else:
            scale, dpi = 4, 400 # Save resolution
            # scale, dpi = 2, 200
            fig, axs = plt.subplots(nrows, ncols, figsize=(nrows * scale, ncols * scale), dpi=dpi)

        # Plot best results of each test
        for i, tm in enumerate(best):
            print(f'Plotting grid {i+1} of {len(best)} for {plot_func.__name__}')
            plot_func(tm, ax=axs.ravel()[i], title=kwargs['test_kwargs'][i+1][0], show=False, save=False, **kwargs)
        fig.supxlabel(kwargs['test_kwargs'][0][2])
        for col in range(ncols):
            axs[-1, col].set_xlabel(values1[col])
        fig.supylabel(kwargs['test_kwargs'][0][1])
        for row in range(nrows):
            axs[row, 0].set_ylabel(values0[row])
        if save:
            plt.savefig(f'{kwargs["saves_path"]}{kwargs["name"]}/plots/{title}.png')
        if show:
            plt.show()
        plt.close()

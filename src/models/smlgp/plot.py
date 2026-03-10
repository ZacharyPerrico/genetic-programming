"""All functions relevant to plotting."""

import os

from src.models import *
from src.utils.save import load_kwargs, load_pop, load_fits
from src.utils.utils import cartesian_prod


#
# Data Based Plotting
#

def plot_fitness(all_fits, ax=None, save=True, show=True, **kwargs):
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
        y_std = np.std(np.max(all_fits[test], axis=2), axis=0)
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

def get_best(all_fits, gen=slice(None), **kwargs):
    """Get the best result of the given run and gen"""
    best_orgs = []
    best_fits = []
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
    return best_orgs, best_fits


def plot_results(all_fits, **kwargs):
    """Plot all standard plots"""
    path = f'{kwargs["saves_path"]}{kwargs["name"]}/plots/'
    os.makedirs(path, exist_ok=True)
    print('Plotting results')

    # plot_fitness(all_fits, show=True, **kwargs)

    # print(all_fits.shape)

    plot_fitness(all_fits[:,:,:kwargs['dynamic_gen'],:], show=True, **kwargs)
    plot_fitness(all_fits[:,:,kwargs['dynamic_gen']:,:], show=True, **kwargs)

    # plot_means(np.vectorize(lambda x: len(x))(all_pops), 'Average Length', show=False, **kwargs)
    # plot_medians(np.vectorize(lambda x: len(x[0]))(all_pops), 'Average Number of Nodes')
    # plot_hist(np.vectorize(lambda x: len(x[0]))(all_pops), 'Average Number of Nodes')
    # plot_box(all_fits[:,:,-1], 'Final Fitness', show=False, **kwargs)


    # Plot best results of each test
    bests = zip(*get_best(all_fits, **kwargs))
    for i, best in enumerate(bests):
        best_org, best_fit = best
        test_name = kwargs['test_kwargs'][i+1][0]
        print(best_fit)

        l = run_self_rep(best_org, **kwargs)
        print(l)


if __name__ == '__main__':
    name = 'project_test_2'
    kwargs = load_kwargs(name, '../../../saves/smlgp/')
    fits = load_fits(**kwargs)
    plot_results(fits, **kwargs)
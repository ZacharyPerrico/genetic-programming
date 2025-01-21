import numpy as np
from matplotlib import pyplot as plt
import sympy as sp

from gp import function, fitness, run_sim


def plot_min(all_pops, all_fits, **kwargs):
    """Display the history"""

    fig, ax = plt.subplots()
    all_fits = np.array(all_fits)
    x = np.array([range(all_fits.shape[1])]).T

    # x = np.repeat(x, f.shape[0], axis=0)

    # Combined average of all results and trials
    # true_avg_y = np.mean(f, axis=(0,2))
    # plt.plot(x, true_avg_y)

    # Largest and smallest values of all results and trials
    # true_max_y = np.min(f, axis=(0,2))
    # true_min_y = np.max(f, axis=(0,2))
    # ax.fill_between(x, true_min_y, true_max_y, alpha=.5, linewidth=0)

    for i in range(all_fits.shape[0]):
        plt.plot(x, np.min(all_fits[i], axis=1), label=kwargs['labels'][i])
        # Scatter plot all points
        # xx = x.repeat(all_fits.shape[2], axis=1).ravel()
        # yy = all_fits[i].ravel()
        # plt.scatter(xx, yy, 0.1)

    plt.title(kwargs['title'])
    ax.set_yscale('log')
    plt.xlabel('Generation')
    plt.ylabel('Average Fitness Value')
    plt.legend(title=kwargs['label_title'])
    plt.show()

# disp(all_pops, all_fits, labels, 'Types of Leaves', **kwargs)


# def plot(node, x_linspace=(-4,4)):
#     x = np.linspace(*x_linspace)
#     plt.scatter(x, [node(i, algebraic=False) for i in x])
#     plt.plot(x, [node(i, algebraic=False) for i in x])
#     plt.show()


def plot_nodes(nodes, title=None, label_title=None, labels=None, **kwargs):
    """Plot """

    x = np.linspace(*kwargs['x_linspace'])

    label = f'${str(function(sp.Symbol("x"))).replace("**","^")}$'
    plt.scatter(x, function(x), label=label)
    plt.plot(x, function(x))

    for i,node in enumerate(nodes):

        if labels is None:
            label = f'${str(node(sp.Symbol("x"))).replace("**","^")}$'
        else:
            label = f'\"{labels[i]}\", Fitness = {fitness([node], **kwargs)[0]:.3f}'

        plt.scatter(x, [node(i, algebraic=False) for i in x], label=label)
        plt.plot(x, [node(i, algebraic=False) for i in x])

    plt.title(title)
    plt.legend(title=label_title)
    plt.show()


def plot_best(all_pops, all_fits, run=None, gen=slice(None), **kwargs):
    """Plot the best result """
    if run is None:
        runs = range(all_pops.shape[0])
    elif type(run) is not list:
        runs = [run]
    else:
        runs = run
    nodes = []
    # Iterate over all runs
    for run in runs:
        i = all_fits[run,gen,:].argmin()
        node = all_pops[run,gen,:].flatten()[i]
        fit = all_fits[run,gen,:].flatten()[i]
        nodes.append(node)
        # print(np.unravel_index(i, all_fits[run,gen,:].shape))
    plot_nodes(nodes, **kwargs)



def plot_sims(all_pops, all_fits, **kwargs):
    # Display the current histories
    plot_min(all_pops, all_fits, title='', **kwargs)
    # Best result
    plot_best(all_pops, all_fits, title='Best Overall', **kwargs)
    # Best result in last generation
    plot_best(all_pops, all_fits, gen=-1, title='Best of Last Generation', **kwargs)



def save_all(all_fits, all_pops, kwargs):
    np.save('saves/fits', all_fits)
    np.save('saves/pops', all_pops)
    np.save('saves/kwargs', np.array([kwargs]))

def load_all():
    all_fits = np.load('saves/fits.npy')
    all_pops = np.load('saves/pops.npy', allow_pickle=True)
    kwargs = np.load('saves/kwargs.npy', allow_pickle=True)[0]
    return all_fits, all_pops, kwargs

if __name__ == '__main__':
    all_fits, all_pops, kwargs
    plot_sims(all_pops, all_fits, **kwargs)
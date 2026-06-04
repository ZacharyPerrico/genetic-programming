import os

import numpy as np
from matplotlib import pyplot as plt

from src.utils.plot import get_best
from src.utils.save import load_kwargs


def plot_nodes(nodes, result_fitness_func=None, labels=None, title=None, legend_title=None, **kwargs):
    """Plot all given nodes and the fitness function"""

    # Only plot the first domain
    xs = np.linspace(*kwargs['domains'][0])

    # Plot target function if given
    if 'target_func' in kwargs and result_fitness_func is not None:
        label = 'Target Function'
        target_ys = [kwargs['target_func'](x) for x in xs]
        plt.scatter(xs, target_ys, label=label)
        plt.plot(xs, target_ys, lw=5)

    # Plot nodes
    for i, node in enumerate(nodes):
        # Determine label based on what info is known
        if labels is not None:
            label = labels[i]
        elif 'test_kwargs' in kwargs:
            label = kwargs['test_kwargs'][i + 1][0]
        else:
            label = ''
        if 'target_func' in kwargs and result_fitness_func is not None:
            label += f' Fitness = {result_fitness_func([node], **kwargs)[0]:f}'

        # Evaluate and plot real part and imaginary part if applicable
        node_ys = [node(i, eval_method=kwargs['eval_method']) for i in xs]
        plt.scatter(xs, np.real(node_ys), label=label)
        plt.plot(xs, np.real(node_ys))
        if np.iscomplex(node_ys).any():
            label = label.split('Fitness')[0] + 'Imaginary Part'
            plt.scatter(xs, np.imag(node_ys), label=label)
            plt.plot(xs, np.imag(node_ys), ':')

    plt.title(title)
    plt.legend(title=kwargs['test_kwargs'][0][0])
    plt.savefig(f'{kwargs["saves_path"]}{kwargs["name"]}/plots/{title}.png')
    plt.show()


def plot_results(**kwargs):
    """Function to plot results called by this module or through the main module"""
    kwargs['plot_path'] = f'{kwargs["saves_path"]}{kwargs["name"]}/plots/'
    os.makedirs(kwargs['plot_path'], exist_ok=True)
    print('Plotting results')

    # plot_fitness(**kwargs)

    # Plot best results of each test
    bests = get_best(**kwargs)
    for test, seed, gen, id, fit, data in bests:
        print(test, seed, gen, id, fit, data)

        # plot_nodes


if __name__ == '__main__':
    name = 'test/node'
    kwargs = load_kwargs('../../../saves/'+name)
    plot_results(**kwargs)
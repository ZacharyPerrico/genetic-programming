import os

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from models.abstract.plot import get_best, plot_fitness
from models.daggp.methods import simulate_cart_pole
from models.daggp.model import Node
from src.utils.save import load_kwargs
from utils.save import sql_query


def plot_nodes(nodes, labels=None, title=None, fits=None, figsize=None, dpi=None, save=True, show=True, **kwargs):
    """Plot all given nodes and the fitness function"""

    plt.figure(figsize=figsize, dpi=dpi)

    # Only plot the first domain
    xs = kwargs['domains'][0]

    # Plot target function if given
    if 'target_func' in kwargs:
        label = 'Target Function'
        target_ys = [kwargs['target_func'](x) for x in xs]
        plt.scatter(xs, target_ys, label=label)
        plt.plot(xs, target_ys, lw=5)

    # Plot nodes
    for i, node in enumerate(nodes):
        # Determine label based on what info is known
        if labels is not None:
            label = labels[i]
        else:
            label = ''

        # Append fitness to label if possible
        if fits is not None:
            label += f' (Fitness = {fits[i]:f})'
        elif 'target_func' in kwargs:
            label += f' (Fitness = {kwargs['fitness_func']([node], **kwargs)[0]:f})'

        # Evaluate and plot real part and imaginary part if applicable
        node_ys = [node(i, eval_method=kwargs['eval_method']) for i in xs]
        plt.scatter(xs, np.real(node_ys), label=label)
        plt.plot(xs, np.real(node_ys))
        if np.iscomplex(node_ys).any():
            label = label.split('Fitness')[0] + 'Imaginary Part'
            plt.scatter(xs, np.imag(node_ys), label=label)
            plt.plot(xs, np.imag(node_ys), ':')

    # Determine title and suptitle
    if len(labels) == 1:
        title = labels[0]
        plt.suptitle(f'${nodes[0].latex()}$')
    elif 'test_label' in kwargs:
        title = kwargs['test_label']

    plt.ylim(-1,1)

    plt.title(title)
    plt.legend()
    if save:
        plt.savefig(f'{kwargs["plot_path"]}/{title} Plot.svg')
    if show:
        plt.show()
    plt.close()




def plot_graph(node:Node, title=None, fit=None, layout='topo', scale=1, figsize=None, dpi=None, save=True, show=True, **kwargs):
    """Plot the node as a graph"""

    def to_graph(node, verts=None, edges=None, vert_props=None, edge_props=None):
        """Identical to Node.to_lists() but returns extra values"""
        if verts is None:
            node.reset_index()
            verts, edges = [], []
            vert_props, edge_props = [], []
        if node.temp_index is None:
            node.temp_index = len(verts)
            verts.append(node.value)
            # vert_props.append()
            for i, child in enumerate(node.children):
                to_graph(child, verts, edges, vert_props, edge_props)
                edges.append((node.temp_index, child.temp_index))
                edge_props.append(i)
        return verts, edges, vert_props, edge_props

    verts, edges, vert_props, edge_props = to_graph(node)

    # Create networkxs graph
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    G = nx.MultiDiGraph()
    G.add_nodes_from(range(len(verts)))
    G.add_edges_from(edges)
    connectionstyle = [f"arc3,rad={r}" for r in [0, .5]]
    if layout == 'traversal':
        # Traversal layout
        pos = sorted([(n.temp_index, n.depth()) for n in node.nodes()])
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        plt.xlabel('Traversal Order')
        plt.ylabel('Depth')
    else:
        # Topological layout
        for layer, ns in enumerate(nx.topological_generations(G)):
            for n in ns:
                G.nodes[n]["layer"] = layer
        pos = nx.multipartite_layout(G, subset_key="layer")
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=range(len(verts)),
        node_color='white',
        edgecolors='black',
        node_size=600 * scale,
    )
    nx.draw_networkx_labels(
        G,
        pos,
        labels={key: vert for key, vert in enumerate(verts)},
        font_color='black',
        font_size=10 * scale,
    )
    nx.draw_networkx_edges(
        G,
        pos,
        arrowstyle=[["-|>", "->"][i] for i in edge_props],
        edgelist=edges, # Specify edge order
        connectionstyle=connectionstyle,
        arrowsize=20 * scale,
        # edge_color = edge_props,
        # edge_cmap = plt.cm.tab10,
        # edge_vmax = 9,
        width=2 * scale,
        # alpha=0.5,
        node_size=600 * scale,
    )
    # nx.draw_networkx_edge_labels(
    #     G,
    #     pos,
    #     connectionstyle=connectionstyle,
    #     edge_labels = {edges[key]: label for key,label in enumerate(edge_props)},
    #     alpha=0.5,
    #     label_pos=0.0,
    #     node_size=24000 * scale,
    #     bbox=None,
    # )

    plt.suptitle(f'${node.latex()}$')
    plt.title(title)

    # Append fitness to label if possible
    if fit is not None:
        plt.legend(title=f'Fitness = {fit}')
    elif 'fitness_func' in kwargs:
        plt.legend(title=f'Fitness = {kwargs['fitness_func']([node], **kwargs)[0]}')

    if save:
        plt.savefig(f'{kwargs["plot_path"]}/{title} Graph.svg')
    if show:
        plt.show()
    plt.close()



def plot_pole(node, **kwargs):

    x_history, dx_history, theta_history, dtheta_history = simulate_cart_pole(node)

    t = list(range(len(x_history)))
    fig, axs = plt.subplots(4, 1, sharex=True)

    axs[0].plot(t, x_history)
    axs[0].axhline(-2.4, color='red')
    axs[0].axhline(2.4, color='red')
    axs[0].set_ylabel('Cart Position (m)')
    axs[0].grid(True)

    axs[1].plot(t, dx_history)
    axs[1].axhline(-1, color='red')
    axs[1].axhline(1, color='red')
    axs[1].set_ylabel('Cart Velocity (m/s)')
    axs[1].grid(True)

    axs[2].plot(t, theta_history * 180 / np.pi)
    axs[2].axhline(-12, color='red')
    axs[2].axhline(12, color='red')
    axs[2].set_ylabel('Pole Angle (deg)')
    axs[2].grid(True)

    axs[3].plot(t, dtheta_history * 180 / np.pi)
    axs[3].axhline(-1.5, color='red')
    axs[3].axhline(1.5, color='red')
    axs[3].set_ylabel('Pole Angular Velocity (deg/s)')
    axs[3].grid(True)

    plt.xlabel('Time (s)')
    plt.tight_layout()
    plt.show()


# def plot_dist(**kwargs):
#
#     r = sql_query("""
#         WITH sub AS (
#             SELECT gen, COUNT() AS c, fit, data
#             FROM data
#             GROUP BY gen, fit, data
#         )
#         SELECT *
#         FROM sub
#         WHERE data IN (
#             SELECT data
#             FROM sub
#             GROUP BY data
#             HAVING fit = 0
#         )
#     """, **kwargs)
#     for i in r: print(i)
#
#     xy = {}
#     for x,y,fit,label in r:
#         if label not in xy:
#             xy[label] = [0] * kwargs['num_gens']
#         xy[label][x] = y
#
#     fig, ax = plt.subplots()
#
#     for label in xy:
#         plot_label = str(kwargs['load_formater_func'](label))
#         plt.plot(xy[label], label=plot_label)
#
#     # labels = [str(kwargs['load_formater_func'](label)) for label in xy.keys()]
#     # ax.stackplot(list(range(kwargs['num_gens'])), xy.values(), labels=labels, alpha=0.8)
#
#     title = 'dist'
#     ax.legend(title='Equations')
#     plt.savefig(f'{kwargs["plot_path"]}/{title}.png')
#     plt.show()




# def plot_dist(**kwargs):
#
#     r = sql_query("""
#         WITH sub AS (
#             SELECT gen, COUNT() AS c, fit, data, test
#             FROM data
#             GROUP BY gen, fit, data, test
#         )
#         SELECT *
#         FROM sub
#         WHERE data, test IN (
#             SELECT data, test
#             FROM sub
#             GROUP BY data, test
#             HAVING fit = 0
#         )
#     """, **kwargs)
#     for i in r: print(i)
#
#     xy = {}
#     for x,y,fit,label in r:
#         if label not in xy:
#             xy[label] = [0] * kwargs['num_gens']
#         xy[label][x] = y
#
#     fig, ax = plt.subplots()
#
#     for label in xy:
#         plot_label = str(kwargs['load_formater_func'](label))
#         plt.plot(xy[label], label=plot_label)
#
#     # labels = [str(kwargs['load_formater_func'](label)) for label in xy.keys()]
#     # ax.stackplot(list(range(kwargs['num_gens'])), xy.values(), labels=labels, alpha=0.8)
#
#     title = 'dist'
#     ax.legend(title='Equations')
#     plt.savefig(f'{kwargs["plot_path"]}/{title}.png')
#     plt.show()






def plot_sql_query(query, save=True, show=True, **kwargs):
    """
    Plots a query result as (x, y, *label) with the label containing one or more columns.
    A plot is made for each label column but only colored in respect to the first
    """
    print('Running query')
    result, col_names = sql_query(query, True, **kwargs)
    print('Formating query results')
    # Format result into a dict with each entry being a list of y values with the key as the label
    plot_dict = {}
    for x, y, *key in result:
        key = tuple(key)
        if key not in plot_dict:
            plot_dict[key] = [0] * kwargs['num_gens']
        plot_dict[key][x] = y
    print('Plotting query results')
    # Plot each entry in the dict of plots
    # Plots with the same first key value will share plot colors and labels
    plot_colors = {}
    for key in plot_dict:
        y_values = plot_dict[key]
        label = key[0] if type(key) == tuple else key
        if label not in plot_colors:
            p = plt.plot(y_values, label=label)
            plot_colors[label] = p[0].get_color()
        else:
            plt.plot(y_values, color=plot_colors[label])
    plt.xlabel(col_names[0])
    plt.ylabel(col_names[1])
    legend_title = col_names[2] if len(col_names) > 1 else None
    plt.legend(title=legend_title)
    if save:
        plt.savefig(f'{kwargs['plot_path']}{col_names[1]}.svg')
    if show:
        plt.show()
    plt.close()



def plot_lens(**kwargs):
    plot_sql_query(
        """
        SELECT 
            gen AS 'Generation', 
            AVG(LENGTH(genotype)) AS 'Average Length', 
            test AS 'Field', 
            seed
        FROM data
        GROUP BY gen, test, seed
        """,
        ylabel='Length',
        **kwargs
    )


def plot_conv(**kwargs):
    plot_sql_query("""
        SELECT 
            gen AS 'Generation', 
            COUNT() AS 'Count', 
            test AS 'Field'
        FROM data
        WHERE fit = 0
        GROUP BY gen, test
    """, **kwargs)


def plot_dist(**kwargs):
    plot_sql_query("""
        WITH sub AS (
            SELECT gen, COUNT() AS c, fit, data, test
            FROM data
            GROUP BY gen, fit, data, test
        )
        SELECT 
            gen AS 'Generation',
            c AS 'Count',
            test AS 'Field',
            data AS 'Data'
        FROM sub
        WHERE (data, test) IN (
            SELECT data, test
            FROM sub
            GROUP BY data, test
            HAVING fit < 3 AND MAX(c) > 5
        )
    """, **kwargs)




def plot_results(**kwargs):
    """Function to plot results called by this module or through the main module"""
    kwargs['plot_path'] = f'{kwargs["saves_path"]}/plots/'
    os.makedirs(kwargs['plot_path'], exist_ok=True)
    print('Plotting results')

    # Append plot kwargs
    kwargs |= {
        'figsize': (6.4, 4.8),
        'dpi': 100,
        'save': True,
        'show': True,
        'scale': .35,
    }

    kwargs['domains'] = [list(np.linspace(-3,3,100))]

    # plot_dist(**kwargs)
    # plot_conv(**kwargs)
    plot_lens(**kwargs)

    # quit()

    plot_fitness(**kwargs)

    # Plot best results of each test
    bests = get_best(**kwargs)

    # Plot all the best results together
    tests, seeds, gens, ids, fits, datas = zip(*bests)
    # plot_nodes(datas, fits=fits, labels=tests, **kwargs)

    # Plot each best result individually
    for test, seed, gen, id, fit, data in bests:
        print(f'Best of {test}, (Fit = {fit}) at ({seed}, {gen}, {id}, {data})')
        # plot_nodes([data], fits=[fit], labels=[test], **kwargs)
        plot_graph(data, fit=fit, title=test, **kwargs)

        plot_pole(data)



# Manually load and plot saved results
if __name__ == '__main__':
    # name = 'tuning'
    # name = 'real_dist'
    name = 'pole_test'
    kwargs = load_kwargs('../../../saves/daggp/'+name)
    plot_results(**kwargs)
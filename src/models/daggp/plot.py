import os

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from models import koza_3
from src.models import Node
from src.utils.plot import get_best, plot_fitness
from src.utils.save import load_kwargs
from utils.save import sql_query


def plot_nodes(nodes, labels=None, title=None, **kwargs):
    """Plot all given nodes and the fitness function"""

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
        if 'target_func' in kwargs:
            # Append fitness to label if possible
            label += f' Fitness = {kwargs['fitness_func']([node], **kwargs)[0]:f}'

        # Evaluate and plot real part and imaginary part if applicable
        node_ys = [node(i, eval_method=kwargs['eval_method']) for i in xs]
        plt.scatter(xs, np.real(node_ys), label=label)
        plt.plot(xs, np.real(node_ys))
        if np.iscomplex(node_ys).any():
            label = label.split('Fitness')[0] + 'Imaginary Part'
            plt.scatter(xs, np.imag(node_ys), label=label)
            plt.plot(xs, np.imag(node_ys), ':')

    # Determine title
    if len(labels) == 1:
        title = labels[0]
        plt.suptitle(f'${nodes[0].latex()}$')

    plt.title(title)
    plt.legend(title=kwargs['test_label'])
    plt.savefig(f'{kwargs["plot_path"]}/{title}.png')
    plt.show()


def plot_graph(node:Node, layout='topo', scale=1, title=None, **kwargs):
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
    fig, ax = plt.subplots()
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
    if 'result_fitness_func' in kwargs:
        plt.legend(title=f'Fitness = {kwargs['result_fitness_func']([node], **kwargs)[0]}')
    if 'name' in kwargs:
        plt.savefig(f'saves/{kwargs["name"]}/plots/{title}.png')
    plt.show()





def plot_dist(**kwargs):

    # r = sql_query("""
    #     SELECT  gen, COUNT(), data
    #     FROM data
    #     GROUP BY gen, data
    #     HAVING COUNT() > 3
    # """, **kwargs)
    r = sql_query("""
        SELECT  gen, COUNT(), data
        FROM data
        GROUP BY gen, data
        HAVING COUNT() > 0
    """, **kwargs)
    for i in r: print(i)

    d = {}
    for x,y,label in r:
        if label not in d:
            d[label] = []
        d[label].append([x,y])

    fig, ax = plt.subplots()
    for label in d:
        x,y = zip(*sorted(d[label]))
        if max(y) > 20:
            label = str(kwargs['load_formater_func'](label))
            plt.plot(x, y, label=label)

    ax.legend(loc='upper left', reverse=True)
    plt.show()






def plot_results(**kwargs):
    """Function to plot results called by this module or through the main module"""
    kwargs['plot_path'] = f'{kwargs["saves_path"]}/plots/'
    os.makedirs(kwargs['plot_path'], exist_ok=True)
    print('Plotting results')

    plot_fitness(**kwargs)

    plot_dist(**kwargs)

    # Plot best results of each test
    bests = get_best(**kwargs)
    for test, seed, gen, id, fit, data in bests:
        print(test, seed, gen, id, fit, data)

        plot_nodes([data], labels=[test], **kwargs)

        plot_graph(data)


if __name__ == '__main__':
    name = 'test/node'
    kwargs = load_kwargs('../../../saves/'+name)
    plot_results(**kwargs)

    # x = Node('x')
    # f = x + x
    # g = 2 * f + f
    # plot_graph(g)

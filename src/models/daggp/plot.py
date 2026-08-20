import os

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt, animation

from models.abstract.plot import get_best, plot_fitness, plot_sql_query
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
        target_ys = [kwargs['target_func'](x, **kwargs) for x in xs]
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

    # plt.ylim(-1,1)

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

    if save and 'plot_path' in kwargs:
        plt.savefig(f'{kwargs["plot_path"]}/{title} Graph.svg')
    if show:
        plt.show()
    plt.close()










def plot_pole(node, title=None, fit=None, figsize=None, dpi=None, save=True, show=True, **kwargs):

    x_history, dx_history, theta_history, dtheta_history, force_history = simulate_cart_pole(node, **kwargs)
    # t = list(range(int(fit)))
    t = list(range(kwargs['timeout']))
    fig, axs = plt.subplots(5, 1, sharex=True, figsize=figsize, dpi=dpi)
    plt.suptitle(f'${node.latex()}$')
    plt.xlabel('Time (s)')
    # plt.tight_layout()

    axs[0].set_title(title)
    axs[0].plot(t, x_history[:len(t)])
    axs[0].axhline(-kwargs['position_boundary'], color='red')
    axs[0].axhline(kwargs['position_boundary'], color='red')
    # axs[0].set_ylabel('Cart Position (m)')
    axs[0].set_ylabel('$x$ (m)')

    axs[1].plot(t, dx_history[:len(t)])
    # axs[1].axhline(-1, color='red')
    # axs[1].axhline(1, color='red')
    # axs[1].set_ylabel('Cart Velocity (m/s)')
    axs[1].set_ylabel('$\\dot{x}$ (m)')

    axs[2].plot(t, theta_history[:len(t)] * 180 / np.pi)
    axs[2].axhline(-kwargs['angle_boundary'] * 180 / np.pi, color='red')
    axs[2].axhline(kwargs['angle_boundary'] * 180 / np.pi, color='red')
    # axs[2].set_ylabel('Pole Angle (deg)')
    axs[2].set_ylabel('$\\theta$ (deg)')

    axs[3].plot(t, dtheta_history[:len(t)] * 180 / np.pi)
    # axs[3].axhline(-1.5, color='red')
    # axs[3].axhline(1.5, color='red')
    # axs[3].set_ylabel('Pole Angular Velocity (deg/s)')
    axs[3].set_ylabel('$\\dot{\\theta}$ (deg/s)')

    axs[4].plot(t, force_history[:len(t)])
    # axs[4].axhline(-10, color='red')
    # axs[4].axhline(10, color='red')
    # axs[4].set_ylabel('Force (N)')
    axs[4].set_ylabel('$F$ (N)')

    if save:
        plt.savefig(f'{kwargs["plot_path"]}/{title} Pole Plot.svg')
    if show:
        plt.show()
    plt.close()


def animate_cart_pole(node, title=None, fit=None, figsize=None, dpi=None, save=True, show=True, **kwargs):
    time_scale = 2
    frame_skip = 1
    x_history, dx_history, theta_history, dtheta_history, force_history = simulate_cart_pole(node, **kwargs)
    t = list(range(len(x_history)))
    # Generic plot components
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(autoscale_on=False, xlim=(min(x_history)-1, max(x_history)+1), ylim=(-0.5, 1.5))
    # ax = fig.add_subplot(autoscale_on=False, ylim=(-0.5, 1.5))
    ax.set_aspect('equal')
    ax.grid()
    # Animation artists
    cart = ax.scatter([], [], marker='s', s=100, c='white', edgecolors='black')
    pole, = ax.plot([], [], lw=2, c='orange')
    # trace, = ax.plot([], [], '.-', lw=1, ms=2)
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
    # Animation function
    def animate(i):
        pos_0 = [x_history[i], 0]
        pos_1 = [np.sin(theta_history[i]) + x_history[i], np.cos(theta_history[i])]
        cart.set_offsets(pos_0)
        pole.set_data([pos_0[0], pos_1[0]], [pos_0[1], pos_1[1]])
        time_text.set_text(f'Time = {(i*0.02*time_scale):.3f} sec')
        return pole, cart, time_text
    ani = animation.FuncAnimation(
        fig=fig,
        func=animate,
        frames=range(0, len(x_history), frame_skip),
        interval=0.02 * 1000 * time_scale * frame_skip,
        blit=True,
    )
    # if save:
        # ani.save(filename=f'{kwargs["plot_path"]}/{title} Pole Animation.mp4')
        # writer = animation.FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        # ani.save(filename=f'{kwargs["plot_path"]}/{title} Pole Animation.mp4', writer=writer)
        # print('saved')
    if show:
        plt.show()
    plt.close()


def plot_lens(**kwargs):
    plot_sql_query("""
        SELECT 
            gen AS 'Generation', 
            AVG(LENGTH(genotype)) AS 'Mean Save Length', 
            test AS 'Field', 
            seed
        FROM data
        GROUP BY gen, test, seed
    """, **kwargs)


def plot_perfect_fits(**kwargs):
    if 'fitness_threshold' in kwargs:
        target_fit = kwargs['fitness_threshold']
    else:
        target_fit = 0
    if kwargs['minimize_fitness']:
        inequal = '>='
    else:
        inequal = '<='
    plot_sql_query(f"""
        SELECT
            A.gen AS 'Generation', 
            SUM(IFNULL(B.counts, 0)) AS 'Number of Perfect Fits', 
            A.test AS 'Field'
        FROM (
            SELECT gen, test
            FROM data
            GROUP BY gen, test
        ) AS A
        LEFT JOIN (
            SELECT gen, test, COUNT() as counts
            FROM data
            WHERE fit = {target_fit}
            GROUP BY gen, test
        ) AS B
        ON A.gen {inequal} B.gen AND A.test = B.test
        GROUP BY A.gen, A.test
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


def plot_all_reps(**kwargs):
    plot_sql_query("""
        SELECT 
            gen AS 'Generation', 
            MAX(fit) AS 'Fitness',
            test AS 'Field',
            seed
        FROM data
        GROUP BY gen, test, seed
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
        'scale': 1,
        # 'domains': [list(np.linspace(
        #     min(kwargs['domains'][0]),
        #     max(kwargs['domains'][0]),
        #     100))]
    }

    # plot_all_reps(**kwargs)
    plot_fitness(**kwargs)
    plot_lens(**kwargs)
    # plot_perfect_fits(**kwargs)
    # plot_dist(**kwargs)

    # quit()

    # Plot best results of each test
    bests = get_best(**kwargs)

    # Plot all the best results together
    tests, seeds, gens, ids, fits, genotypes, tests_kwargs = zip(*bests)
    if 'target_func' in kwargs:
        plot_nodes(genotypes, fits=fits, labels=tests, **kwargs)

    # Plot each best result individually
    for test, seed, gen, index, fit, genotype, test_kwargs in bests:
        print(f'Plotting best of "{test}" (Fit = {fit}) at ({seed}, {gen}, {index}, {genotype})')
        plot_graph(genotype, fit=fit, title=test, **test_kwargs)
        if 'target_func' in test_kwargs:
            plot_nodes([genotype], fits=[fit], labels=[test], **kwargs)
        else:
            plot_pole(genotype, fit=fit, title=test, **test_kwargs)
            animate_cart_pole(genotype, fit=fit, title=test, **test_kwargs)


# Manually load and plot saved results
if __name__ == '__main__':
    # name = 'tuning'
    # name = 'real_dist'
    # name = 'pole_10_real_sign'
    name = 'fix_pole_success'
    kwargs = load_kwargs('../../../saves/daggp/' + name)
    # plot_results(**kwargs)

    e = Node('e')
    i = Node('i')
    pi = Node('pi')
    x = Node('x')
    y = Node('y')
    z = Node('z')


    # a = x
    # b = y + a
    # c = z + a + b
    # f = c

    # f = Node.const(9-15j)#.to_tree()

    f = Node.sin(x) #+ Node.cos(x)

    # f = x + 2 + 2

    f = f.limited()

    # f = Node.const(2**9).to_tree()

    # b.replace(e)

    #
    #
    # g = 2 * x + 9
    #
    # f = g + g * 7
    #
    # g.replace(e)

    # f = Node.sin(x) + Node.cos(x)
    # f = Node(7j)
    # f = f.limited(consts=True)
    plot_graph(f)


    # Append plot kwargs
    # kwargs |= {
    #     'figsize': (6.4, 4.8),
    #     'dpi': 100,
    #     'save': False,
    #     'show': True,
    #     'scale': 1,
    # }
    #
    # x0 = Node('x0')
    # x1 = Node('x1')
    # x2 = Node('x2')
    # x3 = Node('x3')
    # y = x1 + (x0 * x3)
    # plot_pole(y, title='test', **kwargs)

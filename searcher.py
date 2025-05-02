import itertools

from node import *

def builder(ops, terminals, reps=1, data=None):

    data = [[Node(t) for t in terminals]] if data is None else data

    data.append([])

    for base in data[-2]:
        l = len([n for n in base.nodes() if len(n) == 0])
        # node to replace
        for i in range(l):
            for op in ops:
                for op_terminals in itertools.product(terminals, repeat=2):
                    op_terminals = [Node(op_terminal) for op_terminal in op_terminals]
                    new_node = Node(op, op_terminals)
                    root = base.copy()
                    # Get a list of all terminals
                    nodes = [n for n in root.nodes() if len(n) == 0]
                    # Only replace nodes that match the first
                    if nodes[i].value == terminals[0]:
                        # continue
                        nodes[i].replace(new_node)
                        data[-1].append(root)

    if reps > 1:
        builder(ops=ops, terminals=terminals, reps=reps - 1, data=data)

    return data


# def finder(y, *x, reps=2, ops=('+', '-', '*', '/', '**')):
#     terminals = ['x_'+str(i) for i in range(len(x))]
#     y = np.array(y)
#     x = [np.array(x_i) for x_i in x]
#     built = builder(ops=ops, terminals=terminals, reps=reps)
#     for node in built[-1]:
#         y_n = node(*x)
#         if (y_n > 1e10).any():
#             print('Found:', y_n, node)

alias = {
    '0000': 'False',
    '0001': '$a_1$ and $a_2$',
    '0010': '$a_1$ and not $a_2$',
    '0011': '$a_1$',
    '0100': 'not $a_1$ and $a_2$',
    '0101': '$a_2$',
    '0110': '$a_1$ xor $a_2$',
    '0111': '$a_1$ or $a_2$',
    '1000': '$a_1$ nor $a_2$',
    '1001': '$a_1$ xnor $a_2$',
    '1010': 'not $a_2$',
    '1011': '$a_1 \\leftarrow a_2$',
    '1100': 'not $a_1$',
    '1101': '$a_1 \\rightarrow a_2$',
    '1110': '$a_1$ nand $a_2$',
    '1111': 'True',
}

def finder(*x, reps=2, ops=('+', '-', '*', '/', '**')):
    terminals = ['x_' + str(i) for i in range(len(x))]
    x = [np.array(x_i) for x_i in x]
    built = builder(ops=ops, terminals=terminals, reps=reps)

    # table
    # table = {}
    table = {tuple(int(j) for j in '{:04b}'.format(i)) : [0]*len(built) for i in range(16)}

    table['other'] = [0]*len(built)
    table['total'] = [0]*len(built)

    for col in range(len(built)):
        # Calculate values of all nodes
        ys = []
        for node in built[col]:
            y = node(*x)
            ys.append(y)
            # if ((y != 1) & (y != 0)).any():
            #     print('Found:', y, node)

        vals, counts = np.unique(ys, axis=0, return_counts=True)

        # Move unique to the dict table
        for v, c in zip(vals, counts):
            t = tuple(v)
            if t in table:
                table[t][col] = c
            else:
                if ((v == 1) | (v == 0)).all():
                    table[t] = [0]*len(built)
                    table[t][col] = c
                else:
                    print(v,c)
                    table['other'][col] += c
            table['total'][col] += c

    # Print the table as LaTeX
    for key in table.keys():
        if type(key) != str:
            bit_str = "".join(str(int(i)) for i in key if type(key) != str)
            a = alias[bit_str]
        else:
            bit_str = key
            a = ''
        # bit_str = "".join(map(str, map(int, key)))
        row = a + ' & ' + bit_str + ' & ' + ' & '.join(map(str, table[key])) + ' \\\\'
        print(row)


if __name__ == '__main__':

    # f = (x+x).to_tree()

    # b = finder([[x]], ['+','-','*','/','**'], ['x_0','x_1'], 2)
    # b = builder(['+', '-', '*', '/'], ['x_0', 'x_1'], 2)

    # for i in b[-1]:
    #     print(i)

    # Boolean
    # finder(
    #     [1,1,math.inf,1],
    #     [0,0,1,1],
    #     [0,1,0,1],
    #     reps=3
    # )

    finder(
        [0, 0, 1, 1],
        [0, 1, 0, 1],
        reps=4
    )

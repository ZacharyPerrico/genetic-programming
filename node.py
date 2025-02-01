import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import random
import math
from math import sin


class Node:
    """A basic class for genetic programming. A Node holds a single value and points to zero or more children Nodes."""

    # All possible values for a node and the number of children it can have
    ops = {
        '+': 2,
        '-': 2,
        '*': 2,
        '/': 2,
        # '**': 2,
        'min': 2,
        'max': 2,
        'abs': 1,
        'if_then_else': 3,
        '&': 2,
        '|': 2,
        '%': 2,
        'sin': 1,
    }

    terminals = [
        'x',
    ]

    def __init__(self, value, children=None):
        self.parent = None
        # If the value is already a node use its values
        # This allows both objects and Nodes to be cast to a Node
        if type(value) == Node:
            self.children = value.copy().children
            self.value = value.value
        # Cast int to a Node containing only valid terminals
        # elif type(value) == int:
        #     # Copy values from the const Node into self
        #     int_node = Node.const(value)
        #     self.children = int_node.copy().children
        #     self.value = int_node.value
        else:
            self.value = value
            self.children = children if children is not None else []

    #
    # Children
    #

    @property
    def children(self):
        return self._children

    @children.setter
    def children(self, children):
        """Setting a child also sets the parent of the child"""
        for child in children:
            child.parent = self
        self._children = children

    def __len__(self): return len(self.children)
    def __getitem__(self, i): return self.children[i]
    def __setitem__(self, i, value): self.children[i] = value
    def __iter__(self): yield from self.children

    def nodes(self, node_list=None):
        """Returns a list of all nodes"""
        if node_list is None: node_list = []
        node_list.append(self)
        for child in self:
            child.nodes(node_list)
        return node_list

    def depth(self):
        return max([1] + [child.depth() for child in self.children])

    def root(self):
        """Returns the root of the tree"""
        return self if self.parent is None else self.parent.root()
        # if self.parent is None:
        #     return self
        # else:
        #     return self.parent.root()

    def replace(self, new_node):
        """Replaces this node and all children with a new branch"""
        # Create a copy of the new node
        new_node = new_node.copy()
        # Return the new node if self is the root of the tree
        if self.parent is None: return new_node
        # Parent's index for self
        parent_index = self.parent.children.index(self)
        # Replace the parent's reference to self
        self.parent[parent_index] = new_node
        # Replace the new Node's reference to parent
        new_node.parent = self.parent
        # Remove self reference to parent
        self.parent = None
        # Return the full new tree
        return new_node.root()

    #
    # Evaluation
    #

    def __call__(self, x, algebraic=False):
        """Calling evaluates the value of the entire tree"""

        # Simplify algebraically before evaluation
        if algebraic:
            return self.simplify().evalf(subs={'x': x})

        # Evaluate as is
        else:
            match self.value:
                case 'x': return x
                case '+': return self[0](x) +  self[1](x)
                case '-': return self[0](x) -  self[1](x)
                case '*': return self[0](x) *  self[1](x)
                case '**': return self[0](x) ** self[1](x)
                case '/': return 1 if self[1](x) == 0 else self[0](x) / self[1](x)
                case '|': return self[0](x) or self[1](x)
                case '&': return self[0](x) and self[1](x)
                case 'min': return min(self[0](x), self[1](x))
                case 'max': return max(self[0](x), self[1](x))
                case 'abs': return abs(self[0](x))
                case 'if_then_else': return self[1](x) if self[0](x) else self[2](x)
                case 'if_then_else': return sin(self[0](x))
            return self.value

    #
    # Utils
    #

    def simplify(self):
        return sp.sympify(self(sp.Symbol('x')))

    def __str__(self):
        if len(self) == 0:
            return str(self.value)
        elif self.value in ['+','-','*','/','**','&','|']:
            return f'({self[0]}{self.value}{self[1]})'
        else:
            return self.value + '(' + ','.join([str(c) for c in self]) + ')'

    def __repr__(self):
        """String representation"""
        return str(self)

    def copy(self):
        """Returns a recursive deepcopy of all Nodes"""
        return Node(self.value, [child.copy() for child in self])

    #
    # Native Python Conversion
    #

    @staticmethod
    def const(n):
        return n
        if n == 0:
            return x - x
        elif n > 0:
            # return sum([x]*(n-1),x)/x
            return sum([x / (x - x)] * (n - 1), x / (x - x))
        else:
            return sum([x] * (n - 1), x) / (x - x - x)

    @staticmethod
    def op(operation, *operands):
        """Return a new Node from an operation on other Nodes"""
        # Convert operands to a list to be modified
        operands = list(operands)
        # Cast each operand to a Node or copy it if it is already a Node
        for i in range(len(operands)):
            if type(operands[i]) != Node:
                operands[i] = Node(operands[i])
            else:
                operands[i] = operands[i].copy()
        # Return a new Node with the operands as the children
        return Node(operation, operands)

    def      __add__(self, other): return Node.op('+',  self, other)
    def      __sub__(self, other): return Node.op('-',  self, other)
    def      __mul__(self, other): return Node.op('*',  self, other)
    def  __truediv__(self, other): return Node.op('/',  self, other)
    def      __pow__(self, other): return Node.op('**', self, other)
    def     __radd__(self, other): return Node.op('+',  other, self)
    def     __rsub__(self, other): return Node.op('-',  other, self)
    def     __rmul__(self, other): return Node.op('*',  other, self)
    def __rtruediv__(self, other): return Node.op('/',  other, self)
    def     __rpow__(self, other): return Node.op('**', other, self)

    def  __and__(self, other): return self * other
    def __rand__(self, other): return other * self
    def   __or__(self, other): return self + other
    def  __ror__(self, other): return other + self
    def   __eq__(self, other): return Node.const(0) / (self - other)
    def  __abs__(self): return (self * self) ** (Node.const(1) / Node.const(2))
    def   __lt__(self, other): return (Node.const(1) - abs(self - other) / (self - other)) / Node.const(2)
    def   __gt__(self, other): return (Node.const(1) - abs(other - self) / (other - self)) / Node.const(2)
    def   __le__(self, other): return (abs(other - self) / (other - self) + Node.const(1)) / Node.const(2)
    def   __ge__(self, other): return (abs(self - other) / (self - other) + Node.const(1)) / Node.const(2)

    def __mod__(self, other):
        if other == 2:
            return (1 - (-1) ** self) / 2

    @staticmethod
    def min(*args): return args[0] * (args[0] < args[1]) + args[1] * (args[0] >= args[1])
    @staticmethod
    def max(*args): return args[0] * (args[0] > args[1]) + args[1] * (args[0] <= args[1])

    @staticmethod
    def if_then(cond, if_true, if_false=None):
        if if_false is None:
            return cond * if_true
        else:
            return cond * if_true + (Node.const(1) - cond) * if_false


x = Node('x')




# x*(0.5 + 0.5*(x**2)**0.5/x)
# f = Node.max(0, x)

# f = Node.const()

# f = abs(x)

# g = x

# f = (x==-1)
# for i in range(2,9):
#     f = f + (x==-i)

if __name__ == '__main__':

    # ReLu
    f = Node.if_then(
        x >= 0,
        x
    )
    # x*(0.5 + 0.5*(x**2)**0.5/x)

    # Collatz Conjecture
    f = Node.if_then(
        x % 2,
        3 * x + 1,
        x / 2,
    )

    # f = Divide[1,4] (2 + 7 x - Power[\(40)-1\(41),x] (2 + 5 x))
    # f = x*((-1)**x/2 + 1/2)/2 + (1/2 - (-1)**x/2)*(3*x + 1)

    print(f)
    print(f.simplify())


    i = 9

    # for _ in range(10):

    while i != 1:
        print(i)
        i = f(i)

    # print(f(1))
    # print(f(3, True))
    # print(f(sp.Symbol('x'), True))
    # plot_nodes([f], (-2,2))


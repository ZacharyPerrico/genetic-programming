import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import random
import math

from sympy.strategies.branch import condition


class Node:
    """A Node"""

    ops = ('+', '-', '*', '/')

    def __init__(self, value, children=None):
        self.parent = None
        # If the value is already a node use its values
        # This allows both objects and Nodes to be cast to a Node
        if type(value) == Node:
            self.children = value.copy().children
            self.value = value.value
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

    def __len__(self):
        return len(self.children)

    def __getitem__(self, i):
        return self.children[i]

    def __setitem__(self, i, value):
        self.children[i] = value

    def __iter__(self):
        yield from self.children

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
        if self.parent is None:
            return self
        else:
            return self.parent.root()

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
    # Evaluation and native Python operations
    #

    def __call__(self, x, algebraic=False):
        """Calling evaluates the value of the entire tree"""

        # Simplify algebraically before evaluation
        if algebraic:
            # The Node's value is not a string, return it
            if type(self.value) != str:
                return sp.Number(self.value)
            # The Node's value is a variable, return  of x
            if self.value == 'x':
                return sp.Symbol('x')
            # The node has children, recursively evaluate them
            return_value = None
            match self.value:
                case '+':  return_value = self[0](None, algebraic) +  self[1](None, algebraic)
                case '-':  return_value = self[0](None, algebraic) -  self[1](None, algebraic)
                case '*':  return_value = self[0](None, algebraic) *  self[1](None, algebraic)
                case '**': return_value = self[0](None, algebraic) ** self[1](None, algebraic)
                case '/':
                    # Return 1 if dividing by zero
                    if self[1](None, algebraic) == 0:
                        return_value = sp.Number(1)
                    else:
                        return_value = self[0](None, algebraic) / self[1](None, algebraic)
            if x is None:
                return return_value
            else:
                return return_value.evalf(subs={'x': x})

        # Evaluate as is
        else:
            if type(self.value) != str:
                return self.value
            # The Node's value is a variable, return the value of x
            elif self.value == 'x':
                return x
            # The node has children, recursively evaluate them
            match self.value:
                case '+':  return self[0](x, algebraic) +  self[1](x, algebraic)
                case '-':  return self[0](x, algebraic) -  self[1](x, algebraic)
                case '*':  return self[0](x, algebraic) *  self[1](x, algebraic)
                case '**': return self[0](x, algebraic) ** self[1](x, algebraic)
                case '/':
                    if self[1](x, algebraic) == 0:
                        return 1
                    else:
                        # print('HERE', self[0](x, algebraic))
                        return self[0](x, algebraic) / self[1](x, algebraic)

    #
    # Utils
    #

    @staticmethod
    def const(n):
        # return n
        if n == 0:
            return x - x
        elif n > 0:
            # return sum([x]*(n-1),x)/x
            return sum([x / (x - x)] * (n - 1), x / (x - x))
        else:
            return sum([x] * (n - 1), x) / (x - x - x)

    def __str__(self):
        if len(self) == 2:
            return f'({self[0]}{self.value}{self[1]})'
        else:
            return str(self.value)

    def __repr__(self):
        """String representation"""
        return str(self)

    def copy(self):
        """Returns a recursive deepcopy of all Nodes"""
        return Node(self.value, [child.copy() for child in self])

    #
    # Basic Operations
    #

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

    #
    # Advanced Operations
    #

    def __and__(self, other): return self * other
    def __rand__(self, other): return other * self
    def __or__(self, other): return self + other
    def __ror__(self, other): return other + self

    def __eq__(self, other): return Node.const(0) / (self - other)

    def __abs__(self): return (self * self) ** (Node.const(1) / Node.const(2))

    def __lt__(self, other): return (Node.const(1) - abs(self - other) / (self - other)) / Node.const(2)
    def __gt__(self, other): return (Node.const(1) - abs(other - self) / (other - self)) / Node.const(2)
    def __le__(self, other): return (abs(other - self) / (other - self) + Node.const(1)) / Node.const(2)
    def __ge__(self, other): return (abs(self - other) / (self - other) + Node.const(1)) / Node.const(2)

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
    # def __min__(self,other):

    # def if_else(self, cond, if_false):
    #     if if_false is None:
    #         return cond * self
    #     else:
    #         return cond * self + (1 - cond) * if_false

    # @staticmethod
    # def min():






def plot(node, x_linspace=(-10,10,21), algebraic=False):
  xs = np.linspace(*x_linspace)
  plt.scatter(xs, [node(i, algebraic=algebraic) for i in xs])
  plt.plot(xs, [node(i, algebraic=algebraic) for i in xs])
  plt.show()

def sign(g): return abs(g)/g



x = Node('x')
# y = Node('y')
y = sp.Symbol('y')

n = 2

g = x - n

# s = (g*g)**(const(1)/const(2))

# f = x < 2

# f = Node.min(x, 0)

# f = x*(0.5 - 0.5*(x**2)**0.5/x)

# f = Node.if_then(
#     ((0 <= x) & (x <= 2)) | (x > 7),
#     x,
#     x * x
# )

# f = Node.if_then(
#     x > 1,
#     x,
#     x * x
# )

# x*(0.5 + 0.5*(x**2)**0.5/x)
f = Node.if_then(
    x >= 0,
    x
)

# x*(0.5 + 0.5*(x**2)**0.5/x)
f = Node.max(0, x)

# f = Node.const()

# f = abs(x)

# g = x

# f = (x==-1)
# for i in range(2,9):
#     f = f + (x==-i)

# if __name__ == '__main__':
    # print(f)
    # print(f(sp.Symbol('x'), True))
    # plot(f, (-2,2))
    #
    #



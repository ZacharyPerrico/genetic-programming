import math
import numpy as np
import sympy as sp

class Node:
    """
    A basic class for genetic programming.
    A Node holds a single value and points to zero or more children Nodes.
    A graph of Nodes is used to represent a function in some form.
    """

    # All possible operations for a node and the number of children it must have
    # Used when randomly generating graphs
    valid_ops = {
        # Basic Operations
        'noop': 1,
        'neg': 1,
        '+': 2,
        '-': 2,
        '*': 2,
        '/': 2,
        '**': 2,
        'abs': 1,
        # Comparisons
        'eq': 2,
        '<': 2,
        '>': 2,
        '<=': 2,
        '>=': 2,
        'min': 2,
        'max': 2,
        'if_then_else': 3,
        # Trigonometry
        'sin': 1,
        'cos': 1,
        # Logarithms
        'exp': 1,
        'ln': 1,
        # Complex
        'real': 1,
        'imag': 1,
        # Bit
        '&': 2,
        '|': 2,
        '%': 2,
        'get_bits': 3,
    }

    #
    # Construction
    #

    def __init__(self, value, children=None):
        # self.parent = None
        self.parents = []
        # If the value is already a node use its value so that Nodes can be cast to a Node
        # This also allows for shallow copies of a Node to be made through casting
        if type(value) == Node:
            self.children = value.copy().children
            self.value = value.value
        else:
            self.value = value
            self.children = children if children is not None else []
        # Used when creating a list of all nodes to prevent repeats
        # None indicates that all children also have a temp_index of None
        # Setting this to -1 and then resetting results in it being None
        self.temp_index = -1
        # Previously returned value used for semantic analysis
        self.returned_value = None
        # If all descendants are in the simplest form
        self.is_limited = False

    #
    # Children and Parents
    #

    @property
    def children(self):
        return self._children

    @children.setter
    def children(self, children):
        """Setting a child also sets the parent of the child"""
        for child in children:
            # child.parent = self
            child.parents.append(self) #FIXME remove unused parents
        self._children = children

    def __len__(self): return len(self.children)
    def __getitem__(self, i): return self.children[i]
    def __setitem__(self, i, value): self.children[i] = value
    def __iter__(self): yield from self.children

    def reset_index(self):
        """Set the temp_index of all nodes to None"""
        if self.temp_index is not None:
            self.temp_index = None
            for child in self.children:
                child.reset_index()

    def index_in(self, l):
        """Returns the first index of this object in the given iterable. The `in` keyword and `index` method will not work for Nodes"""
        for i, node in enumerate(l):
            if node is self:
                return i
        return -1

    # def __eq__(self, other):


    def nodes(self, node_list=None):
        """Returns a list of all nodes"""
        if node_list is None:
            node_list = []
            self.reset_index()
        if self.temp_index is None:
            self.temp_index = len(node_list)
            node_list.append(self)
            for child in self:
                child.nodes(node_list)
        return node_list

    def reset_parents(self):
        """Remove all parent pointers from all nodes"""
        if len(self.parents) > 0:
            self.parents = []
            for child in self:
                child.reset_parents()

    def set_parents(self):
        """Append all parent pointers back to the nodes"""
        for child in self.children:
            child.parents.append(self)
            child.set_parents()

    #
    # Information
    #

    def root(self):
        """Returns the root Node of the graph"""
        return self if len(self.parents) == 0 else self.parents[0].root()

    def size(self):
        """Returns the number of nodes"""
        return len(self.nodes())

    def height(self):
        """Returns the longest distance to a leaf"""
        return max([0] + [1 + child.height() for child in self.children])

    def depth(self):
        """Returns the longest distance to the root"""
        return max([0] + [1 + parent.depth() for parent in self.parents])

    # def effective_code(self, a=None):
    #     """The effective code of the last evaluation"""
    #     init_call = a is None
    #     a = [] if init_call else a
    #     # Call recursively for each child
    #     for child in self:
    #         child.effective_code(a)
    #         a.append(np.linalg.norm(self.returned_value - child.returned_value))
    #     # Calculate the result from all semantic vectors
    #     if init_call:
    #         effective_code_value = np.sum(np.bool(a)) / (len(self.nodes()) - 1)
    #         effective_code_value = np.nan_to_num(effective_code_value, nan=0)
    #         return effective_code_value

    #
    # String Representation
    #

    def __str__(self):
        if len(self) == 0:
            return str(self.value)
        elif not self.value.isalpha():
            # Infixed operation
            return f'({self[0]}{self.value}{self[1]})'
        else:
            # Prefixed operation
            return self.value + '(' + ','.join([str(child) for child in self]) + ')'

    def __repr__(self):
        return str(self)

    def latex(self):
        try:
            s = sp.latex(self.simplify())
        except:
            s = str(self)
        return s

    #
    # Modification
    #

    def replace(self, new_node):
        """Replaces this node and all children with a new branch"""
        root = self.root()
        # Create a copy of the new node
        # new_node = new_node.copy()
        new_node = new_node
        # Return the new node if self is the root of the tree
        if len(self.parents) == 0:
            self.value = new_node.value
            self.children = new_node.children
        # Change all nodes pointing at this node to be pointing at the new node
        for parent in self.parents:
            if self in parent:
                # Parent's index for self
                self_index = parent.children.index(self)
                # self_index = self.index_in(parent)
                # Replace the parent's reference to self
                parent[self_index] = new_node
        # Recalculate all links_adj to parents
        # This is because the original structure may still point to descendants of the original
        root.reset_parents()
        root.set_parents()
        # Return the full new tree
        return root

    #
    # Conversion
    #

    def to_tree(self):
        """Returns a recursive deepcopy of all Nodes"""
        return Node(self.value, [child.to_tree() for child in self])

    def to_lists(self, verts=None, edges=None):
        """Returns lists representing the vertices and edges. Used for saving and advanced plotting"""
        # Initial call
        if verts is None:
            self.reset_index()
            verts, edges = [], []
        if self.temp_index is None:
            self.temp_index = len(verts)
            verts.append(self.value)
            for child in self.children:
                child.to_lists(verts, edges)
                edges.append((self.temp_index, child.temp_index))
        return verts, edges

    @staticmethod
    def from_lists(verts, edges):
        """Returns a Node tree from lists representing the vertices and edges"""
        nodes = [Node(vert) for vert in verts]
        for edge in edges:
            nodes[edge[0]]._children.append(nodes[edge[1]])
            nodes[edge[1]].parents.append(nodes[edge[0]])
        return nodes[0]

    def copy(self):
        return Node.from_lists(*self.to_lists())

    #
    # Evaluation
    #

    def reset_returned_value(self):
        """Set the returned_value of all nodes to None"""
        if self.returned_value is not None:
            self.returned_value = None
            for child in self.children:
                child.reset_returned_value()

    def __call__(self, *x, eval_method=None, **kwargs):
        """
        Calling evaluates the value of the entire graph.
        Input values can be numbers, ndarrays, or Sympy expressions.
        """

        match eval_method:

            # Returns x once f(x)==0 otherwise x:=f(x)
            case 'zero':
                if type(x[0]) == np.ndarray:
                    return_value = []
                    for i in range(len(x[0])):
                        xs = [xi[i] for xi in x]
                        return_value.append(self(*xs, eval_method=eval_method, **kwargs))
                    return return_value
                else:
                    # Only use the first parameter
                    return_value = x[0]
                    for _ in range(100):
                        new_return_value = self(return_value)
                        if new_return_value == 0:
                            return new_return_value
                        return_value = new_return_value
                    return return_value

                # return_values = []
                # for i in range(len(x[0])):
                #     xs = [xi[i] for xi in x]
                #     # return_value = self(*x)
                #     return_value = self(*xs)
                #     for _ in range(100):
                #         new_return_value = self(return_value)
                #         print(return_value)
                #         if new_return_value == 0:
                #             # return new_return_value
                #             return_values.append(new_return_value)
                #
                #         return_value = new_return_value
                #     # return return_value
                #     return_values.append(return_value)
                # return return_values

            # Evaluate x:=f(x) until even
            case 'even':
                return_value = self(*x)
                for _ in range(100):
                    return_value = self(return_value)
                    if return_value % 2 == 0: break
                return return_value // 2

            # Default evaluation
            case _:

                # Non strings are not operations and must have values extracted
                if type(self.value) is not str:
                    if isinstance(x[0], sp.Expr):
                        return_value = sp.Number(self.value)
                    else:
                        # A node with a number value should return in the shape of the inputs
                        return_value = self.value * np.ones_like(x[0])

                # Strings are matched to the operation they represent
                else:
                    match self.value:

                        # Basic Operations
                        case '+': return_value = self[0](*x, **kwargs) + self[1](*x, **kwargs)
                        case '-': return_value = self[0](*x, **kwargs) - self[1](*x, **kwargs)
                        case '*':
                            s0, s1 = self[0](*x, **kwargs), self[1](*x, **kwargs)
                            return_value = s0 * s1
                        case '/':
                            s0, s1 = self[0](*x, **kwargs), self[1](*x, **kwargs)
                            # Only return s0 / s1 for symbolic expressions
                            if isinstance(x[0], sp.Expr):
                                return_value =  s0 / s1
                            else:
                                return_value = np.ones_like(s0, 'complex')
                                return_value *= np.inf
                                ind = s1 != 0
                                np.true_divide(s0, s1, out=return_value, where=ind, dtype='complex')
                        case '**':
                            s0, s1 = self[0](*x, **kwargs), self[1](*x, **kwargs)
                            # Only return s0 ** s1 for symbolic expressions
                            if isinstance(x[0], sp.Expr):
                                return_value = s0 ** s1
                            else:
                                return_value = np.ones_like(s0, 'complex')
                                # Valid where s0 is not zero or s1 is a positive real number
                                ind = (s0 != 0) | (np.isreal(s1) & (np.real(s1) > 0))
                                np.power(s0, s1, out=return_value, where=ind, dtype='complex')
                        case 'noop': return_value = self[0](*x, **kwargs)
                        case 'neg': return_value = -self[0](*x, **kwargs)
                        case '%':  return_value = self[0](*x, **kwargs) % self[1](*x, **kwargs)
                        case 'abs': return_value = abs(self[0](*x, **kwargs))

                        # Comparisons
                        case '<': return_value = self[0](*x, **kwargs) < self[1](*x, **kwargs)
                        case '>': return_value = self[0](*x, **kwargs) > self[1](*x, **kwargs)
                        case '<=': return_value = self[0](*x, **kwargs) <= self[1](*x, **kwargs)
                        case '>=': return_value = self[0](*x, **kwargs) >= self[1](*x, **kwargs)
                        case 'eq': return_value = self[0](*x, **kwargs) == self[1](*x, **kwargs)
                        case 'min': return_value = min(self[0](*x, **kwargs), self[1](*x, **kwargs))
                        case 'max': return_value = max(self[0](*x, **kwargs), self[1](*x, **kwargs))
                        case 'if_then_else': return_value = self[1](*x, **kwargs) if self[0](*x, **kwargs) else self[2](*x, **kwargs)

                        # Trigonometry
                        case 'pi':
                            return_value = kwargs['pi'] if 'pi' in kwargs else np.pi * np.ones_like(x[0])
                        case 'sin':
                            s0 = self[0](*x, **kwargs)
                            if isinstance(x[0], sp.Expr):
                                return_value = sp.sin(s0)
                            else:
                                return_value = np.sin(s0)
                        case 'cos':
                            s0 = self[0](*x, **kwargs)
                            if isinstance(x[0], sp.Expr):
                                return_value = sp.cos(s0)
                            else:
                                return_value = np.cos(s0)

                        # Logarithms
                        case 'e':
                            return_value = kwargs['e'] if 'e' in kwargs else np.e * np.ones_like(x[0])
                        case 'exp':
                            s0 = self[0](*x, **kwargs)
                            if isinstance(x[0], sp.Expr):
                                return_value = sp.exp(s0)
                            else:
                                return_value = np.exp(s0)
                        case 'ln':
                            s0 = self[0](*x, **kwargs)
                            if isinstance(x[0], sp.Expr):
                                return_value = sp.ln(s0)
                            else:
                                return_value = np.log(s0)

                        # Complex
                        case 'i':
                            return_value = kwargs['i'] if 'i' in kwargs else 1j * np.ones_like(x[0])
                        case 'inf':
                            return_value = kwargs['inf'] if 'inf' in kwargs else np.inf * np.ones_like(x[0])
                        case 'real':
                            s0 = self[0](*x, **kwargs)
                            if isinstance(x[0], sp.Expr):
                                return_value = sp.re(s0)
                            else:
                                return_value = np.real(s0)
                        case 'imag':
                            s0 = self[0](*x, **kwargs)
                            if isinstance(x[0], sp.Expr):
                                return_value = sp.im(s0)
                            else:
                                return_value = np.imag(s0)

                        # Bit
                        case '|': return_value = self[0](*x, **kwargs) | self[1](*x, **kwargs)
                        case '&': return_value = self[0](*x, **kwargs) & self[1](*x, **kwargs)
                        case '>>': return_value = self[0](*x, **kwargs) >> self[1](*x, **kwargs)
                        case '<<': return_value = self[0](*x, **kwargs) << self[1](*x, **kwargs)
                        case 'get_bit': return_value = (int(self[0](*x, **kwargs)) >> self[1](*x, **kwargs)) & 1
                        case 'get_bits':
                            s0, s1, s2 = self[0](*x, **kwargs), self[1](*x, **kwargs), self[2](*x, **kwargs)
                            return_value = (np.int64(s0) >> np.int64(s1)) % np.int64(2.0 ** s2)

                        # Terminals and constants
                        case 'x': return_value = x[0]
                        case 'y': return_value = x[1]
                        case 'z': return_value = x[2]

                        # Arbitrary Variable
                        # Convert string to int then use as an index
                        # This is how 'x1', 'x2', ... are supported
                        case _: return_value = np.float64(x[int(''.join([s for s in self.value if s.isdigit()]))])

                # Store the last returned value for analysis if needed
                self.returned_value = return_value
                return return_value

    def simplify(self):
        """Returns a SymPy Expression representing the graph"""
        return sp.sympify(self(sp.Symbol('x'), sp.Symbol('y'), sp.Symbol('z'), e=sp.E, i=sp.I, pi=sp.pi))

    #
    # Construction
    # Easily create graphs using native Python operations and static Node methods
    # All implementations must use the op function as a basis
    #

    @staticmethod
    def op(operation, *operands):
        """Returns a new Node from an operation on other Nodes"""
        # Convert operands to a list to be modified
        operands = list(operands)
        # Cast each operand to a Node, operands must not be copied as pointers need to be preserved
        for i in range(len(operands)):
            if type(operands[i]) != Node:
                operands[i] = Node(operands[i])
        # Create a new Node with the operands as the children
        new_node = Node(operation, operands)
        return new_node

    def      __add__(self, other): return Node.op('+',  self, other)
    def     __radd__(self, other): return Node.op('+',  other, self)
    def      __sub__(self, other): return Node.op('-',  self, other)
    def     __rsub__(self, other): return Node.op('-',  other, self)
    def      __mul__(self, other): return Node.op('*',  self, other)
    def     __rmul__(self, other): return Node.op('*',  other, self)
    def  __truediv__(self, other): return Node.op('/',  self, other)
    def __rtruediv__(self, other): return Node.op('/',  other, self)
    def      __pow__(self, other): return Node.op('**', self, other)
    def     __rpow__(self, other): return Node.op('**', other, self)
    def      __neg__(self       ): return Node.op('neg',self       )
    def      __and__(self, other): return Node.op('&',  self, other)
    def     __rand__(self, other): return Node.op('&',  other, self)
    def       __or__(self, other): return Node.op('|',  self, other)
    def      __ror__(self, other): return Node.op('|',  other, self)
    def      __abs__(self       ): return Node.op('abs',self       )
    def       __lt__(self, other): return Node.op('<',  self, other)
    def       __gt__(self, other): return Node.op('>',  self, other)
    def       __le__(self, other): return Node.op('<=', self, other)
    def       __ge__(self, other): return Node.op('>=', self, other)
    def   __lshift__(self, other): return Node.op('<<', self, other)
    def   __rshift__(self, other): return Node.op('>>', self, other)
    def      __mod__(self, other): return Node.op('%',  self, other)

    @staticmethod
    def eq(*operands): return Node.op('eq', *operands)
    @staticmethod
    def max(*operands): return Node.op('max', *operands)
    @staticmethod
    def min(*operands): return Node.op('min', *operands)
    @staticmethod
    def sin(*operands): return Node.op('sin', *operands)
    @staticmethod
    def cos(*operands): return Node.op('cos', *operands)
    @staticmethod
    def get_bits(f, start, length): return Node.op('get_bits', f, start, length)
    @staticmethod
    def if_then_else(cond, if_true, if_false): return Node.op('if_then_else', cond, if_true, if_false)
    @staticmethod
    def noop(*operands): return Node.op('noop', *operands)
    @staticmethod
    def real(*operands): return Node.op('real', *operands)
    @staticmethod
    def imag(*operands): return Node.op('imag', *operands)
    @staticmethod
    def ln(*operands): return Node.op('ln', *operands)
    @staticmethod
    def exp(*operands): return Node.op('exp', *operands)

    #
    # Limited Equivalence
    # Convert a graph of various operation types into one of only basic operations
    #

    @staticmethod
    def const(n, defined=None):
        """A basic implementation to convert integers into limited trees"""
        if defined is None:
            defined = {'x': Node('x')}
        # Return the constant and define it to be used recursively
        if n == 0:
            if 0 not in defined:
                defined[0] = defined['x'] - defined['x']
            return defined[0]
        elif n == 1:
            if 1 not in defined:
                defined[1] = defined['x'] / defined['x']
            return defined[1]
        elif n == -1:
            if -1 not in defined:
                defined[-1] = Node.const(0, defined) - Node.const(1, defined)
            return defined[-1]
        elif n == 1j:
            if 1j not in defined:
                return Node.const(-1, defined) ** (Node.const(1, defined) / Node.const(2, defined))
            return defined[1j]
        elif np.iscomplex(n):
            return Node.const(np.real(n), defined) + Node.const(1j, defined) * Node.const(np.imag(n), defined)
        elif n < 0:
            return Node.const(-1, defined) * Node.const(-n, defined)

        elif np.log2(n).is_integer():
            if n not in defined:
                defined[n] = Node.const(n/2, defined) + Node.const(n/2, defined)
            return defined[n]

        else:
            return_value = None
            for i, bit in enumerate(bin(int(n))[:1:-1]):
                if bit == '1':
                    u = Node.const(2**i, defined)
                    if return_value is None:
                        return_value = u
                    else:
                        return_value = u + return_value
            return return_value

        # else:
        #     c = Node.const(1, defined)
        #     return sum([c for _ in range(int(n)-1)], c)




    def limited(self, consts=True, defined=None):
        """Returns the expression in terms of only the five basic operations"""
        # Store pointers to all previously used constants
        if defined is None:
            defined = {'x': Node('x')}
        if self.is_limited:
            return self
        elif type(self.value) is not str:
            if consts:
                # Return here to prevent recursive calls with return_value
                return_value = Node.const(self.value, defined)
                return_value.is_limited = True
                return return_value
            else:
                self.is_limited = True
                return self
        else:
            match self.value:
                case '+' | '-' | '*' | '/' | '**':
                    self.children = [child.limited(consts=consts, defined=defined) for child in self]
                    self.is_limited = True
                    return self
                case 'neg': return_value = 0 - self[0]
                case '|': return_value = self[0] ** 0 ** self[1]
                case '&': return_value = self[0] * self[1]
                case 'eq': return_value = 0 / (self[0] - self[1])
                case 'abs': return_value = (self[0] * self[0]) ** (Node(1) / 2)
                case '<': return_value = (1 - abs(self[0] - self[1]) / (self[0] - self[1])) / 2
                case '>': return_value = (1 - abs(self[1] - self[0]) / (self[1] - self[0])) / 2
                case '<=': return_value = ((abs(self[1] - self[0]) / (self[1] - self[0]) + 1) / 2)
                case '>=': return_value = ((abs(self[0] - self[1]) / (self[1] - self[0]) + 1) / 2)
                case '<<': return_value = (self[0] * 2 ** self[1])
                case '>>':
                    s1 = self[1].value
                    if s1 == 0:
                        return_value = self[0]
                    else:
                        rec = (self[0] >> s1-1)
                        return_value = ((rec - rec % 2) / 2)
                case '%':
                    if self[1].value == 1:
                        return_value = Node(0)
                    elif self[1].value == 2:
                        return_value = (1 - (-1) ** self[0]) / 2
                    else:
                        k = int(math.log2(self[1].value))
                        return_value = ((((self[0] >> k-1) % 2) << k-1) + (self[0] % 2**(k-1)))
                case 'sin':
                    e = Node('e')
                    i = Node('i')
                    return_value = (e ** (i * self[0]) - e ** (i * -self[0])) / (2 * i)
                case 'cos':
                    e = Node('e')
                    i = Node('i')
                    return_value = (e ** (i * self[0]) + e ** (i * -self[0])) / 2
                case 'get_bits':
                    return_value = ((self[0] >> self[1].value) % (2 ** self[2].value))
                case 'i':
                    return_value = Node.const(1j, defined)
                case 'exp':
                    e = Node('e')
                    return_value = e ** self[0]
                case _:
                    if self.value in defined:
                        return defined[self.value]
                    else:
                        return self
        # Recursively call limiting
        return_value = return_value.limited(consts=consts, defined=defined)
        self.replace(return_value)
        return_value.is_limited = True
        return return_value






if __name__ == '__main__':

    e = Node('e')
    i = Node('i')
    pi = Node('pi')
    x = Node('x')
    y = Node('y')
    z = Node('z')


    a = x
    b = y

    # b.replace(a)

    b = Node.const(12)

    print(b)

    f = 2 * x

    # p = x in [0,x,1]

    # p = x.index_in(f)


    f = Node.sin(x) + Node.cos(x)

    # f = x << 2

    f = f.limited(consts=not True)

    print(f)

    # a = [x,y]
    #
    # q = z in a

    # print(q)

    # f = (Node.ln((1+2))+(((0+((1*0)+(((1-0)/(1+2))/(((1-0)/(1+2))+Node.exp(2)))))+(2/1))/(Node.exp(2)/0)))

    # print(p)
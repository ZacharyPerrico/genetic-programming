#@title Default kwargs
from gp import *
from utils import *

# Default kwargs
kwargs = {
  # The random seed to use for random and Numpy.random
  'seed': 1,

  # Initial max depth of a tree
  'tree_depth': 4,

  # Population size
  'M': 600,

  # Number of randomly chosen parents for each tournanet
  'k': 4,

  # Probability of crossover
  'p_c': 0.9,

  'max_crossover_depth': 4,

  'max_depth': 400,

  # Function used to create next generation
  'crossover_func': crossover,

  # Probability of a bit mutating
  'p_m': 0.5,

  # Function used to create next generation
  'mutate_func': subtree_mutation,

  # Generations before timeout
  'T_max': 1,

  # Print changes and updates
  'verbose': 1,

  # Simplify algebraicly before evaluating
  'algebraic': False,

  # The domain of the problem expressed using np.linspace
  'x_linspace': (-1,1,21),

  # The valid leaves of the tree
  'leaves': ['x'],
}



#@title Types of Leaves
kwargs['label_title'] = 'Types of Leaves'
kwargs['key'] = 'leaves'
kwargs['labels'] = ['$x$ and -5 to 5', '$x$ only']
kwargs['values'] = [['x',-5,-4,-3,-2,-1,0,1,2,3,4,5], ['x']]

# Run simulation
all_pops, all_fits = run_sims(**kwargs)

np.save('saves/fits', all_fits)
np.save('saves/pops', all_pops)
np.save('saves/kwargs', np.array([kwargs]))

plot_sims(all_pops, all_fits, **kwargs)

from dwave_bm.rbm import BM

# Model dimensions.
num_visible = 10

# the hidden layer is a bipartite graph of 9 by 5
num_hidden = [9, 5] 

# Create the Boltzmann machine object
r = BM(num_visible=num_visible, num_hidden=num_hidden)
r

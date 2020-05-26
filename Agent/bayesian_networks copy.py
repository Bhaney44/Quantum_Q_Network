
# There are two types of graphs:

# (1) Directed graph provides a compact representation of the probability distribution, where the causal relationships are explicitly expressed by directed edges.
# A special category of the directed graph, namely the directed acylic graphs (DAGs), is commonly known as Bayesian networks.
# The idea behind Bayesian networks is simple: folllowing the chain rule of expressing a joint probability distribution as factorized conditional probability distributions.
# the conditional probability can be greatly simplified by only considering the parent nodes.
# Variables that have conditional independencies can be effectively represented by Bayesian networks.
# Learning a Bayesian network from complete data can be achieved by maximizing the likelihood function.

# (2) Different from the directed graph, undirected graph provides a less compact representation of the probability distribution by assuming no obvious independencies among the variables.
# Undirected graph incurs more parameters in the model, which makes it more difficult to learn and infer.
# A special category of the undirected graph is Markov networks by assuming the Markovian properties.
# Both learning and inference of a Markov network require knowledge of the partition function. In general case, this is NP-hard.
# Obviously, Markov networks are more powerful than Bayesian networks.
# With techniques called moralization, we can convert Bayesian networks to Markov networks.
# Although this doesn't solve the partition function nightmare, it provides a very interesting middleground between the two.
# Our claim is based on the availablity of a D-Wave quantum computer.
# Training of a Markov network becomes relatively easy with D-Wave quantum computer.
# Note that the D-Wave quantum computer exhibits as an undirected graphical model.
# In this report, we will study how to migrate the probability distribution represented by a Bayesian network to that of a Markov network powered by quantum computer.
# We expect interesting stories to happen when Bayesian networks meet with Markov networks.


# Moralization of Bayesian Networks to Markov Networks
# Converting a Bayesian network to Markov network requires a method called moralization, which is composed of two steps:

# (1) Marrying all the parents
# (2) Dropping all the directionalities of the edges

# Obviously, the independency constraints among the parents in a Bayesian network is ruined by adding edges among them.
# Moralization is the first step to a junction tree algorithm, which is a method to marginalize a general graph.
# Goodness of the conversion
# In general, the conversion from a Bayesian network to a Markov network is not lossless, which means the conditional independencies (CIs) might be lost.

# Experiment


from __future__ import print_function

import time
import itertools

from operator import mul

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import tensorflow as tf
import dimod

from pgmpy.models import BayesianModel, MarkovModel
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
from pgmpy.sampling import BayesianModelSampling
from pgmpy.factors.discrete import TabularCPD
from pgmpy.extern.six.moves import reduce
from pgmpy.inference import VariableElimination
from sympy import symbols, Poly
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite

from helper import bit2int, calculate_histogram, kld
get_ipython().run_line_magic('matplotlib', 'inline')



# Define the model
grass_model = BayesianModel()
NODES = ['Cloudy', 'Rain', 'Sprinkler', 'Wet']
EDGES = [(0, 1), (0, 2), (1, 3), (2, 3)]
# Add nodes to graph
grass_model.add_nodes_from(NODES)

# Add edges to graph
for edge in EDGES:
    grass_model.add_edge(NODES[edge[0]], NODES[edge[1]])

# Moralize the Bayesian network
moral_graph = grass_model.moralize()
temp = list(moral_graph.edges)
MORAL_EDGES = [(NODES.index(v[0]), NODES.index(v[1])) if  NODES.index(v[1]) > NODES.index(v[0]) 
                                              else (NODES.index(v[1]), NODES.index(v[0])) for v in temp]
MORAL_EDGES.sort()
print("edges after moralization", MORAL_EDGES)

# Visualize the networks
plt.subplot(121)
plt.title('Bayesian network')
nx.draw(grass_model, with_labels=True)
plt.subplot(122)
plt.title('Markovian network')
nx.draw(moral_graph, with_labels=True)

print(grass_model.nodes)

# Define the conditional probability distributions (CPDs).

cpds = []
cpds.append(TabularCPD(variable=NODES[0], variable_card=2,
                       values=[[0.5], [0.5]]))

cpds.append(TabularCPD(variable=NODES[1], variable_card=2,
                       values=[[0.8, 0.2], [0.2, 0.8]],
                       evidence=[NODES[0]], evidence_card=[2]))

cpds.append(TabularCPD(variable=NODES[2], variable_card=2,
                       values=[[0.5, 0.9], [0.5, 0.1]],
                       evidence=[NODES[0]], evidence_card=[2]))

cpds.append(TabularCPD(variable=NODES[3], variable_card=2,
                       values=[[1.0, 0.1, 0.1, 0.01],
                               [0.0, 0.9, 0.9, 0.99]],
                       evidence=[NODES[1], NODES[2]],                    
                       evidence_card=[2, 2]))

# Associating the parameters with the model structure.
for cpd in cpds:
    grass_model.add_cpds(cpd)

# Checking if the cpds are valid for the model.
grass_model.check_model()
grass_model.get_cpds()


# Inference with BN
# Now that we have represented the BN with a complete JPD of all variables, it is theoretically possible to answer any query of certain variable(s) by marginalizing all irrelevant variables.
# This procedure is called **inference**.
# In general, a variable elimination method is employed to make use of the CPDs. 


# Do exact inference using Variable Elimination
grass_infer = VariableElimination(grass_model)

# Computing the probability of cloudy, sprinkler and rain given evidence of wet grass.
q = grass_infer.query(variables=NODES[:-1], evidence={NODES[-1]: 1})
print('Inference with Evidence of Wet=True')
for node in NODES[:-1]:
    print(node, '\n', q[node])


# Migrate distribution from  BN to MN

# Chimera-structured Boltzmann machine.

# Represent a Bayesian network with joint probability
# Moralize Bayesian network to Markov network
# Define the two features as binary quadratic functions



# Determine parameters of MN

# If we consider all the equations together, we end up solving 16 equations for 11 parameters (2 constants included), which gives us an overdetermined system.
# For overdetermined system, the exact solution may not exist.
# Usually, we would have to settle down for approximate solutions. 

# Method (1) Least-square solution.
# Method (2) explicitly build in Chimera structure and explore the sampling capacity of D-Wave quantum chip.

# * KLD solution

# A natural choice of objective function will be the Kullback-Leibler (KL) divergence of two probability distribution $p(\mathbf{x})$ and $q(\mathbf{x})$, i.e., $KL(p(\mathbf{x})||q(\mathbf{x}))$.
# This should be straightforward using the standard Boltzmann machine training since the
# Bayesian network also allows us to evaluate the expected $\phi(\mathbf{x})$ under $p(\mathbf{x})$ tractably.
# The only issue left is the negative phase in training Boltzmann machine, which requires evaluation of partition function.
# We can circumvent this requirement by using either Monte-Carlo Markov Chain (MCMC) method like contrastive divergence (CD) or persistent CD (PCD),
# or quantum hardware to draw samples with quantum mechanics.

# We can also find the optimal solution by minimizing the KL-divergence.
# Obtain the joint probability distribution (JPD) from conditional probability distributiom.


# calculate the true JPD fron CPDs
factors = [cpd.to_factor() for cpd in grass_model.get_cpds()]
factor_prod = reduce(mul, factors)
p_true = factor_prod.values.flatten().astype('float32')
print(factor_prod)


#------------------------------------
# Method (1) Least-squares solution
#------------------------------------

#optimization function

def optimize_lsq(p, nodes):

    # Define phi
    x1, x2, x3, x4 = symbols('x1, x2, x3, x4')
    h1, h2, h3, h4, J12, J13, J23, J24, J34, a, b = symbols('h1, h2, h3, h4, J12, J13, J23, J24, J34, a, b')

    phi = x1 * h1 + x2 * h2 + x3 * h3 + x4 * h4 + J12 * x1 * x2 + J13 * x1 * x3 + J23 * x2 * x3 + J24 * x2 * x4 + J34 * x3 * x4 + a + b
    phi1 = x1 * h1 + x2 * h2 + x3 * h3 + J12 * x1 * x2 + J13 * x1 * x3 + J23 * x2 * x3 + a
    phi2 = x2 * h2 + x3 * h3 + x4 * h4 + J23 * x2 * x3 + J24 * x2 * x4 + J34 * x3 * x4 + b
    coeff = Poly(phi, (h1, h2, h3, h4, J12, J13, J23, J24, J34, a, b)).coeffs()
    coeff1 = Poly(phi1, (h1, h2, h3, J12, J13, J23, a)).coeffs()
    coeff2 = Poly(phi2, (h2, h3, h4, J23, J24, J34, b)).coeffs()

    A = np.zeros(shape=(2 ** len(nodes), 11))
    A1 = np.zeros(shape=(2 ** len(nodes), 11))

    for i, v in enumerate(list(itertools.product([0, 1], repeat=len(nodes)))):
        A[i] = [ii.subs({x1: v[0], x2: v[1], x3: v[2], x4: v[3]}) for ii in coeff]

    for i, v in enumerate(list(itertools.product([0, 1], repeat=len(nodes) - 1))):
        A1[i, [0, 1, 2, 4, 5, 6, 9]] = [ii.subs({x1: v[0], x2: v[1], x3: v[2]}) for ii in coeff1]
        A1[i + 8, [1, 2, 3, 6, 7, 8, 10]] = [ii.subs({x2: v[0], x3: v[1], x4: v[2]}) for ii in coeff2]

    for i, v in enumerate(list(itertools.product([0, 1], repeat=len(nodes) - 1))):
        A1[i, [0, 1, 2, 4, 5, 6, 9]] = [ii.subs({x1: v[0], x2: v[1], x3: v[2]}) for ii in coeff1]
        A1[i + 8, [1, 2, 3, 6, 7, 8, 10]] = [ii.subs({x2: v[0], x3: v[1], x4: v[2]}) for ii in coeff2]

    p_true_view = p[[bit2int(v) for v in A1[:, :4]]]
    B = np.array(A[:, :9])

    ## CONST_IN_ISING = False
    
    ## if not CONST_IN_ISING:
    ##     A = A[:, :-1]
    ##     plt.suptitle('No constant in QUBO')
    ## solve y = A*x
    
    Z_lsq = 1.0
    y = -np.log(Z_lsq * (p_true_view + 1e-20))
    x = np.linalg.lstsq(A1, y, rcond=None)
    hJ = x[0][:9]
    
    ## Visualize the solution 
    plt.bar(np.array(range(len(y))), y, width=0.3, label = 'true y')
    plt.bar(np.array(range(len(y))) + 0.3, np.dot(A1, x[0]), width=0.3, label='lsq-fit')    
    plt.xlabel('x')
    plt.ylabel('probability')
    plt.legend()
    print('Z_lsq', np.sum(np.exp(-np.dot(A, x[0]))))

    return hJ



hJ_lsq = optimize_lsq(p_true, NODES)



# Define the solver
NUM_READS = 5000
NUM_EPOCHS = 100
lrate = 0.2
DW_PARAMS = {'auto_scale': True,
             'num_spin_reversal_transforms': 5
             }

# define sampler
dwave_sampler = DWaveSampler(solver={'qpu': True})

# Some accounts need to replace this line with the next:
# dwave_sampler = DWaveSampler(token = 'My API Token', solver='Solver Name')
sa_sampler = dimod.SimulatedAnnealingSampler()
emb_sampler = EmbeddingComposite(dwave_sampler)

emb_sampler.parameters


T = 1./20
Q_lsq = dict(((key, key), T * value) for (key, value) in zip(NODES, hJ_lsq[:len(NODES)]))
Q_lsq.update(dict(((NODES[key[0]], NODES[key[1]]), T * value) for (key, value) in zip(MORAL_EDGES, hJ_lsq[len(NODES):])))

print('Solving Q_LSQ on QPU...')
response_lsq = emb_sampler.sample_qubo(Q_lsq, num_reads=NUM_READS, **DW_PARAMS)

samples_lsq = np.asarray(
               [[datum.sample[v] for v in NODES] for datum in response_lsq.data() for __ in range(datum.num_occurrences)])

p_samp_lsq = calculate_histogram(samples_lsq)


plt.bar(np.array(range(2 ** len(NODES))) - 0.4, p_true, width=0.4, label='true')
plt.bar(np.array(range(2 ** len(NODES))), p_samp_lsq, width=0.4,
        label='LSQ, kld:' + str("%5.4f" % kld(p_true, p_samp_lsq)))
plt.xticks(range(2 ** len(NODES)), range(2 ** len(NODES)))
plt.ylim([0, 0.5])
plt.legend()
plt.xlabel('x in lexicographic order')
plt.ylabel('probability')


# Without surprise, the least-square method provides a __bad__ solution.
# Next, let us define another objective function, _i.e._ the KL divergence.


#------------------------------------
# Method 2: Training Markov Network
#------------------------------------
# Generate samples from Bayesian network

bn_sampler = BayesianModelSampling(grass_model)
bn_sampler.topological_order = NODES # make sure the topological oder is consistent with NODES order
kld_temp = 10.
i = 1

while kld_temp > 0.001:
    print('Iteration %d, kld %f'%(i, kld_temp))
    i += 1
    bn_samps = bn_sampler.forward_sample(size=NUM_READS, return_type='dataframe')
    # calculate true data stats
    data_stats = np.zeros(shape=(len(NODES) + len(MORAL_EDGES),))
    np.copyto(data_stats[:len(NODES)], np.mean(bn_samps, axis=0))
    np.copyto(data_stats[len(NODES):],
              np.dot(bn_samps.T, bn_samps)[np.array(MORAL_EDGES)[:, 0], np.array(MORAL_EDGES)[:, 1]] / (NUM_READS * 1.))
    p_data_bn = calculate_histogram(bn_samps.as_matrix())
    kld_temp = kld(p_true, p_data_bn)

plt.bar(range(16), p_true, width=0.4)
plt.bar(np.array(range(16))+0.4, p_data_bn, width=0.4)
print('KLD from true distribution to generated data distribution:', kld_temp)

#------------------------------------
# Method (2.1) Training MN on QC
#------------------------------------

# Function for training MN using QC

def optimize_samp(sampler, data_train, p, nodes, edges, epochs=1000, lrate=0.01):
    """
    Optimize kld using sampling method
    :return:
    """

    # Initialization of variables
    hJ_dw = np.random.normal(0, 0.001, size=(len(nodes) + len(edges),))
    
    ## hJ_dw = np.array([ 0.80723454, 3.49884406,  2.39950454,  3.39063915,
    ##                    -2.89170106,  2.33961665,  3.29197739, -5.67770571, -5.48619494,])
    
    grad_hJ_dw = np.zeros(shape=(len(nodes) + len(edges),))
    h_dw = hJ_dw[:len(nodes)]
    grad_h_dw = grad_hJ_dw[:len(nodes)]
    J_dw = hJ_dw[len(nodes):]
    grad_J_dw = grad_hJ_dw[len(nodes):]

    kld_curve = []

    T = 1. / 14
    NUM_READS_TRAIN = 1000
    TRAIN_SIZE = len(data_train)
    MINIBATCH_SIZE = TRAIN_SIZE
    t0 = time.clock()
    for i in range(epochs):
        np.random.shuffle(data_train)
        for mb in np.array_split(data_train, -(-TRAIN_SIZE//MINIBATCH_SIZE)):
            print('>', end='')
            Q = dict(((key, key), T * value) for (key, value) in zip(nodes, h_dw))
            Q.update(dict(((nodes[key[0]], nodes[key[1]]), T * value) for (key, value) in zip(edges, J_dw)))
            res = sampler.sample_qubo(Q, num_reads=NUM_READS_TRAIN, **DW_PARAMS)
            samps = np.asarray(
               [[datum.sample[v] for v in nodes] for datum in res.data() for __ in range(datum.num_occurrences)])

            # Calculate gradient
            np.copyto(grad_h_dw, np.mean(mb, axis=0) - np.mean(samps, axis=0))
            np.copyto(grad_J_dw,
                      np.dot(mb.T, mb)[np.array(edges)[:, 0], np.array(edges)[:, 1]]/(len(mb) * 1.) -
                      np.dot(samps.T, samps)[np.array(edges)[:, 0], np.array(edges)[:, 1]]/(NUM_READS_TRAIN * 1.))

            # Update params - SGD
            hJ_dw -= lrate * grad_hJ_dw

        p_hist_model = calculate_histogram(samps)
        kl = kld(p, p_hist_model)

        kld_curve.append(kl)

        # if i % 10 == 0:
        print('epoch: %d, kld: %f, lrate: %f, time: %f' % (i, kl, lrate, time.clock() - t0))

    print('learnt hJ on DW:', hJ_dw)
    return (hJ_dw, kld_curve)


# Train Markov networks on DW
hJ_dw, kld_dw = optimize_samp(emb_sampler, bn_samps.as_matrix(), p_data_bn, NODES, MORAL_EDGES, epochs=NUM_EPOCHS, lrate=lrate)


plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.plot(kld_dw)
plt.xlabel('Epoch')
plt.ylabel('KL Divergence')
T = 1./13
Q_dw = dict(((key, key), T * value) for (key, value) in zip(NODES, hJ_dw[:len(NODES)]))
Q_dw.update(dict(((NODES[key[0]], NODES[key[1]]), T * value) for (key, value) in zip(MORAL_EDGES, hJ_dw[len(NODES):])))

print('Solving Q_DW on QPU...')


response_dw = emb_sampler.sample_qubo(Q_dw, num_reads=NUM_READS, **DW_PARAMS)

samples_dw = np.asarray([[datum.sample[v] for v in NODES] for datum in response_dw.data() for __ in range(datum.num_occurrences)])

p_samples_dw = calculate_histogram(samples_dw)

plt.subplot(122)
plt.bar(np.array(range(2 ** len(NODES))) -.4, p_true, width=0.4, label='true')
plt.bar(np.array(range(2 ** len(NODES))), p_samples_dw, width=0.4,
        label='DW, kld:' + str("%5.4f" % kld(p_true, p_samples_dw)))
plt.xticks(range(2 ** len(NODES)), range(2 ** len(NODES)))
plt.ylim([0, 0.5])
plt.legend()
plt.xlabel('x in lexicographic order')
plt.ylabel('Probability')

#------------------------------------
# Method (2.2) Training MN with Tensorflow
#------------------------------------

# Define function to optimize KLD on tf
# Reset tf graph to make sure you want to run this cell multiple times

tf.reset_default_graph()

def optimize_tf(p, nodes, edges, epochs=1000, lrate=0.01):
    """
    Optimize KLD using tensorflow
    :param p:
    :param nodes:
    :param edges:
    :param epochs:
    :param lrate:
    :return:
    """

    def kld_tf(p, q):
        """
        Calculate KLD (p||q)
        :param p: true probability
        :param q: fitted probability
        :return: KLD value
        """
        cross_entropy = -tf.reduce_sum(p * tf.log(q + 1e-10))
        entropy = -tf.reduce_sum(p * tf.log(p + 1e-10))
        return cross_entropy - entropy

    num_nodes = len(nodes)
    num_edges = len(edges)
    # Enumerate all x's [0,...0,] to [1,...1]
    all_x = np.array(list(itertools.product([0, 1], repeat=num_nodes))).astype('float32')

    # Seperate edges into left part and right part
    e_left = [v[0] for v in edges]
    e_right = [v[1] for v in edges]
    all_x1 = all_x[:, e_left]
    all_x2 = all_x[:, e_right]

    h = tf.get_variable(name='h', shape=(num_nodes))
    J = tf.get_variable(name='J', shape=(num_edges))
    E = tf.reduce_sum(tf.multiply(h, all_x), axis=1) + tf.reduce_sum(tf.multiply(tf.multiply(all_x1, all_x2), J),
                                                                     axis=1)
    # exp of -E
    expE = tf.exp(-E, name='expE')

    # partition function
    Z = tf.reduce_sum(expE)

    q = tf.div(expE, tf.reduce_sum(expE))
    # objective function
    kl_divergence = kld_tf(p, q)

    # optimizer = tf.train.GradientDescentOptimizer(0.01)
    optimizer = tf.train.AdamOptimizer(lrate)
    train = optimizer.minimize(kl_divergence)
    init = tf.global_variables_initializer()

    klds = []

    with tf.Session() as session:
        session.run(init)
        print("E", session.run(expE), session.run(q))
        print("starting at", "\nh:", session.run(h), "\nJ:", session.run(J), "\nkld:", session.run(kl_divergence))
        for step in range(epochs):
            session.run(train)
            klds.append(session.run(kl_divergence))
            if (step + 1) % 1000 == 0:
                print("step", step, "h:", session.run(h), "kld:", session.run(kl_divergence))

        print("finished at", "\nh:", session.run(h), "\nJ:", session.run(J), "\nkld:", session.run(kl_divergence))
        print("Z", session.run(Z))

        hh = session.run(h)
        JJ = session.run(J)
        kl = session.run(kl_divergence)
        ZZ = session.run(Z)
    return (hh, JJ, klds, ZZ)



# Optimize with tensorflow
with tf.variable_scope('BN', reuse=tf.AUTO_REUSE):
    h, J, kld_tf, Z = optimize_tf(p_true, NODES, MORAL_EDGES, epochs=NUM_EPOCHS, lrate=lrate)
hJ_tf = np.hstack((h, J))

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.plot(kld_tf)
plt.xlabel('Epoch')
plt.ylabel('KL Divergence')
T = 1./13
Q_tf = dict(((key, key), T * value) for (key, value) in zip(NODES, hJ_tf[:len(NODES)]))
Q_tf.update(dict(((NODES[key[0]], NODES[key[1]]), T * value) for (key, value) in zip(MORAL_EDGES, hJ_tf[len(NODES):])))

print('Solving Q_TF on QPU...')
response_tf = emb_sampler.sample_qubo(Q_dw, num_reads=NUM_READS, **DW_PARAMS)
#Samples_tf = np.array([[samp[k] for k in NODES] for samp in response_tf])

samples_tf = np.asarray([[datum.sample[v] for v in NODES] for datum in response_tf.data() for __ in range(datum.num_occurrences)])

p_samp_tf = calculate_histogram(samples_tf)

plt.subplot(122)
plt.bar(np.array(range(2 ** len(NODES))) - 0.4, p_true, width=0.4, label='true')
plt.bar(np.array(range(2 ** len(NODES))), p_samp_tf, width=0.4,
        label='TF, kld:' + str("%5.4f" % kld(p_true, p_samp_tf)))
plt.xticks(range(2 ** len(NODES)), range(2 ** len(NODES)))
plt.ylim([0, 0.5])
plt.legend()
plt.xlabel('x in lexicographic order')
plt.ylabel('Probability')


# Learning BN from data
# In this section, we will take the samples from QPU and learn a Bayesian network with it.


def learn_BN(bn, data):
    """
    learn bayesian network from data
    :param bn:
    :param data:
    :return:
    """
    bn_copy = bn.copy()

    # Learning BN from data
    bn.fit(data, estimator=MaximumLikelihoodEstimator)

    return bn_copy


grass_model_copy = learn_BN(grass_model, bn_samps)
for cpd in grass_model_copy.get_cpds():
    print("CPD of {variable}:".format(variable=cpd.variable))
    print(cpd)

# JPD
# calculate the true JPD fron CPDs
factors = [cpd.to_factor() for cpd in grass_model_copy.get_cpds()]
factor_prod = reduce(mul, factors)
p_data = factor_prod.values.flatten().astype('float32')
p_data

# visualize the learned jpd vs true jpd
plt.bar(np.array(range(2**len(NODES)))-0.3, p_true, width=0.3, label='true')
plt.bar(np.array(range(2**len(NODES))), p_data, width=0.3, label='data')
plt.title('KLD: %52.f' % (kld(p_true, p_data)))
plt.legend()
plt.xlabel('x in lexicographic order')
plt.ylabel('Probability')


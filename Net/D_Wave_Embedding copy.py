#-----------------------------
#Exact Solver
#-----------------------------
import dimod

#Use the exact solver to find energies for all states. 
exactsolver = dimod.ExactSolver()

# Set up the QUBO. 
# These have the embedding explicitly expressed, with a variable for chainstrength.

#Chainstrength is a variable embedded in the matrix
#Chainstrength usually 1.5 times highest absolute value in matrix

chainstrength = 1

#Q-Matrix

# -1 + q_0 + q_4 - 2 q_0 q_4
# 2 q_1 q_4 - q_4 - 0.5 (q_1 + q_5)
# -1 + q_0 + 0.5 (q_1 + q_5) - 2 q_0 q_5
# c (-1 + q_1 + q_5 - 2q_1 q_5)

Q = {(0, 0): 2, (1, 1): chainstrength, (5, 5): chainstrength, (0, 4): -2, (1, 4): 2, (0, 5): -2, (1, 5): -2 * chainstrength}

# There's no need for a constant, so use exactsolver directly.
results = exactsolver.sample_qubo(Q)

#Print the results
for smpl, energy in results.data(['sample', 'energy']):
    print(smpl, energy)

#---------------------------
#QPU Embedding
#---------------------------

from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite

# Set up the QUBO. Start with the equations from the slides:
# x + y - 2xy -1
# 2yz - y - z
# -2zx + z + x - 1
# QUBO: 2x - 2xy + 2yz - 2zx - 2

chainstrength = 5
numruns = 100
Q = {(0, 0): 2, (0, 1): -2, (0, 2): -2, (1, 2): 2}

response = EmbeddingComposite(DWaveSampler()).sample_qubo(Q, chain_strength=chainstrength, num_reads=numruns)

R = iter(response)
E = iter(response.data())
for line in response:
    sample = next(R)
    print(sample, str(next(E).energy))


#---------------------
#miner QPU
#--------------------

from minorminer import find_embedding
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import FixedEmbeddingComposite

# Set up the QUBO. 
# Equations:
    # x + y - 2xy -1
    # 2yz - y - z
    # -2zx + z + x - 1
    # QUBO: 2x - 2xy + 2yz - 2zx - 2
    
# Q-matrix
Q = {(0, 0): 2, (0, 1): -2, (0, 2): -2, (1, 2): 2}

# Chainstrength
chainstrength = 5

#Number of runs
numruns = 100

dwave_sampler = DWaveSampler()
A = dwave_sampler.edgelist
embedding = find_embedding(Q, A)
print(embedding)

response = FixedEmbeddingComposite(DWaveSampler(), embedding).sample_qubo(Q, chain_strength=chainstrength, num_reads=numruns)

R = iter(response)
E = iter(response.data())
for line in response:
    sample = next(R)
    print(sample, str(next(E).energy))




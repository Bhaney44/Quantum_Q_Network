#Set list of numbers

S = [17, 21, 19]

#lagrange parameter
g = 4

#Set up QUBO dictionary

Q={}
Q[(0,0)]= S[0]-3*g
Q[(0,1)]= 2*g
Q[(0,2)]= 2*g

Q[(1,1)]= 2*g
Q[(1,2)]= S[1]-3*g

Q[(2,2)]=S[2]-3*g


#Print Dictionary
#print(Q)

#Set up QPU parameters

chainstrength = 20

#Number of returns
numruns = 10

#Run the QUBO on the solver from your config file

from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite

response = EmbeddingComposite(DWaveSampler()).sample_qubo(Q, chain_strength=chainstrength, num_reads=numruns)

#Return Results
R = iter(response)
E = iter(response.data())

print(response)

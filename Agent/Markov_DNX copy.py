#Markov Networks

import dimod

from dwave_networkx.utils import binary_quadratic_model_sampler

__all__ = 'sample_markov_network', 'markov_network_bqm'

# Samples from a markov network using the provided sampler.

def sample_markov_network(MN, sampler=None, fixed_variables=None,
                          return_sampleset=False,
                          **sampler_args):


    bqm = markov_network_bqm(MN)

    # use the FixedVar
    fv_sampler = dimod.FixedVariableComposite(sampler)

    sampleset = fv_sampler.sample(bqm, fixed_variables=fixed_variables,
                                  **sampler_args)

    if return_sampleset:
        return sampleset
    else:
        return list(map(dict, sampleset.samples()))


# Construct a binary quadratic model for a markov network.

def markov_network_bqm(MN):
    """Construct a binary quadratic model for a markov network.
    Parameters
    ----------
    G : NetworkX graph
        A Markov Network as returned by :func:`.markov_network`
    Returns
    -------
    bqm : :obj:`dimod.BinaryQuadraticModel`
        A binary quadratic model.
    """

    bqm = dimod.BinaryQuadraticModel.empty(dimod.BINARY)

    # the variable potentials
    for v, ddict in MN.nodes(data=True, default=None):
        potential = ddict.get('potential', None)

        if potential is None:
            continue

        # for single nodes we don't need to worry about order

        phi0 = potential[(0,)]
        phi1 = potential[(1,)]

        bqm.add_variable(v, phi1 - phi0)
        bqm.add_offset(phi0)

    # the interaction potentials
    for u, v, ddict in MN.edges(data=True, default=None):
        potential = ddict.get('potential', None)

        if potential is None:
            continue

        # in python<=3.5 the edge order might not be consistent so we use the
        # one that was stored
        order = ddict['order']
        u, v = order

        phi00 = potential[(0, 0)]
        phi01 = potential[(0, 1)]
        phi10 = potential[(1, 0)]
        phi11 = potential[(1, 1)]

        bqm.add_variable(u, phi10 - phi00)
        bqm.add_variable(v, phi01 - phi00)
        bqm.add_interaction(u, v, phi11 - phi10 - phi01 + phi00)
        bqm.add_offset(phi00)

    return bqm

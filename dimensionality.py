# Taku Ito
# Module for computing dimensionality

# 07/31/2017

import numpy as np
import sys
import loadNetworks as ln


# Using final partition
networkdef, networkmappings, networkorder = ln.loadNetworks()
networks = networkmappings.keys()

xticks = {}
reorderednetworkaffil = networkdef[networkorder]
for net in networks:
    netNum = networkmappings[net]
    netind = np.where(reorderednetworkaffil==netNum)[0]
    tick = np.max(netind)
    xticks[tick] = net

sortednets = np.sort(xticks.keys())
orderednetworks = []
for net in sortednets: orderednetworks.append(xticks[net])


def runConnectivityDimensionality(data):
    
    fcmat = np.corrcoef(data)
    data = fcmat
#     data = computeOutOfNetworkSimilarity((fcmat,networkdef))
    
    # Run PCA on out-of-network connectivity separately
    inputs = []
    net_dimensionality = np.zeros((len(orderednetworks),))
    netcount = 0
    for net in orderednetworks:
        net_ind = np.where(networkdef==networkmappings[net])[0]
        net_ind.shape = (len(net_ind),1)       
        out_ind = np.where(networkdef!=networkmappings[net])[0]
        out_ind.shape = (len(out_ind),1)
        
        netdata = np.squeeze(data[net_ind,out_ind.T])
        netdata = np.cov(netdata)
        
        net_dimensionality[netcount] = getDimensionality(netdata)
        netcount += 1
            
    return net_dimensionality
    
def getDimensionality(data):
    """
    data needs to be a square, symmetric matrix
    """
    corrmat = data
    eigenvalues, eigenvectors = np.linalg.eig(corrmat)
    dimensionality_nom = 0
    dimensionality_denom = 0
    for eig in eigenvalues:
        dimensionality_nom += np.real(eig)
        dimensionality_denom += np.real(eig)**2
    
    dimensionality = dimensionality_nom**2/dimensionality_denom

    return dimensionality


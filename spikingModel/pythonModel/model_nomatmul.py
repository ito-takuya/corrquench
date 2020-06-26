# Taku Ito
# 07/01/2019

# LIF balanced spiking model

import numpy as np
import numpy.matlib as matlib
from scipy.signal import convolve2d

def spikingModel(nE, nI, W, stim_e, stim_i,
                 time=1000, dt=0.1, Vth=1.0, Vre=0.0,
                 tau_e=15.0, tau_i=10.0, ref_e=5.0, ref_i=5.0, 
                 syntau2_e=3.0, syntau2_i=2.0, syntau1=1.0):
    """
    Spiking network, using parameters and equations from Huang et al., 2019 (Neuron)
    """

    T = np.arange(0,time,dt)
    Ncells = nE + nI

    # Set initial conditions
    V = np.random.uniform(0,1,size=(Ncells,))
    # Set time constants
    tau = np.zeros((Ncells,))
    tau[:nE] = tau_e
    tau[nE:] = tau_i
    # Instantiate synaptic currents empty matrix
    I = np.zeros((Ncells,))
    # Instantiate spiking matrix
    # Synaptic rise gating variable
    xrse_e = np.zeros((Ncells,))
    xdec_e = np.zeros((Ncells,))
    xrse_i = np.zeros((Ncells,))
    xdec_i = np.zeros((Ncells,))

    forwardInputsE = np.zeros((Ncells,))
    forwardInputsI = np.zeros((Ncells,))
    forwardInputsEPrev = np.zeros((Ncells,))
    forwardInputsIPrev = np.zeros((Ncells,))

    lastSpike = np.ones((Ncells,))*-100

    print("Starting simulation")

    # Set random biases from a uniform distribution
    # Excitatory neurons
    mu_e = np.random.uniform(1.1,1.2,size=(nE,))
    #mu_e = np.random.uniform(1.05,1.15,size=(nE,)) # Imbalanced state
    # Inhibitory neurons
    mu_i = np.random.uniform(1.0,1.05,size=(nI,))
    mu = np.hstack((mu_e,mu_i))

    maxrate = 500 # max rate is 100hz
    maxtimes = int(np.round(maxrate*time/1000))
    spktimes = np.zeros((Ncells,maxrate))
    ns = np.zeros((Ncells,),dtype=int)

    for t in np.arange(0,len(T)-1):
        if int(np.mod(t,(len(T)-1)/100.0)) == 1: print('\r',np.round(100*t/(len(T)-1)))

        
        forwardInputsE[:] = 0
        forwardInputsI[:] = 0

        for ci in range(Ncells):
            xrse_e[ci] += -dt * xrse_e[ci] / syntau1 + forwardInputsEPrev[ci]
            xdec_e[ci] += -dt * xdec_e[ci] / syntau2_e + forwardInputsEPrev[ci]
            xrse_i[ci] += -dt * xrse_i[ci] / syntau1 + forwardInputsIPrev[ci]
            xdec_i[ci] += -dt * xdec_i[ci] / syntau2_i + forwardInputsIPrev[ci]

            synInput = (xdec_e[ci] - xrse_e[ci])/(syntau2_e - syntau1) + (xdec_i[ci] - xrse_i[ci])/(syntau2_i - syntau1)
            refrac = 5
            if t*dt > (lastSpike[ci] + refrac):
                V[ci] += dt * ((1/tau[ci]) * (mu[ci]-V[ci]) + synInput)

                if V[ci] > Vth:
                    V[ci] = Vre
                    lastSpike[ci] = t*dt
                    ns[ci] += 1
                    if ns[ci] <= maxtimes:
                        spktimes[ci,ns[ci]] = t*dt

                        for j in range(Ncells):
                            if W[ci,j] > 0: # E synapse
                                forwardInputsE[j] += W[ci,j]
                            elif W[ci,j] < 0: # I synapse
                                forwardInputsI[j] += W[ci,j]

        forwardInputsEPrev = forwardInputsE.copy()
        forwardInputsIPrev = forwardInputsI.copy()

    return spktimes,ns,nE,Ncells,T


 
def slidingWindow(data,binSize=50,shiftSize=10,nproc=10):
    """
    data - organized by time (ms) x trial, data represents data from only one region/neuron
    binsize - window size to compute number of spikes
    shiftsize - shift window by this amount
    
    Effectively downsamples data
    """
    
    tLength = data.shape[0]
    nTrials = data.shape[1]
    
    ###
    
    inputs = []
    for trial in range(nTrials):
        inputs.append(data[:,trial],binSize,shiftSize)
        
    pool = mp.Pool(processes=nproc)
    results = pool.map_async(_slide,inputs).get()
    pool.close()
    pool.join()
    
    out = []
    for result in results:
        out.append(result.T)
        
    outarray = np.zeros((result.shape[0],nTrials))
    for i in range(nTrials):
        outarray[:,i] = out[i]

    return outarray

### Helper function for parallel processing
def _slide(trialdata,binSize,shiftSize):
    tLength = trialdata.shape[0]
    
    downSampledData = []
    i = 0
    while i < (tLength-binSize):
        downSampledData.append(np.mean(trialdata[i:(i+binSize)],axis=0))
        i += shiftSize

    return np.asarray(downSampledData)


def constructConnMatrices(nE=4000, nI=1000, n_clusters=50,
                          pEE=.2, pEI=.5, pIE=.5, pII=.5, rEE=1.0,
                          jEE_out=0.024, jEE_in=1.9, jEI=0.014, jIE=-0.045, jII=-0.057): # Need to add synaptic strengths
    """
    Construct clustered connections given a set of probabilities (and clustering coefficients)
    """
    # Structural connectivity (probabilities taken from Litwin-Kumar & Doiron, 2012)
    if rEE==1:
        sEE = np.random.binomial(1,pEE,size=(nE,nE))
        wEE = sEE*jEE_out
    else:
        neurons_per_clust = int(nE/float(n_clusters))
        pEE_out = (pEE*(nE-1))/(rEE*neurons_per_clust + (nE-1) - neurons_per_clust)
        pEE_in = rEE*pEE_out

        # First define all pEE_out connections
        sEE = np.random.binomial(1,pEE_out,size=(nE,nE))
        wEE = sEE * jEE_out
        # Now we will re-do clustered connections
        neuron_count = 0
        for clust in range(n_clusters):
            i = int(neuron_count)
            j = int(neuron_count + neurons_per_clust)
            sEE[i:j,i:j] = 0
            sEE[i:j,i:j] = np.random.binomial(1,pEE_in,size=(neurons_per_clust,neurons_per_clust))
            wEE[i:j,i:j] = sEE[i:j,i:j] * jEE_in
            neuron_count += neurons_per_clust

    # No self connections
    np.fill_diagonal(wEE,0)

    # Structural matrices
    sEI = np.random.binomial(1,pEI,size=(nE,nI))
    sIE = np.random.binomial(1,pIE,size=(nI,nE))
    sII = np.random.binomial(1,pII,size=(nI,nI))

    # Synaptic weights
    wEI = sEI * jEI
    wIE = sIE * jIE
    wII = sII * jII

    # No self connections
    np.fill_diagonal(wII,0)

    W1 = np.vstack((wEE,wIE))
    W2 = np.vstack((wEI,wII))
    W = np.hstack((W1,W2))


    return W

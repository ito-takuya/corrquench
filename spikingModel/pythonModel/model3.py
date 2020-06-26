# Taku Ito
# 07/01/2019

# LIF balanced spiking model

import numpy as np
import numpy.matlib as matlib
from scipy.signal import convolve2d

def spikingModel(wEE, wEI, wIE, wII, stim_e, stim_i,
                 time=1000, dt=0.1, Vth=1.0, Vre=0.0,
                 tau_e=15.0, tau_i=10.0, ref_e=5.0, ref_i=5.0, 
                 syntau2_e=3.0, syntau2_i=2.0, syntau1=1.0):
    """
    Spiking network, using parameters and equations from Huang et al., 2019 (Neuron)
    """

    T = np.arange(0,time,dt)
    nE = wEE.shape[0]
    nI = wII.shape[0]

    Ve = np.zeros((nE,len(T)))
    Vi = np.zeros((nI,len(T)))
    # Set initial conditions
    Ve = np.random.uniform(0,1,size=(nE,))
    Vi = np.random.uniform(0,1,size=(nI,))
    # Instantiate synaptic currents empty matrix
    Ie = np.zeros((nE,len(T)))
    Ii = np.zeros((nI,len(T)))
    # Instantiate spiking matrix
    spkE = np.zeros((nE,time))
    spkI = np.zeros((nI,time))
    # Instantiate synaptic input matrix (temporally downsampled)
    synE = np.zeros((nE,time))
    synI = np.zeros((nI,time))

    bin_spkE = np.zeros((nE,))
    bin_spkI = np.zeros((nI,))
    # Synaptic rise gating variable
    xrse_ee = np.zeros((nE,))
    xdec_ee = np.zeros((nE,))
    xrse_ei= np.zeros((nI,))
    xdec_ei = np.zeros((nI,))
    xrse_ie = np.zeros((nE,))
    xdec_ie = np.zeros((nE,))
    xrse_ii= np.zeros((nI,))
    xdec_ii = np.zeros((nI,))


    # Set random biases from a uniform distribution
    # Excitatory neurons
    mu_e = np.random.uniform(1.1,1.2,size=(nE,))
    #mu_e = np.random.uniform(1.05,1.15,size=(nE,)) # Imbalanced state
    # Inhibitory neurons
    mu_i = np.random.uniform(1.0,1.05,size=(nI,))

    maxrate = 500 # max rate is 100hz
    maxtimes = int(np.round(maxrate*time/1000))
    timesE = np.zeros((nE,maxrate))
    timesI = np.zeros((nI,maxrate))
    ne_s = np.zeros((nE,),dtype=int)
    ni_s = np.zeros((nI,),dtype=int)

    refractory_e = np.zeros((nE,))
    refractory_i = np.zeros((nI,))
    for t in range(len(T)-1):
        ## Using RK2 method

        ## K1s
        Ve = Ve + dt*((mu_e + stim_e - Ve)/tau_e + Ie[:,t])
        Vi = Vi + dt*((mu_i + stim_i - Vi)/tau_i + Ii[:,t])

        # Synaptic gating
        # Excitatory synapses
        xrse_ee = xrse_ee - dt*xrse_ee/syntau1 + np.matmul(bin_spkE,wEE)
        xdec_ee = xdec_ee - dt*xdec_ee/syntau2_e + np.matmul(bin_spkE,wEE)
        xrse_ei = xrse_ei - dt*xrse_ei/syntau1 + np.matmul(bin_spkE,wEI)
        xdec_ei = xdec_ei - dt*xdec_ei/syntau2_e + np.matmul(bin_spkE,wEI)
        # Inhibitory dt*synapses
        xrse_ie = xrse_ie - dt*xrse_ie/syntau1 + np.matmul(bin_spkI,wIE)
        xdec_ie = xdec_ie - dt*xdec_ie/syntau2_i + np.matmul(bin_spkI,wIE)
        xrse_ii = xrse_ii - dt*xrse_ii/syntau1 + np.matmul(bin_spkI,wII)
        xdec_ii = xdec_ii - dt*xdec_ii/syntau2_i + np.matmul(bin_spkI,wII)

        # Calculate synaptic outputs given rise and decay times
        Ie[:,t+1] = (xdec_ee - xrse_ee)/(syntau2_e - syntau1) + (xdec_ie - xrse_ie)/(syntau2_i - syntau1)
        Ii[:,t+1] = (xdec_ii - xrse_ii)/(syntau2_i - syntau1) + (xdec_ei - xrse_ei)/(syntau2_e - syntau1)

        ## Spiking
        # Find which neurons exceed threshold (and are not in a refractory period)
        bin_spkE = np.multiply(Ve>Vth, refractory_e==0.0)
        bin_spkI = np.multiply(Vi>Vth, refractory_i==0.0)

        # Save spike time (and downsample to 1ms)
        tms = int(np.floor(T[t]))
        spkE[bin_spkE,tms] = 1 # spikes are okay - refractory period is 5ms, anyway
        spkI[bin_spkI,tms] = 1
        synE[:,tms] = synE[:,tms] + Ie[:,t]
        synI[:,tms] = synI[:,tms] + Ii[:,t]

        # Reset voltages
        Ve[bin_spkE] = Vre
        Vi[bin_spkI] = Vre

        # spike times
        timesE[bin_spkE,ne_s[bin_spkE]] = T[t+1]
        timesI[bin_spkI,ni_s[bin_spkI]] = T[t+1]
        ne_s[bin_spkE] = ne_s[bin_spkE] + 1
        ni_s[bin_spkI] = ni_s[bin_spkI] + 1


        # Set refractory period
        # Add a refractory time step to neurons who just spiked, and to those are still in a refractory period
        refractory_e = refractory_e + (bin_spkE * dt) + (refractory_e!=0) * dt 
        refractory_i = refractory_i + (bin_spkI * dt) + (refractory_i!=0) * dt
        # Once refractory period is complete, allow to spike
        can_spike_again_e = np.round(refractory_e,1) == ref_e
        can_spike_again_i = np.round(refractory_i,1) == ref_i

        refractory_e[can_spike_again_e] = 0.0
        refractory_i[can_spike_again_i] = 0.0

        # Set neurons who are in their refractory to the baseline membrane potential
        in_refractory_e = refractory_e != 0.0
        in_refractory_i = refractory_i != 0.0

        Ve[in_refractory_e] = Vre
        Vi[in_refractory_i] = Vre
        
    return spkE, spkI, synE, synI, timesE, timesI, ne_s, ni_s

 
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

    return wEE, wEI, wIE, wII

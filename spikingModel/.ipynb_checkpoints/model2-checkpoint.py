# Taku Ito
# 07/01/2019

# LIF balanced spiking model

import numpy as np
import numpy.matlib as matlib
from scipy.signal import convolve2d

def spikingModel(wEE, wEI, wIE, wII, stim_e, stim_i,
                 time=100, dt=0.1, Vth=1.0, Vre=0.0,
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
    Ve[:,0] = np.random.uniform(0,1,size=(nE,))
    Vi[:,0] = np.random.uniform(0,1,size=(nI,))
    # Instantiate synaptic currents empty matrix
    Ie = np.zeros((nE,len(T)))
    Ii = np.zeros((nI,len(T)))
    # Instantiate spiking matrix
    spkE = np.zeros((nE,len(T)))
    spkI = np.zeros((nI,len(T)))

    # Synaptic rise gating variable
    xerise = np.zeros((nE,len(T)))
    xedecay = np.zeros((nE,len(T)))
    xirise = np.zeros((nI,len(T)))
    xidecay = np.zeros((nI,len(T)))


    # Set random biases from a uniform distribution
    # Excitatory neurons
    mu_e = np.random.uniform(1.1,1.2,size=(nE,))
    #mu_e = np.random.uniform(1.05,1.15,size=(nE,)) # Imbalanced state
    # Inhibitory neurons
    mu_i = np.random.uniform(1.0,1.05,size=(nI,))

    refractory_e = np.zeros((nE,))
    refractory_i = np.zeros((nI,))
    for t in range(len(T)-1):
        ## Using RK2 method

        ## K1s
        k1_Ve = (mu_e + stim_e - Ve[:,t])/tau_e + np.matmul(Ie[:,t],wEE) + np.matmul(Ii[:,t],wIE)
        k1_Vi = (mu_i + stim_i - Vi[:,t])/tau_i + np.matmul(Ii[:,t],wII) + np.matmul(Ie[:,t],wEI) 

        # Synaptic gating
        k1xerise = -xerise[:,t]/syntau1 + spkE[:,t]
        k1xedecay = -xedecay[:,t]/syntau2_e + spkE[:,t]
        # 
        k1xirise = -xirise[:,t]/syntau1 + spkI[:,t]
        k1xidecay = -xidecay[:,t]/syntau2_i + spkI[:,t]

        ## Midpoint
        # a1 - midpoint (Euler) estimate
        a_Ve = Ve[:,t] + k1_Ve*dt
        a_Vi = Vi[:,t] + k1_Vi*dt
        # 
        a_xerise = xerise[:,t] + k1xerise*dt
        a_xedecay = xedecay[:,t] + k1xedecay*dt
        a_xirise = xirise[:,t] + k1xirise*dt
        a_xidecay = xidecay[:,t] + k1xidecay*dt
        aIe = (a_xedecay-a_xerise)/(syntau2_e - syntau1)
        aIi = (a_xidecay-a_xirise)/(syntau2_i - syntau1)

        ## K2s
        k2_Ve = (mu_e + stim_e - k1_Ve)/tau_e + np.matmul(aIe,wEE) + np.matmul(aIi,wIE) 
        k2_Vi = (mu_i + stim_i - k1_Vi)/tau_i + np.matmul(aIi,wII) + np.matmul(aIe,wEI) 
        
        # Synaptic gating
        k2xerise = -a_xerise/syntau1 + spkE[:,t]
        k2xedecay = -a_xedecay/syntau2_e + spkE[:,t]
        #
        k2xirise = -a_xirise/syntau1 + spkI[:,t]
        k2xidecay = -a_xidecay/syntau2_i + spkI[:,t]

        ## RK2 estimates
        Ve[:,t+1] = Ve[:,t] + ((k1_Ve + k2_Ve)/2.0)*dt
        Vi[:,t+1] = Vi[:,t] + ((k1_Vi + k2_Vi)/2.0)*dt
        # Inputs
        xerise[:,t+1] = xerise[:,t] + (k1xerise+k2xerise)*dt/2.0
        xedecay[:,t+1] = xedecay[:,t] + (k1xedecay+k2xedecay)*dt/2.0
        #
        xirise[:,t+1] = xirise[:,t] + (k1xirise+k2xirise)*dt/2.0
        xidecay[:,t+1] = xidecay[:,t] + (k1xidecay+k2xidecay)*dt/2.0
        # Calculate synaptic outputs given rise and decay times
        Ie[:,t+1] = (xedecay[:,t+1]-xerise[:,t+1])/(syntau2_e - syntau1)
        Ii[:,t+1] = (xidecay[:,t+1]-xirise[:,t+1])/(syntau2_i - syntau1)

        ## Spiking
        # Find which neurons exceed threshold (and are not in a refractory period)
        bin_spkE = np.multiply(Ve[:,t+1]>Vth, refractory_e==0)
        bin_spkI = np.multiply(Vi[:,t+1]>Vth, refractory_i==0)
        # Set spike
        spkE[:,t+1] = bin_spkE
        spkI[:,t+1] = bin_spkI

        # Reset voltages
        Ve[bin_spkE,t+1] = Vre
        Vi[bin_spkI,t+1] = Vre

        # Set refractory period
        refractory_e = refractory_e + (bin_spkE * dt) + (refractory_e!=0) * dt
        refractory_i = refractory_i + (bin_spkI * dt) + (refractory_i!=0) * dt
        # Once refractory period is complete, allow to spike
        can_spike_again_e = np.round(refractory_e,1) == ref_e
        can_spike_again_i = np.round(refractory_i,1) == ref_i

        refractory_e[can_spike_again_e] = 0
        refractory_i[can_spike_again_i] = 0

        # Set neurons who are in their refractory to the baseline membrane potential
        in_refractory_e = refractory_e != 0
        in_refractory_i = refractory_i != 0

        Ve[in_refractory_e,t+1] = Vre
        Vi[in_refractory_i,t+1] = Vre
        
        ### Synaptic convolution if there was a spike (for the next time point)
        #synE = convolve2d(convolve2d(spkE[:,t+1:t+2],syn_filter_e), synE[:,1:]) # sum the convolution from previous time point and go to next time point 
        #synI = convolve2d(convolve2d(spkI[:,t+1:t+2],syn_filter_i), synI[:,1:]) # sum the convolution from previous time point and go to next time point


    return Ve, Vi, spkE, spkI, Ie, Ii

        


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
        pEE_out = (pEE*nE)/(rEE*neurons_per_clust + nE - neurons_per_clust)
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

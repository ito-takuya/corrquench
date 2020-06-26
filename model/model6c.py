# TaskFCMechs - large scale network model
# Author: Takuya Ito 
# Updated: 3/18/2020 -- Day 5 COVID-19 quarantine

## Import modules
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
from scipy.stats import gamma
import seaborn as sns

# Define transfer function (sigmoid)
#phi = lambda x: np.tanh(x-1)
phi = lambda x: 1/(1+np.exp(-(x-2)))
#phi = lambda x: 1/(1+np.exp(-(x-3.8)))

def generateNetwork(ncommunities=2, innetwork_dsity=.5, outnetwork_dsity=.05,
                    nodespercommunity=50, showplot=False):
    """
    Randomly generates a structural network 

    Parameters:
        ncommunities = number of communities within the network 
        innetwork_dsity = connectivity density of within-network connections
        outnetwork_dsity = connectivity density of out-of-network connections
        showplot = if set to True, will automatically display the structural matrix using matplotlib.pyplot

    Returns: 
        Unweighted structural connectivity matrix (with 1s indicating edges and 0s otherwise)
    """

    totalnodes = nodespercommunity * ncommunities

    W = np.zeros((totalnodes,totalnodes))
    # Construct structural matrix
    for i in range(ncommunities):
        for j in range(ncommunities):
            # Set within network community connections
            if i==j:
                tmp_a = np.random.rand(nodespercommunity,nodespercommunity)<=innetwork_dsity
                # within network synaptic connectivity has mu=2.0, sd=.2
                #synaptic_matrix = np.random.normal(loc=1.0,scale=0.001,size=(nodespercommunity,nodespercommunity))
                #synaptic_matrix = np.ones((nodespercommunity,nodespercommunity))

                indstart = i*nodespercommunity
                indend = i*nodespercommunity+nodespercommunity
                #W[indstart:indend,indstart:indend] = np.multiply(tmp_a,synaptic_matrix)
                W[indstart:indend,indstart:indend] = tmp_a 

                # Normalize so that within-network connections = 1
                #suminputs = np.sum(W[indstart:indend, indstart:indend],axis=0)
                #W[indstart:indend, indstart:indend] = (4-1)/(nodespercommunity-1)/innetwork_dsity # should be 1, but need to account for self-connections
                #W[indstart:indend, indstart:indend] = np.multiply(W[indstart:indend,indstart:indend],(2.0-1)/(nodespercommunity-1)/innetwork_dsity) # should be 5, but need to account for self-connections
            else:
                tmp_b = np.random.rand(nodespercommunity,nodespercommunity)<=outnetwork_dsity

                indstart_i = i*nodespercommunity
                indend_i = i*nodespercommunity + nodespercommunity
                indstart_j = j*nodespercommunity
                indend_j = j*nodespercommunity + nodespercommunity
                W[indstart_i:indend_i, indstart_j:indend_j] = tmp_b

                # Normalize cross-network connections
                #suminputs = np.sum(W[indstart_i:indend_i, indstart_j:indend_j],axis=0)
                #W[indstart_i:indend_i, indstart_j:indend_j] = np.divide(W[indstart_i:indend_i, indstart_j:indend_j],suminputs) * (0.0/nodespercommunity)/outnetwork_dsity
                #W[indstart_i:indend_i, indstart_j:indend_j] = np.multiply(np.multiply(tmp_b,synaptic_matrix),(0)/nodespercommunity/outnetwork_dsity)

    # Normalize all inter-region weights to ge equal to 1, which is the amount of self-coupling
    np.fill_diagonal(W,0)
    W = np.asarray(W,dtype='float')
    input_sum = np.sum(W,axis=0)
    #print(input_sum)
    W = np.divide(W,input_sum)

    # Make sure self-connections exist
    np.fill_diagonal(W, 1.0)
    #np.fill_diagonal(W[:nodespercommunity,:nodespercommunity],1.00)
    # remove nan values
    #nan_ind = np.isnan(W)
    #W[nan_ind] = 0
    

    return W

def networkModel(G, Tmax=100,dt=.1,g=1.0,s=1.0,tau=1,synweights=0,I=None, noise=None):
    """
    G = Synaptic Weight Matrix
    Tmax = 100      (1sec / 1000ms)
    dt = .1         (1ms)
    g = 1.0         Coupling 
    s = 1.0         Self connection
    tau = 1.0       Time constant 
    synweights      SD of synaptic weight distribution. 0 indicates all weights are fixed. Mean of synaptic weights are always 1.
    I = 0.0         Stimulation/Task
    
    
    """
    T = np.arange(0, Tmax, dt)
    totalnodes = G.shape[0]

    # External input (or task-evoked input) && noise input
    if I is None: I = np.zeros((totalnodes,len(T)))
    # Noise parameter
    if noise == None: noise = np.zeros((totalnodes,len(T)))
    #if noise == 1: noise = np.random.normal(0,1.0,size=(totalnodes,len(T)))
    if noise != None: 
        noise = np.random.normal(0,noise,size=(totalnodes,len(T)))

    # Multiply weights by s and g
    diag_ind = np.diag_indices(totalnodes,ndim=2)
    G[diag_ind] = np.multiply(G[diag_ind],s)
    tril_ind = np.tril_indices(totalnodes,k=1)
    triu_ind = np.triu_indices(totalnodes,k=1)
    G[tril_ind] = np.multiply(G[tril_ind],g)
    G[triu_ind] = np.multiply(G[triu_ind],g)
    # Generate synaptic weights
    G[tril_ind] = np.multiply(G[tril_ind],np.random.normal(1,synweights,len(tril_ind[0])))
    G[triu_ind] = np.multiply(G[triu_ind],np.random.normal(1,synweights,len(triu_ind[0])))

    # Initial conditions and empty arrays
    Enodes = np.zeros((totalnodes,len(T)))
    # Initial conditions
    Einit = np.random.normal(0,1,(totalnodes,))
    Enodes[:,0] = Einit

    meaninput = np.zeros((totalnodes,len(T)))
    for t in range(len(T)-1):

        ## Solve using Runge-Kutta Order 2 Method
        # With auto-correlation
        spont_act = (noise[:,t])
        k1e = -Enodes[:,t] + phi(np.matmul(G,Enodes[:,t]) + spont_act + I[:,t]) # input output transfer func 
#        k1e += s*phi(Enodes[:,t] + spont_act) # Local processing
        k1e = k1e/tau
        # 
        ave = Enodes[:,t] + k1e*dt
        # calculate mean input 
        meaninput[:,t] = (-Enodes[:,t] + phi(np.matmul(G,spont_act)))/tau
        #
        # With auto-correlation
        spont_act = (noise[:,t+1])
        k2e = -ave + phi(np.matmul(G,ave) + spont_act + I[:,t+1]) # Coupling
#        k2e += s*phi(ave + spont_act) # Local processing
        k2e = k2e/tau

        Enodes[:,t+1] = Enodes[:,t] + (.5*(k1e+k2e))*dt

        meaninput[:,t+1] = (-Enodes[:,t+1] + phi(np.matmul(G,spont_act)))/tau


    return Enodes

def runSubjectRuns(subj, ncommunities=1, innetwork_dsity = 0.2, outnetwork_dsity = 0.0, nodespercommunity=300, s=1.0, g=4.0, synweights=0):
    """
    MAIN COMPUTATION METHOD

    Procedures performed in this function:
    1. Construct structural and synaptic connectivity matrices for single subject
    2. Run resting-state simulation for single subject
    3. Run task simulations for single subject
    """
    ######################
    #### 0. SET UP PARAMETERS

    # Set number of time points
    Tmax = 1000
    ######################


    ######################
    Ci = np.repeat(np.arange(ncommunities),nodespercommunity) # Construct a community affiliation vector
    totalnodes = nodespercommunity*ncommunities
   
    ##########
    # Construct structural matrix
    W = generateNetwork(ncommunities=ncommunities, innetwork_dsity=innetwork_dsity,
                                  outnetwork_dsity=outnetwork_dsity, 
                                  nodespercommunity=nodespercommunity, showplot=False)
    ######################

    
    ######################
    #### 2. RUN RESTING-STATE SIMULATION AND CONVOLVE TO FMRI DATA
    # Basic static parameters for simulation
    dt = .1 # Sampled @ 10ms
    T = np.arange(0,Tmax,dt)
    tau = .1
    
    #### Compute rest portion of analysis
    ## Run rest simulation prior to task simulations
    restdata = networkModel(W, Tmax=Tmax,
                            dt=dt,g=g,s=s,tau=tau,I=None,noise=1)

    ######################

    ######################
    #### 4. RUN TASK SIMULATION 
    # Generate output variables
    print('Running subject', subj)
    
    #### Task Identification:  Selection stimulation nodes -- Stimulate 1/4 nodes out of nodespercommunity
#    stim_nodes = np.arange(0,totalnodes,2,dtype=int)
    stim_nodes = np.arange(totalnodes,dtype=int)
    stimtimes = np.ones((totalnodes,len(T)))*1
        
    taskdata = networkModel(W,Tmax=Tmax,dt=dt,g=g,s=s,tau=tau,synweights=synweights,
                            I=stimtimes, noise=1)
    ######################

    return restdata, taskdata 

def subjectSimulation(subj, seed, ncommunities=1, innetwork_dsity=0.2, outnetwork_dsity=0.0, nodespercommunity=300, s=1.0, g=4.0, synweights=0):
    """
    Run entire ActFlow procedure for single subject and saves to file specified as the 'outdir' parameter.
    Saves all files to an output directory named ItoEtAl2017_Simulations/

    Parameters:
        subj = subject number (as an int)
    """
    np.random.seed(seed)

    ## First write out to regular directory for regular subjects
    restdata, taskdata = runSubjectRuns(subj, ncommunities=ncommunities,
                                        innetwork_dsity=innetwork_dsity, outnetwork_dsity=outnetwork_dsity, 
                                        nodespercommunity=nodespercommunity, s=s, g=g, synweights=synweights)
    
    ## Save files
    # On Linux server
    #outdir = '/projects3/TaskFCMech/data/results/simulation4/' 
    #if not os.path.exists(outdir): os.makedirs(outdir)
    
    #np.savetxt(outdir + 'subj' + str(subj) + '_restdata_neural.txt', restdata, delimiter=',')
    #np.savetxt(outdir + 'subj' + str(subj) + '_taskdata_neural.txt', taskdata, delimiter=',')
    
    return restdata, taskdata 

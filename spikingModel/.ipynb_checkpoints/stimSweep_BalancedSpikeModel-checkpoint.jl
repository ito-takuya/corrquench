using Statistics
using PyPlot
using HDF5

#### Dynamic parameters
taue  = 15.0
taui = 10.0



##### Construct network and synaptic connections
Ne = 4000
Ni = 1000
Nepop = 80 # cluster sizer (for E-E connections)
# Connectivity probabilities
pee = .2
pei = .5 
pie = .5
pii = .5
ree = 2.5 # Clustering coef
K = Ne*pee # Average number of E-E connections
# Synaptic efficacies
jeeout = 10.0 / (taue * sqrt(K))
jeein = 1.9 * 10.0 / (taue * sqrt(K))
jie = 4.0 / (sqrt(K) * taui)
jei = -16.0 * 1.2 / (taue * sqrt(K))
jii = -16.0 / (taui * sqrt(K))

@time W = constructMatrices(jeeout, jeein, jie, jei, jii, Ne=Ne, Ni=Ni, Nepop=Nepop, pee=pee, pei=pei, pie=pie, pii=pii, ree=ree);


#### Run sweep
include("sim_tito2.jl")

time = 2000 #ms
dt = 0.1 #ms
vre = 0.
threshe = 1
threshi = 1

tauerise = 1
tauedecay = 3 #formerly 3
tauirise = 1
tauidecay = 2 #formerly 2

muemin = 1.1 
muemax = 1.2 
muimin = 1.0
muimax = 1.05

refrace = 5
refraci = 5

stimamp = 0.07 #range(-.2,.4,step=0.02)
stimstart = 0
stimend = 2000

trials = 20

stimamps = range(-.1,.5,step=0.05)

for stim in stimamps
    print("Running simulation for stimamp ", stimamp, "\n")

    for i=1:trials

        outputfilename = string("/projects3/TaskFCMech/data/results/spikingModel/balanced_simoutput_stim", stim, "_trial",i, ".h5")

        if stim<0
            stimamp = abs(stim)
            Nstime = 0
            Nstimi = 20
        elseif stim>=0
            stimamp = abs(stim)
            Nstime = 80
            Nstimi = 0
        end

        times_spont, ns_spont, Ne, Ncells, T, spikes_spont, synInput_spont = sim(W, Ne=Ne, Ni=Ni, T=time, dt=dt, 
                                                                                       vre=vre, threshe=threshe, threshi=threshi, taue=taue, taui=taui,
                                                                                       tauerise=tauerise, tauedecay=tauedecay, tauirise=tauirise, tauidecay=tauidecay,
                                                                                       muemin=muemin, muemax=muemax, muimin=muimin, muimax=muimax,
                                                                                       Nstime=Nstime, Nstimi=Nstimi, stimamp=stimamp, stimstart=stimstart, stimend=stimend,
                                                                                       refrace=refrace, refraci=refraci, save=true, filename=outputfilename, overwrite=true);
#         # Print out mean firing rates
#         println("\tmean excitatory firing rate: ",mean(1000*ns_spont[1:Ne]/T)," Hz")
#         println("\tmean inhibitory firing rate: ",mean(1000*ns_spont[(Ne+1):Ncells]/T)," Hz")
    end
end

#this file is part of litwin-kumar_doiron_cluster_2012
#Copyright (C) 2014 Ashok Litwin-Kumar
#see README for more information
using Statistics
using HDF5
using StatsBase


function constructMatrices(jeeout, jeein, jie, jei, jii ;
                           Ne=4000, Ni=1000, Nepop=80,
                           pee=.2, pei=.5, pie=.5, pii=.5, ree=2.5)
    
    """
    Construct weights as in Litwin-Kumar & Doiron, 2012
    """
    Ncells = Ne + Ni
    K = Ne*pee
    sqrtK = sqrt(K)

    #set up connection probabilities within and without blocks
    ratiopee = ree
    peeout = K/(Nepop*(ratiopee-1) + Ne)
    peein = ratiopee*peeout

    Npop = round(Int,Ne/Nepop)

    weights = zeros(Ncells,Ncells)

    #random connections
    weights[1:Ne,1:Ne] = jeeout*(rand(Ne,Ne) .< peeout)
    weights[1:Ne,(1+Ne):Ncells] = jei*(rand(Ne,Ni) .< pei)
    weights[(1+Ne):Ncells,1:Ne] = jie*(rand(Ni,Ne) .< pie)
    weights[(1+Ne):Ncells,(1+Ne):Ncells] = jii*(rand(Ni,Ni) .< pii)

    #connections within cluster
    for pii = 1:Npop
        ipopstart = 1 + Nepop*(pii-1)
        ipopend = pii*Nepop

        weights[ipopstart:ipopend,ipopstart:ipopend] = jeein*(rand(Nepop,Nepop) .< peein)
    end

    for ci = 1:Ncells
        weights[ci,ci] = 0
    end

    return weights
end

function sim(weights ; Ne=4000, Ni=1000, T=2000, dt=.1,
             vre=0., threshe=1, threshi=1, taue=15, taui=10,
             tauerise=1, tauedecay=3, tauirise=1, tauidecay=2,
             muemin=1.1, muemax=1.2, muimin=1, muimax=1.05,
             Nstime=400, Nstimi=0, stimamp=0.07,stimstart=1500,stimend=2000,
             refrace=5, refraci=5, save=false, filename="simoutput.h5", overwrite=false)

    println("setting up parameters")
    Ncells = Ne + Ni

    #stimulation
    stimstr = stimamp/taue

    maxrate = 100 #(Hz) maximum average firing rate.  if the average firing rate across the simulation for any neuron exceeds this value, some of that neuron's spikes will not be saved

    mu = zeros(Ncells)
    thresh = zeros(Ncells)
    tau = zeros(Ncells)

    mu[1:Ne] = (muemax-muemin)*rand(Ne) .+ muemin
    mu[(Ne+1):Ncells] = (muimax-muimin)*rand(Ni) .+ muimin

    # Find cells to stimulate
    #stimcells = sample(range(1,Ne,step=1), Nstim, replace=false) 
    #for ci in stimcells
    #    mu[ci] += stimamp
    #end

    thresh[1:Ne] .= threshe
    thresh[(1+Ne):Ncells] .= threshi

    tau[1:Ne] .= taue
    tau[(1+Ne):Ncells] .= taui


    maxTimes = round(Int,maxrate*T/1000)
    times = zeros(Ncells,maxTimes)
    ns = zeros(Int,Ncells)

    forwardInputsE = zeros(Ncells) #summed weight of incoming E spikes
    forwardInputsI = zeros(Ncells)
    forwardInputsEPrev = zeros(Ncells) #as above, for previous timestep
    forwardInputsIPrev = zeros(Ncells)

    xerise = zeros(Ncells) #auxiliary variables for E/I currents (difference of exponentials)
    xedecay = zeros(Ncells)
    xirise = zeros(Ncells)
    xidecay = zeros(Ncells)

    v = rand(Ncells) #membrane voltage

    lastSpike = -100*ones(Ncells) #time of last spike

    Nsteps = round(Int,T/dt)

    println("starting simulation")

    spikes = zeros(Ncells,T)
    synInputsEI = zeros(Ncells,Nsteps)

    #begin main simulation loop
    for ti = 1:Nsteps
        if mod(ti,Nsteps/100) == 1  #print percent complete
                print("\r",round(Int,100*ti/Nsteps))
        end
        t = dt*ti
        forwardInputsE[:] .= 0
        forwardInputsI[:] .= 0

        for ci = 1:Ncells
            xerise[ci] += -dt*xerise[ci]/tauerise + forwardInputsEPrev[ci]
            xedecay[ci] += -dt*xedecay[ci]/tauedecay + forwardInputsEPrev[ci]
            xirise[ci] += -dt*xirise[ci]/tauirise + forwardInputsIPrev[ci]
            xidecay[ci] += -dt*xidecay[ci]/tauidecay + forwardInputsIPrev[ci]

            synInput = (xedecay[ci] - xerise[ci])/(tauedecay - tauerise) + (xidecay[ci] - xirise[ci])/(tauidecay - tauirise)
            synInputsEI[ci,ti] = synInput

            if (ci < Nstime) && (t > stimstart) && (t < stimend)
                synInput += stimstr;
            end

            # Stimulate inhibitory neurons
            if (ci>Ne) && (ci < (Ne + Nstimi)) && (t > stimstart) && (t < stimend)
                synInput += stimstr;
            end
            
            if ci<=Ne 
                refrac = refrace
            else
                refrac = refraci
            end
            if t > (lastSpike[ci] + refrac)  #not in refractory period
                v[ci] += dt*((1/tau[ci])*(mu[ci]-v[ci]) + synInput)

                if v[ci] > thresh[ci]  #spike occurred
                    v[ci] = vre
                    #lastSpike[ci] = t
                    #spikes[ci,Int(ceil(t))] = 1
                    ns[ci] = ns[ci]+1
                    if ns[ci] <= maxTimes
                        times[ci,ns[ci]] = t
                        lastSpike[ci] = t
                        spikes[ci,Int(ceil(t))] = 1
                    end

                    for j = 1:Ncells
                        if weights[j,ci] > 0  #E synapse
                                forwardInputsE[j] += weights[j,ci]
                        elseif weights[j,ci] < 0  #I synapse
                                forwardInputsI[j] += weights[j,ci]
                        end
                    end #end loop over synaptic projections
                end #end if(spike occurred)
            end #end if(not refractory)
        end #end loop over neurons

        forwardInputsEPrev = copy(forwardInputsE)
        forwardInputsIPrev = copy(forwardInputsI)
    end #end loop over time
    print("\r")

    times = times[:,1:maximum(ns)]
    
    frE = sum(spikes[1:Ne,:],dims=1)
    frI = sum(spikes[Ne:Ncells,:],dims=1)

    if save 
        println("Saving output to HDF5 file:", filename)
        if isfile(filename) # if filename already exists
            if overwrite
                println("overwriting previous file...")
                rm(filename)
                h5write(filename, "spikes", spikes)
                h5write(filename, "frE", frE)
                h5write(filename, "frI", frI)
                h5write(filename, "Ne", Ne)
                h5write(filename, "Ncells", Ncells)
                h5write(filename, "times", times)
                h5write(filename, "ns", ns)
                h5write(filename, "T", T)
                h5write(filename, "synInputsEI", synInputsEI)
            else
                println("Not overwriting existing file...")
            end
        else
            h5write(filename, "spikes", spikes)
            h5write(filename, "frE", frE)
            h5write(filename, "frI", frI)
            h5write(filename, "Ne", Ne)
            h5write(filename, "Ncells", Ncells)
            h5write(filename, "times", times)
            h5write(filename, "ns", ns)
            h5write(filename, "T", T)
            h5write(filename, "synInputsEI", synInputsEI)
        end
    end

    return times,ns,Ne,Ncells,T,spikes,synInputsEI
end

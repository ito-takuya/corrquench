#this file is part of litwin-kumar_doiron_cluster_2012
#Copyright (C) 2014 Ashok Litwin-Kumar
#see README for more information
using Statistics

function sim()
	println("setting up parameters")
	Ncells = 5000
	Ne = 4000
	Ni = 1000
	T = 2000 #simulation time (ms)

	taue = 15 #membrane time constant for exc. neurons (ms)
	taui = 10

	#connection probabilities
	pei = .5
	pie = .5
	pii = .5

	K = 800 #average number of E->E connections per neuron
	sqrtK = sqrt(K)

	Nepop = 80

	jie = 4. / (taui*sqrtK)
	jei = -16. * 1.2 /(taue*sqrtK)
	jii = -16. / (taui*sqrtK)

	#set up connection probabilities within and without blocks
	ratioejee = 1.9
	jeeout = 10. / (taue*sqrtK)
	jeein = ratioejee*10. / (taue*sqrtK)

	ratiopee = 2.5
	peeout = K/(Nepop*(ratiopee-1) + Ne)
	peein = ratiopee*peeout

	#stimulation
	Nstim = 400 #number of neurons to stimulate (indices 1 to Nstim will be stimulated)
	stimstr = .07/taue
	stimstart = T-500
	stimend = T

	#constant bias to each neuron type
	muemin = 1.1
	muemax = 1.2
	muimin = 1
	muimax = 1.05

	vre = 0. #reset voltage

	threshe = 1 #threshold for exc. neurons
	threshi = 1

	#synaptic time constants (ms)
	tauerise = 1
	tauedecay = 3
	tauirise = 1
	tauidecay = 2

	maxrate = 100 #(Hz) maximum average firing rate.  if the average firing rate across the simulation for any neuron exceeds this value, some of that neuron's spikes will not be saved

	mu = zeros(Ncells)
	thresh = zeros(Ncells)
	tau = zeros(Ncells)

	mu[1:Ne] = (muemax-muemin)*rand(Ne) .+ muemin
	mu[(Ne+1):Ncells] = (muimax-muimin)*rand(Ni) .+ muimin

	thresh[1:Ne] .= threshe
	thresh[(1+Ne):Ncells] .= threshi

	tau[1:Ne] .= taue
	tau[(1+Ne):Ncells] .= taui

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

	spikes = zeros(Ncells,Nsteps)
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

			if (ci < Nstim) && (t > stimstart) && (t < stimend)
				synInput += stimstr;
			end
                        
                        # Set refractory
                        if ci <= Ne
                            refrac=refrace
                        else
                            refrac=refraci
                        end
			if t > (lastSpike[ci] + refrac)  #not in refractory period
				v[ci] += dt*((1/tau[ci])*(mu[ci]-v[ci]) + synInput)

				if v[ci] > thresh[ci]  #spike occurred
					v[ci] = vre
					lastSpike[ci] = t
					spikes[ci,ti] = 1
					ns[ci] = ns[ci]+1
					if ns[ci] <= maxTimes
						times[ci,ns[ci]] = t
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

	return times,ns,Ne,Ncells,T,spikes,synInputsEI
end

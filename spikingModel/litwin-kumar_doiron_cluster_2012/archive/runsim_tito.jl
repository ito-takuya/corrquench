#this file is part of litwin-kumar_doiron_cluster_2012
#Copyright (C) 2014 Ashok Litwin-Kumar
#see README for more information

using Statistics
#uncomment the line below and set doplot=true to plot a raster
using PyPlot
doplot = true

include("sim_tito.jl")

times,ns,Ne,Ncells,T,spikes,synInput = sim()


println("mean excitatory firing rate: ",mean(1000*ns[1:Ne]/T)," Hz")
println("mean inhibitory firing rate: ",mean(1000*ns[(Ne+1):Ncells]/T)," Hz")

if doplot
	println("creating raster plot")
	figure(figsize=(4,4))
	for ci = 1:Ne
		vals = times[ci,1:ns[ci]]
		y = ci*ones(length(vals))
		scatter(vals,y,s=.3,c="k",marker="o",linewidths=0)
	end
	xlim(0,T)
	ylim(0,Ne)
	ylabel("Neuron")
	xlabel("Time")
	tight_layout()
	savefig("raster.png",dpi=150)

	# Create second plot
	println("creating mean firing rate")
	figure(figsize=(4,4))
	plot(range(1,stop=2000),mean(spikes[1:Ne,1:2000],dims=1)[1,:])
	plot(range(1,stop=2000),mean(spikes[Ne+1:Ncells,1:2000],dims=1)[1,:])
	ylabel("rate")
	xlabel("Time")
	legend()
	tight_layout()
	savefig("firingrate.png",dpi=150)

	# Create second plot
	println("creating mean synaptic inputs")
	figure(figsize=(4,4))
	plot(range(1,stop=2000),mean(synInput[1:Ne,1:2000],dims=1)[1,:])
	plot(range(1,stop=2000),mean(synInput[Ne+1:Ncells,1:2000],dims=1)[1,:])
	ylabel("rate")
	xlabel("Time")
	legend()
	tight_layout()
	savefig("synapticinput.png",dpi=150)
end

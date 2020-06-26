# Takuya Ito
# Module for the core functions to run analysis on NHP data
# 01/30/2019

import numpy as np
import multiprocessing as mp
import scipy.stats as stats
import pandas as pd
import os
import scipy
os.environ['OMP_NUM_THREADS'] = str(1)

regions = ['PFC', 'FEF', 'LIP', 'MT', 'IT', 'V4']

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
        inputs.append((data[:,trial],binSize,shiftSize))
        
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
def _slide((trialdata,binSize,shiftSize)):
    tLength = trialdata.shape[0]
    
    downSampledData = []
    i = 0
    while i < (tLength-binSize):
        downSampledData.append(np.mean(trialdata[i:(i+binSize)],axis=0))
        i += shiftSize

    return np.asarray(downSampledData)


def concatenateTS(df):
    """
    Concatenates the sta_removed variable into a single matrix for simpler analysis
    """

    sta_removed = []
    sta = []
    for i in df.area.index:
        tmpmat = df['sta_removed'][i].copy()
        tmpsta = df['spikesBinned'][i].copy()

        sta_removed.append(tmpmat)
        sta.append(tmpsta)
    sta_removed = np.asarray(sta_removed)
    sta = np.asarray(sta)
        
    return sta_removed, sta
    
def computeStatistics(df,sta_removed):
    tmin = -4000
    tmax = 4000
    time = np.linspace(tmin,tmax,sta_removed.shape[1])

    minTPs = 50
    
    # Basic parameters
    nCells = sta_removed.shape[0]
    nTrials = sta_removed.shape[2]

    session_ind = df.area.index[0]
    
    # Create empty arrays to store noise correlations
    preStimNoiseCorr = np.zeros((len(regions),len(regions),nTrials))
    postStimNoiseCorr = np.zeros((len(regions),len(regions),nTrials))
    preCorr_avg = np.zeros((nTrials,))
    postCorr_avg = np.zeros((nTrials,))

    # Create empty arrays to store SD values
    preStimNoiseSD = np.zeros((len(regions),nTrials))
    postStimNoiseSD = np.zeros((len(regions),nTrials))

    # Create empty arrays to store mean FR values
    preStimFR = np.zeros((len(regions),nTrials))
    postStimFR = np.zeros((len(regions),nTrials))

    dimensionalityPre = np.zeros((nTrials,))
    dimensionalityPost = np.zeros((nTrials,))

    crossTrialAct_rest = np.zeros((len(regions),nTrials))
    crossTrialAct_task = np.zeros((len(regions),nTrials))

    badTrials = []
    badTrial1 = 0
    badTrial2 = 0
    for trial in range(nTrials):
        # First identify the beginning of recording (prior to taskStart)
        preStimStart = np.min(np.where(sta_removed[0,:,trial]!=0)[0])
        try:
#             preStimEnd = np.max(np.where(time<(df['taskInfo'][session_ind]['fixptOn'][trial])*1000)[0]) # Convert trial start times to ms
            preStimEnd = np.max(np.where(time<(df['mocolInfo'][session_ind]['fixptOn'][trial])*1000)[0]) # Convert trial start times to ms
        except:
            badTrials.append(trial)
            badTrial1 += 1
            continue
        nTPs = preStimEnd - preStimStart
        if nTPs<minTPs:
            badTrials.append(trial)
            badTrial2 += 1
            continue
        # Make sure ITI period doesn't exceed 1s prior to fixation period
        elif nTPs>100:
            preStimStart = preStimEnd-100

#         postStimStart = np.min(np.where(time>=0)[0])
        postStimStart = np.min(np.where(time>=df['mocolInfo'][session_ind]['cueOn'][trial]*1000)[0])
        postStimEnd = postStimStart + nTPs


        ####
        # Noise correlation calculation
        tmp = np.zeros(sta_removed.shape)
        tmp[:,preStimStart:postStimEnd,trial] = stats.zscore(sta_removed[:,preStimStart:postStimEnd,trial],axis=1)
        #tmp[:,preStimStart:postStimEnd,trial] = sta_removed[:,preStimStart:postStimEnd,trial]
        A = np.corrcoef(tmp[:,preStimStart:preStimEnd,trial])
        np.fill_diagonal(A,0)
        
        icount = 0
        for i in df.area.index:
            try:
                ind_i = regions.index(df.area[i])
            except:
                continue
            jcount = 0
            for j in df.area.index:
                try:
                    ind_j = regions.index(df.area[j])
                except:
                    continue
                preStimNoiseCorr[ind_i,ind_j,trial] = np.arctanh(A[icount,jcount])
                
                jcount += 1
            icount += 1
                
                
        preCorr_avg[trial] = np.nanmean(np.arctanh(A))
        dimensionalityPre[trial] = getDimensionality(np.cov(sta_removed[:,preStimStart:preStimEnd,trial]))

        A = np.corrcoef(tmp[:,postStimStart:postStimEnd,trial])
        np.fill_diagonal(A,0)
        
        icount = 0
        for i in df.area.index:
            try:
                ind_i = regions.index(df.area[i])
            except:
                continue
            jcount = 0
            for j in df.area.index:
                try:
                    ind_j = regions.index(df.area[j])
                except:
                    continue
                postStimNoiseCorr[ind_i,ind_j,trial] = np.arctanh(A[icount,jcount])
                
                jcount += 1
            icount += 1
            
        postCorr_avg[trial] = np.nanmean(np.arctanh(A))
        dimensionalityPost[trial] = getDimensionality(np.cov(sta_removed[:,postStimStart:postStimEnd,trial]))

        # SD calculation
        icount = 0
        for i in df.area.index:
            try:
                ind_i = regions.index(df.area[i])
            except:
                continue            
            # Time series is normalized already so variance = std
            preStimNoiseSD[ind_i,trial] = np.var(tmp[icount,preStimStart:preStimEnd,trial],axis=0)
            postStimNoiseSD[ind_i,trial] = np.var(tmp[icount,postStimStart:postStimEnd,trial],axis=0)

            icount += 1

        # FR calculation
        icount = 0
        for i in df.area.index:
            try:
                ind_i = regions.index(df.area[i])
            except:
                continue            
            preStimFR[ind_i,trial] = np.mean(tmp[icount,preStimStart:preStimEnd,trial],axis=0)
            postStimFR[ind_i,trial] = np.mean(tmp[icount,postStimStart:postStimEnd,trial],axis=0)
            icount += 1

    badTrials = np.asarray(badTrials)
    preStimNoiseCorr = np.delete(preStimNoiseCorr,badTrials,axis=2)
    postStimNoiseCorr = np.delete(postStimNoiseCorr,badTrials,axis=2)
    preCorr_avg = np.delete(preCorr_avg,badTrials,axis=0)
    postCorr_avg = np.delete(postCorr_avg,badTrials,axis=0)
    preStimNoiseSD = np.delete(preStimNoiseSD,badTrials,axis=1)
    postStimNoiseSD = np.delete(postStimNoiseSD,badTrials,axis=1)
    preStimFR = np.delete(preStimFR,badTrials,axis=1)
    postStimFR = np.delete(postStimFR,badTrials,axis=1)
    dimensionalityPre = np.delete(dimensionalityPre,badTrials,axis=0)
    dimensionalityPost = np.delete(dimensionalityPost,badTrials,axis=0)
    
    dimReplications = {}
    dimReplications['avg_post'] = dimensionalityPost
    dimReplications['avg_pre'] = dimensionalityPre

    spkCorrReplications = {}
    spkCorrReplications['avg_post'] = postStimNoiseCorr
    spkCorrReplications['avg_pre'] = preStimNoiseCorr
    
    sdReplications = {}
    sdReplications['avg_post'] = postStimNoiseSD
    sdReplications['avg_pre'] = preStimNoiseSD
    
    frReplications = {}
    frReplications['avg_post'] = postStimFR
    frReplications['avg_pre'] = preStimFR


#     print 'Total number of bad Trials:', badTrial1 + badTrial2, '/', nTrials
#     print '\tNumber of Bad Trials 1', badTrial1
#     print '\tNumber of Bad Trials 2', badTrial2

    return dimReplications, spkCorrReplications, sdReplications, frReplications
        
def computeCrossTrialStatistics2(df,sta_removed,normalized=True):
    tmin = -4000
    tmax = 4000
    time = np.linspace(tmin,tmax,sta_removed.shape[1])

    minTPs = 25
    trialBins = 25
    
    # Basic parameters
    nCells = sta_removed.shape[0]
    nTrials = sta_removed.shape[2]

    session_ind = df.area.index[0]
    
    crossTrialAct_rest = np.zeros((len(regions),nTrials))
    crossTrialAct_task = np.zeros((len(regions),nTrials))

    badTrials = []
    badTrial1 = 0
    badTrial2 = 0
    for trial in range(nTrials):
        # First identify the beginning of recording (prior to taskStart)
        preStimStart = np.min(np.where(sta_removed[0,:,trial]!=0)[0])
        try:
#             preStimEnd = np.max(np.where(time<(df['taskInfo'][session_ind]['fixptOn'][trial])*1000)[0]) # Convert trial start times to ms
            preStimEnd = np.max(np.where(time<(df['mocolInfo'][session_ind]['fixptOn'][trial])*1000)[0]) # Convert trial start times to ms
        except:
            badTrials.append(trial)
            badTrial1 += 1
            continue
        nTPs = preStimEnd - preStimStart
        if nTPs<minTPs:
            badTrials.append(trial)
            badTrial2 += 1
            continue
        # Make sure ITI period doesn't exceed 1s prior to fixation period
        elif nTPs>100:
            preStimStart = preStimEnd-100

#         postStimStart = np.min(np.where(time>=0)[0])
        postStimStart = np.min(np.where(time>=df['mocolInfo'][session_ind]['cueOn'][trial]*1000)[0])
        postStimEnd = postStimStart + nTPs
        
        # Normalize across time points
        tmp = np.zeros(sta_removed.shape)
        if normalized: 
            tmp[:,preStimStart:postStimEnd,trial] = stats.zscore(sta_removed[:,preStimStart:postStimEnd,trial],axis=1)
        else: 
            tmp[:,preStimStart:postStimEnd,trial] = sta_removed[:,preStimStart:postStimEnd,trial]

        # FR calculation
        icount = 0
        for i in df.area.index:
            try:
                ind_i = regions.index(df.area[i])
            except:
                continue            
            crossTrialAct_rest[ind_i,trial] = np.mean(tmp[icount,preStimStart:preStimEnd,trial],axis=0)
            crossTrialAct_task[ind_i,trial] = np.mean(tmp[icount,postStimStart:postStimEnd,trial],axis=0)
            icount += 1

    badTrials = np.asarray(badTrials)

    crossTrialAct_rest = np.delete(crossTrialAct_rest,badTrials,axis=1)
    crossTrialAct_task = np.delete(crossTrialAct_task,badTrials,axis=1)

    var_rest = []
    var_task = []
    rsc_rest = []
    rsc_task = []
    dim_rest = []
    dim_task = []
    fr_rest = []
    fr_task = []
    i = 0
    #n_trials_per_stat = crossTrialAct_rest.shape[1]
    n_trials_per_stat = trialBins
    while (i+n_trials_per_stat)<=crossTrialAct_rest.shape[1]:
        var_rest.append(np.var(crossTrialAct_rest[:,i:(i+n_trials_per_stat)],axis=1))
        var_task.append(np.var(crossTrialAct_task[:,i:(i+n_trials_per_stat)],axis=1))

        if normalized:
            rsc_rest.append(np.corrcoef(crossTrialAct_rest[:,i:(i+n_trials_per_stat)]))
            rsc_task.append(np.corrcoef(crossTrialAct_task[:,i:(i+n_trials_per_stat)]))
        else:
            rsc_rest.append(np.cov(crossTrialAct_rest[:,i:(i+n_trials_per_stat)]))
            rsc_task.append(np.cov(crossTrialAct_task[:,i:(i+n_trials_per_stat)]))
        
#        print crossTrialAct_rest[:,i:(i+n_trials_per_stat)].shape
#        print crossTrialAct_rest[:,i:(i+n_trials_per_stat)]
        cov_rest = np.cov(crossTrialAct_rest[:,i:(i+n_trials_per_stat)])
        cov_task = np.cov(crossTrialAct_task[:,i:(i+n_trials_per_stat)])

        ind = np.isnan(cov_rest)
        cov_rest[ind] = 0
        ind = np.isnan(cov_task)
        cov_task[ind] = 0

        dim_rest.append(getDimensionality(cov_rest))
        dim_task.append(getDimensionality(cov_task))

        fr_rest.append(np.mean(crossTrialAct_rest[:,i:(i+n_trials_per_stat)],axis=1))
        fr_task.append(np.mean(crossTrialAct_task[:,i:(i+n_trials_per_stat)],axis=1))
        i += n_trials_per_stat
    
    dimReplications = {}
    dimReplications['avg_post'] = np.asarray(dim_task)
    dimReplications['avg_pre'] = np.asarray(dim_rest)

    spkCorrReplications = {}
    spkCorrReplications['avg_post'] = np.asarray(rsc_task)
    spkCorrReplications['avg_pre'] = np.asarray(rsc_rest)
    
    sdReplications = {}
    sdReplications['avg_post'] = np.asarray(var_task) 
    sdReplications['avg_pre'] = np.asarray(var_rest)
    
    frReplications = {}
    frReplications['avg_post'] = np.asarray(fr_task) 
    frReplications['avg_pre'] = np.asarray(fr_rest)

    return dimReplications, spkCorrReplications, sdReplications, frReplications

def computeInformationTheoreticStats(df,sta_removed,normalized=True):
    tmin = -4000
    tmax = 4000
    time = np.linspace(tmin,tmax,sta_removed.shape[1])

    minTPs = 25
    trialBins = 25
    
    # Basic parameters
    nCells = sta_removed.shape[0]
    nTrials = sta_removed.shape[2]

    session_ind = df.area.index[0]
    
    crossTrialAct_rest = np.zeros((len(regions),nTrials))
    crossTrialAct_task = np.zeros((len(regions),nTrials))

    badTrials = []
    badTrial1 = 0
    badTrial2 = 0
    for trial in range(nTrials):
        # First identify the beginning of recording (prior to taskStart)
        preStimStart = np.min(np.where(sta_removed[0,:,trial]!=0)[0])
        try:
#             preStimEnd = np.max(np.where(time<(df['taskInfo'][session_ind]['fixptOn'][trial])*1000)[0]) # Convert trial start times to ms
            preStimEnd = np.max(np.where(time<(df['mocolInfo'][session_ind]['fixptOn'][trial])*1000)[0]) # Convert trial start times to ms
        except:
            badTrials.append(trial)
            badTrial1 += 1
            continue
        nTPs = preStimEnd - preStimStart
        if nTPs<minTPs:
            badTrials.append(trial)
            badTrial2 += 1
            continue
        # Make sure ITI period doesn't exceed 1s prior to fixation period
        elif nTPs>100:
            preStimStart = preStimEnd-100

#         postStimStart = np.min(np.where(time>=0)[0])
        postStimStart = np.min(np.where(time>=df['mocolInfo'][session_ind]['cueOn'][trial]*1000)[0])
        postStimEnd = postStimStart + nTPs
        
        # Normalize across time points
        tmp = np.zeros(sta_removed.shape)
        if normalized: 
            tmp[:,preStimStart:postStimEnd,trial] = stats.zscore(sta_removed[:,preStimStart:postStimEnd,trial],axis=1)
        else: 
            tmp[:,preStimStart:postStimEnd,trial] = sta_removed[:,preStimStart:postStimEnd,trial]

        # FR calculation
        icount = 0
        for i in df.area.index:
            try:
                ind_i = regions.index(df.area[i])
            except:
                continue            
            crossTrialAct_rest[ind_i,trial] = np.mean(tmp[icount,preStimStart:preStimEnd,trial],axis=0)
            crossTrialAct_task[ind_i,trial] = np.mean(tmp[icount,postStimStart:postStimEnd,trial],axis=0)
            icount += 1

    badTrials = np.asarray(badTrials)

    crossTrialAct_rest = np.delete(crossTrialAct_rest,badTrials,axis=1)
    crossTrialAct_task = np.delete(crossTrialAct_task,badTrials,axis=1)

    entropy_rest = []
    entropy_task = []
    mi_rest = []
    mi_task = []
    jointent_rest = []
    jointent_task = []
    i = 0
    #n_trials_per_stat = crossTrialAct_rest.shape[1]
    n_trials_per_stat = trialBins
    while (i+n_trials_per_stat)<=crossTrialAct_rest.shape[1]:
        tmp_ent_rest = np.zeros((crossTrialAct_rest.shape[0],))
        tmp_ent_task = np.zeros((crossTrialAct_task.shape[0],))
        for region in range(crossTrialAct_rest.shape[0]):
            tmp_ent_rest[region] = stats.entropy(np.histogram(crossTrialAct_rest[region,i:(i+n_trials_per_stat)],bins=10)[0])
            tmp_ent_task[region] = stats.entropy(np.histogram(crossTrialAct_task[region,i:(i+n_trials_per_stat)],bins=10)[0])
        entropy_rest.append(tmp_ent_rest)
        entropy_task.append(tmp_ent_task)

    
        ## Calculate MI and Joint Entropy
        tmp_mi_task = np.zeros((crossTrialAct_rest.shape[0],crossTrialAct_rest.shape[0]))
        tmp_mi_rest = np.zeros((crossTrialAct_rest.shape[0],crossTrialAct_rest.shape[0]))
        tmp_jointent_task = np.zeros(tmp_mi_task.shape)
        tmp_jointent_rest = np.zeros(tmp_mi_rest.shape)
        for a in range(crossTrialAct_rest.shape[0]):
            for b in range(crossTrialAct_rest.shape[0]):
                if a>=b: continue
                tmp_mi_rest[a,b] = calc_MI((crossTrialAct_rest[a,i:(i+n_trials_per_stat)],crossTrialAct_rest[b,i:(i+n_trials_per_stat)],10))
                tmp_mi_task[a,b] = calc_MI((crossTrialAct_task[a,i:(i+n_trials_per_stat)],crossTrialAct_task[b,i:(i+n_trials_per_stat)],10))
                
                ## rest entropy
                #tmp_ent_rest_a = stats.entropy(np.histogram(crossTrialAct_rest[a,i:(i+n_trials_per_stat)],bins=10)[0])
                #tmp_ent_rest_b = stats.entropy(np.histogram(crossTrialAct_rest[b,i:(i+n_trials_per_stat)],bins=10)[0])
                ## task entropy
                #tmp_ent_task_a = stats.entropy(np.histogram(crossTrialAct_task[a,i:(i+n_trials_per_stat)],bins=10)[0])
                #tmp_ent_task_b = stats.entropy(np.histogram(crossTrialAct_task[b,i:(i+n_trials_per_stat)],bins=10)[0])

                tmp_jointent_rest[a,b] = tmp_ent_rest[a] + tmp_ent_rest[b] - tmp_mi_rest[a,b]
                tmp_jointent_task[a,b] = tmp_ent_task[a] + tmp_ent_task[b] - tmp_mi_task[a,b]
                #tmp_jointent_rest[a,b] = stats.entropy(np.histogram(crossTrialAct_rest[a,i:(i+n_trials_per_stat)])[0],qk=np.histogram(crossTrialAct_rest[b,i:(i+n_trials_per_stat)])[0])
                #tmp_jointent_task[a,b] = stats.entropy(np.histogram(crossTrialAct_task[a,i:(i+n_trials_per_stat)])[0],qk=np.histogram(crossTrialAct_task[b,i:(i+n_trials_per_stat)])[0])

        # only performed on lower triangle, so add transpose 
        mi_rest.append(tmp_mi_rest + tmp_mi_rest.T)
        mi_task.append(tmp_mi_task + tmp_mi_task.T)

        jointent_rest.append(tmp_jointent_rest + tmp_jointent_rest.T)
        jointent_task.append(tmp_jointent_task + tmp_jointent_task.T)

        i += n_trials_per_stat

    #mi_rest = tmp_mi_rest + tmp_mi_rest.T
    #mi_task = tmp_mi_task + tmp_mi_task.T


    #jointent_rest = tmp_jointent_rest + tmp_jointent_rest.T
    #jointent_task = tmp_jointent_task + tmp_jointent_task.T


    entropyReplications = {}
    entropyReplications['avg_post'] = np.asarray(entropy_task) 
    entropyReplications['avg_pre'] = np.asarray(entropy_rest)

    jointEntropyReplications = {}
    jointEntropyReplications['avg_post'] = np.asarray(jointent_task) 
    jointEntropyReplications['avg_pre'] = np.asarray(jointent_rest)
    
    miReplications = {}
    miReplications['avg_post'] = np.asarray(mi_task) 
    miReplications['avg_pre'] = np.asarray(mi_rest)
    
    return entropyReplications, jointEntropyReplications, miReplications

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

from sklearn.metrics import mutual_info_score
def calc_MI((x, y, bins)):
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi

def computeCrossTrialStatistics4(df,sta_removed,sta,normalize=False):
    tmin = -4000
    tmax = 4000
    time = np.linspace(tmin,tmax,sta_removed.shape[1])

    minTPs = 25
    trialBins = 25
    
    # Basic parameters
    nCells = sta_removed.shape[0]
    nTrials = sta_removed.shape[2]

    session_ind = df.area.index[0]
    
    crossTrialAct_rest = np.zeros((len(regions),nTrials))
    crossTrialAct_task = np.zeros((len(regions),nTrials))

    badTrials = []
    for trial in range(nTrials):
        # make sure there is data in this trial
        if np.mean(sta[0,:,trial])==0:
            badTrials.append(trial)
        # First identify the beginning of recording (prior to taskStart)
        preStimStart = np.min(np.where(sta_removed[0,:,trial]!=0)[0])
        try:
            preStimEnd = np.max(np.where(time<(df['mocolInfo'][session_ind]['fixptOn'][trial])*1000)[0]) # Convert trial start times to ms
        except:
            badTrials.append(trial)
            continue
        nTPs = preStimEnd - preStimStart
        if nTPs<minTPs:
            badTrials.append(trial)
            continue
        # Make sure ITI period doesn't exceed 1s prior to fixation period
        elif nTPs>100:
            preStimEnd = preStimStart+100
            #preStimStart = preStimEnd-100
            nTPs = preStimEnd-preStimStart

        postStimStart = np.min(np.where(time>=df['mocolInfo'][session_ind]['cueOn'][trial]*1000)[0])
        postStimEnd = postStimStart + nTPs

        #preStimEnd = preStimStart + minTPs
        #postStimEnd = postStimStart + minTPs

        tmp = np.zeros(sta_removed.shape)
        if normalize:
            tmp[:,preStimStart:postStimEnd,trial] = stats.zscore(sta_removed[:,preStimStart:postStimEnd,trial],axis=1)
        else:
            tmp[:,preStimStart:postStimEnd,trial] = sta_removed[:,preStimStart:postStimEnd,trial]

        # FR calculation
        icount = 0
        for i in df.area.index:
            try:
                ind_i = regions.index(df.area[i])
            except:
                continue            
            crossTrialAct_rest[ind_i,trial] = np.mean(sta_removed[icount,preStimStart:preStimEnd,trial],axis=0)
            crossTrialAct_task[ind_i,trial] = np.mean(sta_removed[icount,postStimStart:postStimEnd,trial],axis=0)
            icount += 1

    badTrials = np.asarray(badTrials)

    crossTrialAct_rest = np.delete(crossTrialAct_rest,badTrials,axis=1)
    crossTrialAct_task = np.delete(crossTrialAct_task,badTrials,axis=1)

    var_rest = []
    var_task = []
    rsc_rest = []
    rsc_task = []
    cov_rest = []
    cov_task = []
    dim_rest = []
    dim_task = []
    fr_rest = []
    fr_task = []
    i = 0
    #n_trials_per_stat = crossTrialAct_rest.shape[1]
    n_trials_per_stat = trialBins
    while (i+n_trials_per_stat)<=crossTrialAct_rest.shape[1]:
        var_rest.append(np.var(crossTrialAct_rest[:,i:(i+n_trials_per_stat)],axis=1))
        var_task.append(np.var(crossTrialAct_task[:,i:(i+n_trials_per_stat)],axis=1))
        #var_rest.append(np.divide(np.var(crossTrialAct_rest[:,i:(i+n_trials_per_stat)],axis=1),np.mean(crossTrialAct_rest[:,i:(i+n_trials_per_stat)],axis=1)))
        #var_task.append(np.divide(np.var(crossTrialAct_task[:,i:(i+n_trials_per_stat)],axis=1),np.mean(crossTrialAct_task[:,i:(i+n_trials_per_stat)],axis=1)))

        rsc_rest.append(np.corrcoef(crossTrialAct_rest[:,i:(i+n_trials_per_stat)]))
        rsc_task.append(np.corrcoef(crossTrialAct_task[:,i:(i+n_trials_per_stat)]))
        
#        print crossTrialAct_rest[:,i:(i+n_trials_per_stat)].shape
#        print crossTrialAct_rest[:,i:(i+n_trials_per_stat)]
        tmp_rest = np.cov(crossTrialAct_rest[:,i:(i+n_trials_per_stat)])
        tmp_task = np.cov(crossTrialAct_task[:,i:(i+n_trials_per_stat)])

        ind = np.isnan(tmp_rest)
        tmp_rest[ind] = 0
        ind = np.isnan(tmp_task)
        tmp_task[ind] = 0

        cov_rest.append(tmp_rest)
        cov_task.append(tmp_task)

        dim_rest.append(getDimensionality(tmp_rest))
        dim_task.append(getDimensionality(tmp_task))

        fr_rest.append(np.mean(crossTrialAct_rest[:,i:(i+n_trials_per_stat)],axis=1))
        fr_task.append(np.mean(crossTrialAct_task[:,i:(i+n_trials_per_stat)],axis=1))
        i += n_trials_per_stat
    
    dimReplications = {}
    dimReplications['avg_post'] = np.asarray(dim_task)
    dimReplications['avg_pre'] = np.asarray(dim_rest)

    spkCorrReplications = {}
    spkCorrReplications['avg_post'] = np.asarray(rsc_task)
    spkCorrReplications['avg_pre'] = np.asarray(rsc_rest)

    spkCovReplications = {}
    spkCovReplications['avg_post'] = np.asarray(cov_task)
    spkCovReplications['avg_pre'] = np.asarray(cov_rest)
    
    sdReplications = {}
    sdReplications['avg_post'] = np.asarray(var_task) 
    sdReplications['avg_pre'] = np.asarray(var_rest)
    
    frReplications = {}
    frReplications['avg_post'] = np.asarray(fr_task) 
    frReplications['avg_pre'] = np.asarray(fr_rest)

    return dimReplications, spkCorrReplications, spkCovReplications, sdReplications, frReplications


def computeInformationTheoreticStats2(df,sta_removed,sta,normalize=True):
    tmin = -4000
    tmax = 4000
    time = np.linspace(tmin,tmax,sta_removed.shape[1])

    minTPs = 25
    trialBins = 25
    
    # Basic parameters
    nCells = sta_removed.shape[0]
    nTrials = sta_removed.shape[2]

    session_ind = df.area.index[0]
    
    crossTrialAct_rest = np.zeros((len(regions),nTrials))
    crossTrialAct_task = np.zeros((len(regions),nTrials))

    badTrials = []
    for trial in range(nTrials):
        # make sure there is data in this trial
        if np.mean(sta[0,:,trial])==0:
            badTrials.append(trial)
        # First identify the beginning of recording (prior to taskStart)
        preStimStart = np.min(np.where(sta_removed[0,:,trial]!=0)[0])
        try:
            preStimEnd = np.max(np.where(time<(df['mocolInfo'][session_ind]['fixptOn'][trial])*1000)[0]) # Convert trial start times to ms
        except:
            badTrials.append(trial)
            continue
        nTPs = preStimEnd - preStimStart
        if nTPs<minTPs:
            badTrials.append(trial)
            continue
        # Make sure ITI period doesn't exceed 1s prior to fixation period
        elif nTPs>100:
            preStimEnd = preStimStart+100
            #preStimStart = preStimEnd-100
            nTPs = preStimEnd-preStimStart

        postStimStart = np.min(np.where(time>=df['mocolInfo'][session_ind]['cueOn'][trial]*1000)[0])
        postStimEnd = postStimStart + nTPs

        #preStimEnd = preStimStart + minTPs
        #postStimEnd = postStimStart + minTPs

        tmp = np.zeros(sta_removed.shape)
        if normalize:
            tmp[:,preStimStart:postStimEnd,trial] = stats.zscore(sta_removed[:,preStimStart:postStimEnd,trial],axis=1)
        else:
            tmp[:,preStimStart:postStimEnd,trial] = sta_removed[:,preStimStart:postStimEnd,trial]

        # FR calculation
        icount = 0
        for i in df.area.index:
            try:
                ind_i = regions.index(df.area[i])
            except:
                continue            
            crossTrialAct_rest[ind_i,trial] = np.mean(sta_removed[icount,preStimStart:preStimEnd,trial],axis=0)
            crossTrialAct_task[ind_i,trial] = np.mean(sta_removed[icount,postStimStart:postStimEnd,trial],axis=0)
            icount += 1

    badTrials = np.asarray(badTrials)

    crossTrialAct_rest = np.delete(crossTrialAct_rest,badTrials,axis=1)
    crossTrialAct_task = np.delete(crossTrialAct_task,badTrials,axis=1)

    entropy_rest = []
    entropy_task = []
    mi_rest = []
    mi_task = []
    jointent_rest = []
    jointent_task = []
    i = 0
    #n_trials_per_stat = crossTrialAct_rest.shape[1]
    n_trials_per_stat = trialBins
    while (i+n_trials_per_stat)<=crossTrialAct_rest.shape[1]:
        tmp_ent_rest = np.zeros((crossTrialAct_rest.shape[0],))
        tmp_ent_task = np.zeros((crossTrialAct_task.shape[0],))
        for region in range(crossTrialAct_rest.shape[0]):
            tmp_ent_rest[region] = stats.entropy(np.histogram(crossTrialAct_rest[region,i:(i+n_trials_per_stat)],bins=8)[0])
            tmp_ent_task[region] = stats.entropy(np.histogram(crossTrialAct_task[region,i:(i+n_trials_per_stat)],bins=8)[0])
        entropy_rest.append(tmp_ent_rest)
        entropy_task.append(tmp_ent_task)

    
        ## Calculate MI and Joint Entropy
        tmp_mi_task = np.zeros((crossTrialAct_rest.shape[0],crossTrialAct_rest.shape[0]))
        tmp_mi_rest = np.zeros((crossTrialAct_rest.shape[0],crossTrialAct_rest.shape[0]))
        tmp_jointent_task = np.zeros(tmp_mi_task.shape)
        tmp_jointent_rest = np.zeros(tmp_mi_rest.shape)
        for a in range(crossTrialAct_rest.shape[0]):
            for b in range(crossTrialAct_rest.shape[0]):
                if a>=b: continue
                tmp_mi_rest[a,b] = calc_MI((crossTrialAct_rest[a,i:(i+n_trials_per_stat)],crossTrialAct_rest[b,i:(i+n_trials_per_stat)],8))
                tmp_mi_task[a,b] = calc_MI((crossTrialAct_task[a,i:(i+n_trials_per_stat)],crossTrialAct_task[b,i:(i+n_trials_per_stat)],8))
                
                ## rest entropy
                #tmp_ent_rest_a = stats.entropy(np.histogram(crossTrialAct_rest[a,i:(i+n_trials_per_stat)],bins=10)[0])
                #tmp_ent_rest_b = stats.entropy(np.histogram(crossTrialAct_rest[b,i:(i+n_trials_per_stat)],bins=10)[0])
                ## task entropy
                #tmp_ent_task_a = stats.entropy(np.histogram(crossTrialAct_task[a,i:(i+n_trials_per_stat)],bins=10)[0])
                #tmp_ent_task_b = stats.entropy(np.histogram(crossTrialAct_task[b,i:(i+n_trials_per_stat)],bins=10)[0])

                tmp_jointent_rest[a,b] = tmp_ent_rest[a] + tmp_ent_rest[b] - tmp_mi_rest[a,b]
                tmp_jointent_task[a,b] = tmp_ent_task[a] + tmp_ent_task[b] - tmp_mi_task[a,b]
                #tmp_jointent_rest[a,b] = stats.entropy(np.histogram(crossTrialAct_rest[a,i:(i+n_trials_per_stat)])[0],qk=np.histogram(crossTrialAct_rest[b,i:(i+n_trials_per_stat)])[0])
                #tmp_jointent_task[a,b] = stats.entropy(np.histogram(crossTrialAct_task[a,i:(i+n_trials_per_stat)])[0],qk=np.histogram(crossTrialAct_task[b,i:(i+n_trials_per_stat)])[0])

        # only performed on lower triangle, so add transpose 
        mi_rest.append(tmp_mi_rest + tmp_mi_rest.T)
        mi_task.append(tmp_mi_task + tmp_mi_task.T)

        jointent_rest.append(tmp_jointent_rest + tmp_jointent_rest.T)
        jointent_task.append(tmp_jointent_task + tmp_jointent_task.T)

        i += n_trials_per_stat

    #mi_rest = tmp_mi_rest + tmp_mi_rest.T
    #mi_task = tmp_mi_task + tmp_mi_task.T


    #jointent_rest = tmp_jointent_rest + tmp_jointent_rest.T
    #jointent_task = tmp_jointent_task + tmp_jointent_task.T


    entropyReplications = {}
    entropyReplications['avg_post'] = np.asarray(entropy_task) 
    entropyReplications['avg_pre'] = np.asarray(entropy_rest)

    jointEntropyReplications = {}
    jointEntropyReplications['avg_post'] = np.asarray(jointent_task) 
    jointEntropyReplications['avg_pre'] = np.asarray(jointent_rest)
    
    miReplications = {}
    miReplications['avg_post'] = np.asarray(mi_task) 
    miReplications['avg_pre'] = np.asarray(mi_rest)
    
    return entropyReplications, jointEntropyReplications, miReplications

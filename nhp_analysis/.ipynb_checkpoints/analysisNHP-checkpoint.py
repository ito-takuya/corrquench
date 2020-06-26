# Takuya Ito
# Module for the core functions to run analysis on NHP data
# 01/30/2019

import numpy as np
import multiprocessing as mp
import scipy.stats as stats
import pandas as pd
import os
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
    for i in df.area.index:
        tmpmat = df['sta_removed'][i].copy()
        sta_removed.append(tmpmat)
    sta_removed = np.asarray(sta_removed)
        
    return sta_removed
    
def computeStatistics(df,sta_removed):
    tmin = -4000
    tmax = 4000
    time = np.linspace(tmin,tmax,sta_removed.shape[1])

    minTPs = 25
    
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
        

def computeCrossTrialStatistics(df,sta_removed):
    tmin = -4000
    tmax = 4000
    time = np.linspace(tmin,tmax,sta_removed.shape[1])

    minTPs = 25
    
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

    
    dimReplications = {}
    dimReplications['avg_post'] = getDimensionality(np.cov(crossTrialAct_task))
    dimReplications['avg_pre'] = getDimensionality(np.cov(crossTrialAct_rest))

    spkCorrReplications = {}
    spkCorrReplications['avg_post'] = np.corrcoef(crossTrialAct_task)
    spkCorrReplications['avg_pre'] = np.corrcoef(crossTrialAct_rest)
    
    sdReplications = {}
    sdReplications['avg_post'] = np.var(crossTrialAct_task,axis=1)
    sdReplications['avg_pre'] = np.var(crossTrialAct_rest,axis=1)
    
    frReplications = {}
    frReplications['avg_post'] = np.mean(crossTrialAct_task,axis=1)
    frReplications['avg_pre'] = np.mean(crossTrialAct_rest,axis=1)

    return dimReplications, spkCorrReplications, sdReplications, frReplications


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

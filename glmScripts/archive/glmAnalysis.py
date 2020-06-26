# Takuya Ito
# 03/28/2018

# Functions to run a GLM analysis

import numpy as np
import os
import glob
from nipy.modalities.fmri.hemodynamic_models import spm_hrf
import multiprocessing as mp
import statsmodels.api as sm


def GlasserGLM_taskdata_HCP(subj, gsr=True, nproc=8, glmtype='canonical'):
    """
    This function runs a GLM on the Glasser Parcels (360) on a single subject
    Will only regress out noise parameters and HRF convolved miniblock onsets

    Input parameters:
        subj = subject number as a string
        gsr = Runs GSR if True, no if False
        nproc = number of processes to use via multiprocessing
    """

    datadir = '/projects3/NetworkDiversity/data/rawPTSeries_Glasser_HCP/'
    framesToSkip = 5
    temporalMask = []

    taskRuns = ['tfMRI_EMOTION', 'tfMRI_GAMBLING','tfMRI_LANGUAGE','tfMRI_MOTOR','tfMRI_RELATIONAL','tfMRI_SOCIAL','tfMRI_WM']
    nTasks = len(taskRuns)
    data = []
    for t in range(nTasks):
        for run in range(2):
            rawfile = datadir + subj + '_' + taskRuns[t] + str(run+1) + '_LR.csv'
            tmp = np.loadtxt(rawfile,delimiter=',')

            tMask = np.ones((tmp.shape[1],))
            tMask[:framesToSkip] = 0
            
            temporalMask.extend(tMask)
            
            # Demean each run
            tMask = np.asarray(tMask,dtype=bool)
            runmean = np.mean(tmp[:,tMask],axis=1)
            runmean.shape = (len(runmean),1)
            tmp = tmp - runmean


            data.extend(tmp.T); 

    data = np.asarray(data).T
    temporalMask = np.asarray(temporalMask,dtype=bool)

    nROIs = data.shape[0]

    # Load regressors for data
    if glmtype=='FIR':
        X = loadStimFiles_task_HCP_withFIR(subj,gsr=gsr);
    elif glmtype=='canonical':
        X = loadStimFiles_task_HCP(subj,gsr=gsr);
    elif glmtype=='byBlock':
        X = loadStimFiles_task_HCPbyBlock(subj,gsr=gsr)

    try:
        taskRegs = X['allRegressors'][temporalMask,:] # These include the two binary regressors

        inputs = []
        for roi in range(nROIs):
            inputs.append((data[roi,temporalMask], taskRegs))

        pool = mp.Pool(processes=nproc)
        results = pool.map_async(_regression,inputs).get()
        pool.close()
        pool.join()

        residual_ts = np.zeros((nROIs,np.sum(temporalMask)))
        betas = np.zeros((nROIs,taskRegs.shape[1]+1)) # All regressors, + constant regressors
        roi = 0
        for result in results:
            betas[roi,:] = result[0]
            residual_ts[roi,:] = result[1]
            roi += 1
    except np.linalg.LinAlgError as err:
        print 'Error occurred...'
        print 'Error:', err
        print 'Re-running nuisance and task regression separately'
        
        print 'Running nuisance regression'
        taskRegs = X['noiseRegressors'][temporalMask,:] # These include the two binary regressors

        inputs = []
        for roi in range(nROIs):
            inputs.append((data[roi,temporalMask], taskRegs))

        pool = mp.Pool(processes=nproc)
        results = pool.map_async(_regression,inputs).get()
        pool.close()
        pool.join()

        data_cleaned = np.zeros((nROIs,np.sum(temporalMask)))
        betas1 = np.zeros((nROIs,taskRegs.shape[1]+1)) # All regressors, + constant regressors
        roi = 0
        for result in results:
            betas1[roi,:] = result[0]
            data_cleaned[roi,:] = result[1]
            roi += 1

        print 'Running task regression'
        taskRegs = X['taskRegressors'][temporalMask,:] # These include the two binary regressors

        inputs = []
        for roi in range(nROIs):
            inputs.append((data_cleaned[roi,:], taskRegs))

        pool = mp.Pool(processes=nproc)
        results = pool.map_async(_regression,inputs).get()
        pool.close()
        pool.join()

        residual_ts = np.zeros((nROIs,np.sum(temporalMask)))
        betas2 = np.zeros((nROIs,taskRegs.shape[1])) # All regressors; exclude constant regressor since it was already demeaned previously
        roi = 0
        for result in results:
            #print 'result[0] shape' result[0].shape
            #print 'betas2 shape' result[0].shape
            betas2[roi,:] = result[0][1:] # exclude constant regressor
            residual_ts[roi,:] = result[1]
            roi += 1

        betas = np.hstack((betas1,betas2))
        


    if gsr:
        outname1 = '/projects3/TaskFCMech/data/results/EmpiricalValidation_HCP/' + glmtype + 'GLM/' + subj + '_task_nuisanceResids_Glasser_GSR.csv'
        outname2 = '/projects3/TaskFCMech/data/results/EmpiricalValidation_HCP/' + glmtype + 'GLM/' + subj + '_task_taskBetas_Glasser_GSR.csv'
        outname3 = '/projects3/TaskFCMech/data/results/EmpiricalValidation_HCP/' + glmtype + 'GLM/' + subj + '_task_nuisanceBetas_Glasser_GSR.csv'
    else:
        outname1 = '/projects3/TaskFCMech/data/results/EmpiricalValidation_HCP/' + glmtype + 'GLM/' + subj + '_task_nuisanceResids_Glasser_noGSR.csv'
        outname2 = '/projects3/TaskFCMech/data/results/EmpiricalValidation_HCP/' + glmtype + 'GLM/' + subj + '_task_taskBetas_Glasser_noGSR.csv'
        outname3 = '/projects3/TaskFCMech/data/results/EmpiricalValidation_HCP/' + glmtype + 'GLM/' + subj + '_task_nuisanceBetas_Glasser_noGSR.csv'
    
    outname4 = '/projects3/TaskFCMech/data/results/EmpiricalValidation_HCP/' + glmtype + 'GLM/' + subj + '_task_stimIndex.csv'

    nTaskRegressors = X['taskRegressors'].shape[1]
    nNoiseRegressors = X['noiseRegressors'].shape[1]
    task_betas = betas[:,-nTaskRegressors:]
    nuis_betas = betas[:,:nNoiseRegressors]
    np.savetxt(outname1, residual_ts, delimiter=',')
    np.savetxt(outname2, task_betas, delimiter=',')
    np.savetxt(outname3, nuis_betas, delimiter=',')
    np.savetxt(outname4, X['stimIndex'], delimiter=',')
    
    output = {}
    output['taskbetas'] = task_betas 
    output['nuisancebetas'] = nuis_betas
    output['residual_ts'] = residual_ts
    output['stimIndex'] = X['stimIndex']

    return output

def GlasserGLM_restdata_HCP_forTaskRestComparison(subj, gsr=True, nproc=8, glmtype='canonical'):
    """
    This function runs a GLM on the Glasser Parcels (360) on a single subject
    This function runs the same TASK FIR model on RESTING STATE DATA, allowing for equal comparison of resting-state v. task-state comparisons

    Input parameters:
        subj = subject number as a string
        gsr = Runs GSR if True, no if False
        nproc = number of processes to use via multiprocessing
    """

    datadir = '/projects3/NetworkDiversity/data/hcppreprocessed/' + subj + '/' 
    data = []
    for run in range(1,5):
        data.extend(np.loadtxt(datadir + subj + '_rest' + str(run) + '_nuisanceResids_Glasser_GSR.csv',delimiter=',').T)
    data = np.asarray(data).T

    nROIs = data.shape[0]

    # Load regressors for data
    if glmtype=='FIR':
        X = loadStimFiles_task_HCP_withFIR(subj,gsr=gsr);
    elif glmtype=='canonical':
        X = loadStimFiles_task_HCP(subj,gsr=gsr);
    elif glmtype=='byBlock':
        X = loadStimFiles_task_HCPbyBlock(subj,gsr=gsr)
    temporalMask = X['tMask']

#    try:
#        taskRegs = X['allRegressors'][temporalMask,:] # These include the two binary regressors
#
#        inputs = []
#        for roi in range(nROIs):
#            inputs.append((data[roi,temporalMask], taskRegs))
#
#        pool = mp.Pool(processes=nproc)
#        results = pool.map_async(_regression,inputs).get()
#        pool.close()
#        pool.join()
#
#        residual_ts = np.zeros((nROIs,np.sum(temporalMask)))
#        betas = np.zeros((nROIs,taskRegs.shape[1]+1)) # All regressors, + constant regressors
#        roi = 0
#        for result in results:
#            betas[roi,:] = result[0]
#            residual_ts[roi,:] = result[1]
#            roi += 1
#    except np.linalg.LinAlgError as err:
#        print 'Error occurred...'
#        print 'Error:', err
#        print 'Re-running nuisance and task regression separately'
#        
#        print 'Running nuisance regression'
#        taskRegs = X['noiseRegressors'][temporalMask,:] # These include the two binary regressors
#
#        inputs = []
#        for roi in range(nROIs):
#            inputs.append((data[roi,temporalMask], taskRegs))
#
#        pool = mp.Pool(processes=nproc)
#        results = pool.map_async(_regression,inputs).get()
#        pool.close()
#        pool.join()
#
#        data_cleaned = np.zeros((nROIs,np.sum(temporalMask)))
#        betas1 = np.zeros((nROIs,taskRegs.shape[1]+1)) # All regressors, + constant regressors
#        roi = 0
#        for result in results:
#            betas1[roi,:] = result[0]
#            data_cleaned[roi,:] = result[1]
#            roi += 1
#
#        print 'Running task regression'
#        taskRegs = X['taskRegressors'][temporalMask,:] # These include the two binary regressors
#
#        inputs = []
#        for roi in range(nROIs):
#            inputs.append((data_cleaned[roi,:], taskRegs))
#
#        pool = mp.Pool(processes=nproc)
#        results = pool.map_async(_regression,inputs).get()
#        pool.close()
#        pool.join()
#
#        residual_ts = np.zeros((nROIs,np.sum(temporalMask)))
#        betas2 = np.zeros((nROIs,taskRegs.shape[1])) # All regressors; exclude constant regressor since it was already demeaned previously
#        roi = 0
#        for result in results:
#            #print 'result[0] shape' result[0].shape
#            #print 'betas2 shape' result[0].shape
#            betas2[roi,:] = result[0][1:] # exclude constant regressor
#            residual_ts[roi,:] = result[1]
#            roi += 1
#
#        betas = np.hstack((betas1,betas2))
        
    print 'Running task regression'
    taskRegs = X['taskRegressors'][temporalMask,:] # These include the two binary regressors

    t_ind = np.where(temporalMask==True)[0]
    inputs = []
    for roi in range(nROIs):
        inputs.append((data[roi,t_ind], taskRegs))

    pool = mp.Pool(processes=nproc)
    results = pool.map_async(_regression,inputs).get()
    pool.close()
    pool.join()

    residual_ts = np.zeros((nROIs,np.sum(temporalMask)))
    betas = np.zeros((nROIs,taskRegs.shape[1]+1)) # All regressors; exclude constant regressor since it was already demeaned previously
    roi = 0
    for result in results:
        #print 'result[0] shape' result[0].shape
        #print 'betas2 shape' result[0].shape
        betas[roi,:] = result[0]
        residual_ts[roi,:] = result[1]
        roi += 1

    if gsr:
        outname1 = '/projects3/TaskFCMech/data/results/EmpiricalValidation_HCP/' + glmtype + 'GLM/' + subj + '_restDataUsingTaskRegression_nuisanceResids_Glasser_GSR.csv'
        outname2 = '/projects3/TaskFCMech/data/results/EmpiricalValidation_HCP/' + glmtype + 'GLM/' + subj + '_restDataUsingTaskRegression_taskBetas_Glasser_GSR.csv'
        outname3 = '/projects3/TaskFCMech/data/results/EmpiricalValidation_HCP/' + glmtype + 'GLM/' + subj + '_restDataUsingTaskRegression_nuisanceBetas_Glasser_GSR.csv'
    else:
        outname1 = '/projects3/TaskFCMech/data/results/EmpiricalValidation_HCP/' + glmtype + 'GLM/' + subj + '_restDataUsingTaskRegression_nuisanceResids_Glasser_noGSR.csv'
        outname2 = '/projects3/TaskFCMech/data/results/EmpiricalValidation_HCP/' + glmtype + 'GLM/' + subj + '_restDataUsingTaskRegression_taskBetas_Glasser_noGSR.csv'
        outname3 = '/projects3/TaskFCMech/data/results/EmpiricalValidation_HCP/' + glmtype + 'GLM/' + subj + '_restDataUsingTaskRegression_nuisanceBetas_Glasser_noGSR.csv'
    
    outname4 = '/projects3/TaskFCMech/data/results/EmpiricalValidation_HCP/' + glmtype + 'GLM/' + subj + '_restDataUsingTaskRegression_stimIndex.csv'

    nTaskRegressors = X['taskRegressors'].shape[1]
    nNoiseRegressors = X['noiseRegressors'].shape[1]
    task_betas = betas[:,-nTaskRegressors:]
#    nuis_betas = betas[:,:nNoiseRegressors]
    np.savetxt(outname1, residual_ts, delimiter=',')
    np.savetxt(outname2, task_betas, delimiter=',')
#    np.savetxt(outname3, nuis_betas, delimiter=',')
    np.savetxt(outname4, X['stimIndex'], delimiter=',')
    
    output = {}
    output['taskbetas'] = task_betas 
#    output['nuisancebetas'] = nuis_betas
    output['residual_ts'] = residual_ts
    output['stimIndex'] = X['stimIndex']

    return output

def loadStimFiles_task_HCP(subj, gsr=True):
    basedir = '/projects3/NetworkDiversity/'
    datadir = basedir + 'data/hcppreprocessed/' + subj + '/nuisanceRegsTasks/'
    hcpdir = '/projects/ExternalDatasets/HCPData/data/minimalpreproc/' + subj + '/MNINonLinear/Results/'
    nMotionParams = 12
    trLength = .720
    totalTRs = 3880
    nCond = 24

    tasks = ['tfMRI_EMOTION_RL','tfMRI_EMOTION_LR','tfMRI_GAMBLING_RL','tfMRI_GAMBLING_LR','tfMRI_LANGUAGE_RL','tfMRI_LANGUAGE_LR','tfMRI_MOTOR_RL','tfMRI_MOTOR_LR','tfMRI_RELATIONAL_RL','tfMRI_RELATIONAL_LR','tfMRI_SOCIAL_RL','tfMRI_SOCIAL_LR','tfMRI_WM_RL','tfMRI_WM_LR']
    TRsPerRun = [176,176,253,253,316,316,284,284,232,232,274,274,405,405]

    timeseriesRegressors = np.zeros((totalTRs,6)) # WM, ventricle, and global signal regressors + their derivatives
    motionRegressors = np.zeros((totalTRs,12)) # 12 motion regressors
    stimMat = np.zeros((totalTRs,nCond))
    linearTrendRegressor = np.zeros((totalTRs,len(tasks)))
    demeanRegressor = np.zeros((totalTRs,1))

    numTaskStims = {}
    trcount = 0
    stimcount = 0
    taskkey = ''
    for i in range(len(tasks)):

        task = tasks[i]

        trstart = trcount
        trend = trcount + TRsPerRun[i]

        ## Create derivative time series
        print 'Creating derivative time series for ventricle, white matter, and whole brain signal for subject', subj
        
        os.system('1d_tool.py -overwrite -infile ' + datadir + subj + '_WM_timeseries_' + task + '.1D -derivative -write ' + datadir + subj + '_WM_timeseries_deriv_' + task + '.1D')
        os.system('1d_tool.py -overwrite -infile ' + datadir + subj + '_ventricles_timeseries_' + task + '.1D -derivative -write ' + datadir + subj + '_ventricles_timeseries_deriv_' + task + '.1D')
        os.system('1d_tool.py -overwrite -infile ' + datadir + subj + '_wholebrain_timeseries_' + task + '.1D -derivative -write ' + datadir + subj + '_wholebrain_timeseries_deriv_' + task + '.1D')

       ## Import task noise parameters
        print 'Importing wm, ventricle, and global brain time series for subject', subj

        timeseriesRegressors[trstart:trend,0] = np.loadtxt(datadir + subj + '_WM_timeseries_' + task + '.1D')
        timeseriesRegressors[trstart:trend,1] = np.loadtxt(datadir + subj + '_WM_timeseries_deriv_' + task + '.1D')

        timeseriesRegressors[trstart:trend,2] = np.loadtxt(datadir + subj + '_ventricles_timeseries_' + task + '.1D')
        timeseriesRegressors[trstart:trend,3] = np.loadtxt(datadir + subj + '_ventricles_timeseries_deriv_' + task + '.1D')

        timeseriesRegressors[trstart:trend,4] = np.loadtxt(datadir + subj + '_wholebrain_timeseries_' + task + '.1D')
        timeseriesRegressors[trstart:trend,5] = np.loadtxt(datadir + subj + '_wholebrain_timeseries_deriv_' + task + '.1D')

        ## Import rest movement regressors
        movementReg = hcpdir + task + '/Movement_Regressors.txt'
        motionRegressors[trstart:trend,:] = np.loadtxt(movementReg)

        if gsr != True:
           timeseriesRegressors = timeseriesRegressors[:,:4] # Exclude last two regressors

        linearTrendRegressor[trstart:trend,i] = np.arange(trend-trstart)
        demeanRegressor[trstart:trend,0] = i + 1

        trcount += TRsPerRun[i]

        if taskkey==task[6:-3]: 
            continue
        else:
            taskkey = task[6:-3] # Define string identifier for tasks
            stimdir = basedir + 'data/timingfiles/'
            stimfiles = glob.glob(stimdir + subj + '*EV*' + taskkey + '*1D')
            numTaskStims[i] = len(stimfiles)

            for stim in range(numTaskStims[i]):
                stimMat[:,stimcount] = np.loadtxt(stimfiles[stim])
                stimcount += 1


    # Trivial implementation of knowing which stim corresponds to which col; but need for FIR compatibility
    stim_index = []
    for stim in range(stimMat.shape[1]):
        stim_index.append(stim)
    stim_index = np.asarray(stim_index)

    ## 
    # Convolve taskstim regressors based on SPM canonical HRF (likely period of task-induced activity)
    print 'Convolving task stimulus binary files with SPM canonical HRF for subject', subj
    taskStims_HRF = np.zeros(stimMat.shape)
    spm_hrfTS = spm_hrf(trLength,oversampling=1)
    
    temporalMask = []
    trcount = 0
    for i in range(len(tasks)):
        trstart = trcount
        trend = trcount + TRsPerRun[i]
        
        # Create run-wise temporal mask; skip first 5 TRs
        runMask = np.ones((TRsPerRun[i],),dtype=bool)
        runMask[:5] = False
        temporalMask.extend(runMask)
        
        for stim in range(stimMat.shape[1]):

            # Perform convolution
            tmpconvolve = np.convolve(stimMat[trstart:trend,stim],spm_hrfTS)
            tmpconvolve_run = tmpconvolve[:TRsPerRun[i]] # Make sure to cut off at the end of the run
            taskStims_HRF[trstart:trend,stim] = tmpconvolve_run

        trcount += TRsPerRun[i]


    taskRegressors = taskStims_HRF.copy()
    #np.savetxt(subj + '_regressors_canonical.csv',stimMat, delimiter=',')

    output = {}
    # Commented out since we demean each run prior to loading data anyway
    #noiseRegressors = np.hstack((demeanRegressor,linearTrendRegressor,motionRegressors,timeseriesRegressors))
    noiseRegressors = np.hstack((linearTrendRegressor,motionRegressors,timeseriesRegressors))
    output['noiseRegressors'] = noiseRegressors
    output['taskRegressors'] = taskRegressors
    output['taskDesignMat'] = stimMat
    output['allRegressors'] = np.hstack((noiseRegressors,taskRegressors))
    output['stimIndex'] = stim_index
    output['tMask'] = np.asarray(temporalMask,dtype=bool)

    return output


def loadStimFiles_task_HCP_withFIR(subj, gsr=True, nRegsFIR=25):
    basedir = '/projects3/NetworkDiversity/'
    datadir = basedir + 'data/hcppreprocessed/' + subj + '/nuisanceRegsTasks/'
    hcpdir = '/projects/ExternalDatasets/HCPData/data/minimalpreproc/' + subj + '/MNINonLinear/Results/'
    nMotionParams = 12
    trLength = .720
    totalTRs = 3880
    nCond = 24

    tasks = ['tfMRI_EMOTION_RL','tfMRI_EMOTION_LR','tfMRI_GAMBLING_RL','tfMRI_GAMBLING_LR','tfMRI_LANGUAGE_RL','tfMRI_LANGUAGE_LR','tfMRI_MOTOR_RL','tfMRI_MOTOR_LR','tfMRI_RELATIONAL_RL','tfMRI_RELATIONAL_LR','tfMRI_SOCIAL_RL','tfMRI_SOCIAL_LR','tfMRI_WM_RL','tfMRI_WM_LR']
    TRsPerRun = [176,176,253,253,316,316,284,284,232,232,274,274,405,405]

    timeseriesRegressors = np.zeros((totalTRs,6)) # WM, ventricle, and global signal regressors + their derivatives
    motionRegressors = np.zeros((totalTRs,12)) # 12 motion regressors
    stimMat = np.zeros((totalTRs,nCond))
    linearTrendRegressor = np.zeros((totalTRs,len(tasks)))
    demeanRegressor = np.zeros((totalTRs,1))

    numTaskStims = {}
    trcount = 0
    stimcount = 0
    taskkey = ''
    for i in range(len(tasks)):

        task = tasks[i]

        trstart = trcount
        trend = trcount + TRsPerRun[i]

        ## Create derivative time series
        print 'Creating derivative time series for ventricle, white matter, and whole brain signal for subject', subj
        
        os.system('1d_tool.py -overwrite -infile ' + datadir + subj + '_WM_timeseries_' + task + '.1D -derivative -write ' + datadir + subj + '_WM_timeseries_deriv_' + task + '.1D')
        os.system('1d_tool.py -overwrite -infile ' + datadir + subj + '_ventricles_timeseries_' + task + '.1D -derivative -write ' + datadir + subj + '_ventricles_timeseries_deriv_' + task + '.1D')
        os.system('1d_tool.py -overwrite -infile ' + datadir + subj + '_wholebrain_timeseries_' + task + '.1D -derivative -write ' + datadir + subj + '_wholebrain_timeseries_deriv_' + task + '.1D')

       ## Import task noise parameters
        print 'Importing wm, ventricle, and global brain time series for subject', subj

        timeseriesRegressors[trstart:trend,0] = np.loadtxt(datadir + subj + '_WM_timeseries_' + task + '.1D')
        timeseriesRegressors[trstart:trend,1] = np.loadtxt(datadir + subj + '_WM_timeseries_deriv_' + task + '.1D')

        timeseriesRegressors[trstart:trend,2] = np.loadtxt(datadir + subj + '_ventricles_timeseries_' + task + '.1D')
        timeseriesRegressors[trstart:trend,3] = np.loadtxt(datadir + subj + '_ventricles_timeseries_deriv_' + task + '.1D')

        timeseriesRegressors[trstart:trend,4] = np.loadtxt(datadir + subj + '_wholebrain_timeseries_' + task + '.1D')
        timeseriesRegressors[trstart:trend,5] = np.loadtxt(datadir + subj + '_wholebrain_timeseries_deriv_' + task + '.1D')

        ## Import rest movement regressors
        movementReg = hcpdir + task + '/Movement_Regressors.txt'
        motionRegressors[trstart:trend,:] = np.loadtxt(movementReg)

        if gsr != True:
           timeseriesRegressors = timeseriesRegressors[:,:4] # Exclude last two regressors

        linearTrendRegressor[trstart:trend,i] = np.arange(trend-trstart)
        demeanRegressor[trstart:trend,0] = i + 1

        trcount += TRsPerRun[i]

        if taskkey==task[6:-3]: 
            continue
        else:
            taskkey = task[6:-3] # Define string identifier for tasks
            stimdir = basedir + 'data/timingfiles/'
            stimfiles = glob.glob(stimdir + subj + '*EV*' + taskkey + '*1D')
            numTaskStims[i] = len(stimfiles)

            for stim in range(numTaskStims[i]):
                stimMat[:,stimcount] = np.loadtxt(stimfiles[stim])
                stimcount += 1



    ## 
    # Convolve taskstim regressors based on SPM canonical HRF (likely period of task-induced activity)
    print 'Modeling task-timing block events with an FIR model using a', nRegsFIR, 'TR lag'

    ## First set up FIR design matrix
    stim_index = []
    taskStims_FIR = [] 
    for stim in range(stimMat.shape[1]):
        taskStims_FIR.append([])
        time_ind = np.where(stimMat[:,stim]==1)[0]
        blocks = _group_consecutives(time_ind) # Get blocks (i.e., sets of consecutive TRs)
        # Identify the longest block - set FIR duration to longest block
        maxRegsForBlocks = 0
        for block in blocks:
            if len(block) > maxRegsForBlocks: maxRegsForBlocks = len(block)
        taskStims_FIR[stim] = np.zeros((stimMat.shape[0],maxRegsForBlocks+nRegsFIR)) # Task timing for this condition is TR x length of block + FIR lag
        stim_index.extend(np.repeat(stim,maxRegsForBlocks+nRegsFIR))
    stim_index = np.asarray(stim_index)

    ## Now fill in FIR design matrix
    temporalMask = []
    trcount = 0
    for i in range(len(tasks)):
        # Make sure to cut-off FIR models for each run/task separately
        trstart = trcount
        trend = trstart + TRsPerRun[i]
        
        # Create run-wise temporal mask; skip first 5 TRs
        runMask = np.ones((TRsPerRun[i],),dtype=bool)
        runMask[:5] = False
        temporalMask.extend(runMask)

        for stim in range(stimMat.shape[1]):
            time_ind = np.where(stimMat[:,stim]==1)[0]
            blocks = _group_consecutives(time_ind) # Get blocks (i.e., sets of consecutive TRs)
            for block in blocks:
                reg = 0
                for tr in block:
                    # Set impulses for this run/task only
                    if trstart < tr < trend:
                        taskStims_FIR[stim][tr,reg] = 1
                        reg += 1

                    if not trstart < tr < trend: continue # If TR not in this run, skip this block

                # If TR is not in this run, skip this block
                if not trstart < tr < trend: continue

                # Set lag due to HRF
                for lag in range(1,nRegsFIR+1):
                    # Set impulses for this run/task only
                    if trstart < tr+lag < trend:
                        taskStims_FIR[stim][tr+lag,reg] = 1
                        reg += 1
        
        # Re-do for next runs
        trcount += TRsPerRun[i]

    taskStims_FIR2 = np.zeros((stimMat.shape[0],1))
    task_index = []
    for stim in range(stimMat.shape[1]):
        task_index.extend(np.repeat(stim,taskStims_FIR[stim].shape[1]))
        taskStims_FIR2 = np.hstack((taskStims_FIR2,taskStims_FIR[stim]))

    taskStims_FIR2 = np.delete(taskStims_FIR2,0,axis=1)

    #taskRegressors = np.asarray(taskStims_FIR)
    taskRegressors = taskStims_FIR2
#    np.savetxt(subj + '_regressors_FIR.csv',taskStims_FIR2, delimiter=',')

    # To prevent SVD does not converge error, make sure there are no columns with 0s
    zero_cols = np.where(np.sum(taskRegressors,axis=0)==0)[0]
    taskRegressors = np.delete(taskRegressors, zero_cols, axis=1)
    stim_index = np.delete(stim_index, zero_cols)


    output = {}
    # Commented out since we demean each run prior to loading data anyway
    #noiseRegressors = np.hstack((demeanRegressor,linearTrendRegressor,motionRegressors,timeseriesRegressors))
    noiseRegressors = np.hstack((linearTrendRegressor,motionRegressors,timeseriesRegressors))
    output['noiseRegressors'] = noiseRegressors
    output['taskRegressors'] = taskRegressors
    output['taskDesignMat'] = stimMat
    output['stimIndex'] = stim_index
    output['allRegressors'] = np.hstack((noiseRegressors,taskRegressors))
    output['tMask'] = np.asarray(temporalMask,dtype=bool)

    return output


def loadStimFiles_task_HCPbyBlock(subj, gsr=True):
    basedir = '/projects3/NetworkDiversity/'
    datadir = basedir + 'data/hcppreprocessed/' + subj + '/nuisanceRegsTasks/'
    hcpdir = '/projects/ExternalDatasets/HCPData/data/minimalpreproc/' + subj + '/MNINonLinear/Results/'
    nMotionParams = 12
    trLength = .720
    totalTRs = 3880
    nCond = 24

    tasks = ['tfMRI_EMOTION_RL','tfMRI_EMOTION_LR','tfMRI_GAMBLING_RL','tfMRI_GAMBLING_LR','tfMRI_LANGUAGE_RL','tfMRI_LANGUAGE_LR','tfMRI_MOTOR_RL','tfMRI_MOTOR_LR','tfMRI_RELATIONAL_RL','tfMRI_RELATIONAL_LR','tfMRI_SOCIAL_RL','tfMRI_SOCIAL_LR','tfMRI_WM_RL','tfMRI_WM_LR']
    TRsPerRun = [176,176,253,253,316,316,284,284,232,232,274,274,405,405]

    timeseriesRegressors = np.zeros((totalTRs,6)) # WM, ventricle, and global signal regressors + their derivatives
    motionRegressors = np.zeros((totalTRs,12)) # 12 motion regressors
    stimArrs = {}
    nBlocks = 0
    linearTrendRegressor = np.zeros((totalTRs,len(tasks)))
    demeanRegressor = np.zeros((totalTRs,1))

    numTaskStims = {}
    trcount = 0
    stimcount = 0
    taskkey = ''
    for i in range(len(tasks)):

        task = tasks[i]

        trstart = trcount
        trend = trcount + TRsPerRun[i]

        ## Create derivative time series
        print 'Creating derivative time series for ventricle, white matter, and whole brain signal for subject', subj
        
        os.system('1d_tool.py -overwrite -infile ' + datadir + subj + '_WM_timeseries_' + task + '.1D -derivative -write ' + datadir + subj + '_WM_timeseries_deriv_' + task + '.1D')
        os.system('1d_tool.py -overwrite -infile ' + datadir + subj + '_ventricles_timeseries_' + task + '.1D -derivative -write ' + datadir + subj + '_ventricles_timeseries_deriv_' + task + '.1D')
        os.system('1d_tool.py -overwrite -infile ' + datadir + subj + '_wholebrain_timeseries_' + task + '.1D -derivative -write ' + datadir + subj + '_wholebrain_timeseries_deriv_' + task + '.1D')

       ## Import task noise parameters
        print 'Importing wm, ventricle, and global brain time series for subject', subj

        timeseriesRegressors[trstart:trend,0] = np.loadtxt(datadir + subj + '_WM_timeseries_' + task + '.1D')
        timeseriesRegressors[trstart:trend,1] = np.loadtxt(datadir + subj + '_WM_timeseries_deriv_' + task + '.1D')

        timeseriesRegressors[trstart:trend,2] = np.loadtxt(datadir + subj + '_ventricles_timeseries_' + task + '.1D')
        timeseriesRegressors[trstart:trend,3] = np.loadtxt(datadir + subj + '_ventricles_timeseries_deriv_' + task + '.1D')

        timeseriesRegressors[trstart:trend,4] = np.loadtxt(datadir + subj + '_wholebrain_timeseries_' + task + '.1D')
        timeseriesRegressors[trstart:trend,5] = np.loadtxt(datadir + subj + '_wholebrain_timeseries_deriv_' + task + '.1D')

        ## Import rest movement regressors
        movementReg = hcpdir + task + '/Movement_Regressors.txt'
        motionRegressors[trstart:trend,:] = np.loadtxt(movementReg)

        if gsr != True:
           timeseriesRegressors = timeseriesRegressors[:,:4] # Exclude last two regressors

        linearTrendRegressor[trstart:trend,i] = np.arange(trend-trstart)
        demeanRegressor[trstart:trend,0] = i + 1

        trcount += TRsPerRun[i]

        taskkey = task[6:] # Define string identifier for tasks
        stimdir = basedir + 'data/timingfiles_byblock/'
        stimfiles = glob.glob(stimdir + subj + '_' +  taskkey + '_cond*' + '*1D')
        numTaskStims[i] = len(stimfiles)
        stimArrs[i] = []

        for stim in range(numTaskStims[i]):
            tmp = np.loadtxt(stimfiles[stim],delimiter=',')
            stimArrs[i].append(tmp)
            try:
                nBlocks += tmp.shape[1] # Count number of columns in the last appended array
            except IndexError: # If tmp has no 2nd dimension (i.e., a 1d array), add only 1 since there is only 1 column
                nBlocks += 1

    temporalMask = []
    stimMat = np.zeros((totalTRs,nBlocks))
    stimcount = 0
    trcount = 0
    print 'Total number of blocks', nBlocks
    for i in range(len(tasks)):
        trstart = trcount
        trend = trstart + TRsPerRun[i]
        # Create temporal mask
        runMask = np.ones((TRsPerRun[i],),dtype=bool)
        runMask[:5] = False
        temporalMask.extend(runMask)

        for j in range(numTaskStims[i]):
            stimstart = stimcount
            try:
                stimend = stimstart + stimArrs[i][j].shape[1] # count number of blocks associated with this stim
            except IndexError:
                stimend = stimstart + 1 # if stimArrs[i]pj] has no 2nd dimension (i.e., a 1d array), add only a 1 for this
            
            if stimend==stimstart+1:
                stimArrs[i][j].shape = (len(stimArrs[i][j]),1)
            stimMat[trstart:trend,stimstart:stimend] = stimArrs[i][j]
            
            try:
                stimcount += stimArrs[i][j].shape[1] # count number of blocks associated with this stim
            except IndexError:
                stimcount += 1 # if stimArrs[i]pj] has no 2nd dimension (i.e., a 1d array), add only a 1 for this

        trcount += TRsPerRun[i]

    # Trivial implementation of knowing which stim corresponds to which col; but need for FIR compatibility
    # each block is its own column; here block and stim are synonymous
    stim_index = []
    for stim in range(stimMat.shape[1]):
        stim_index.append(stim)
    stim_index = np.asarray(stim_index)

    ## 
    # Convolve taskstim regressors based on SPM canonical HRF (likely period of task-induced activity)
    print 'Convolving task stimulus binary files with SPM canonical HRF for subject', subj
    taskStims_HRF = np.zeros(stimMat.shape)
    spm_hrfTS = spm_hrf(trLength,oversampling=1)
    
    trcount = 0
    for i in range(len(tasks)):
        trstart = trcount
        trend = trcount + TRsPerRun[i]
        for stim in range(stimMat.shape[1]):

            # Perform convolution
            tmpconvolve = np.convolve(stimMat[trstart:trend,stim],spm_hrfTS)
            tmpconvolve_run = tmpconvolve[:TRsPerRun[i]] # Make sure to cut off at the end of the run
            taskStims_HRF[trstart:trend,stim] = tmpconvolve_run

        trcount += TRsPerRun[i]


    taskRegressors = taskStims_HRF.copy()
    #np.savetxt(subj + '_regressors_canonical.csv',stimMat, delimiter=',')

    output = {}
    # Commented out since we demean each run prior to loading data anyway
    #noiseRegressors = np.hstack((demeanRegressor,linearTrendRegressor,motionRegressors,timeseriesRegressors))
    noiseRegressors = np.hstack((linearTrendRegressor,motionRegressors,timeseriesRegressors))
    output['noiseRegressors'] = noiseRegressors
    output['taskRegressors'] = taskRegressors
    output['taskDesignMat'] = stimMat
    output['allRegressors'] = np.hstack((noiseRegressors,taskRegressors))
    output['stimIndex'] = stim_index
    output['tMask'] = np.asarray(temporalMask,dtype=bool)

    return output


def _regression((data,regressors)):
    # Add 'constant' regressor
    regressors = sm.add_constant(regressors)
    model = sm.OLS(data, regressors,missing='drop')
    results = model.fit()
    betas = results.params[:] 
    resid = results.resid 
    return betas, resid

def _group_consecutives(vals, step=1):
    """Return list of consecutive lists of numbers from vals (number list)."""
    run = []
    result = [run]
    expect = None
    for v in vals:
        if (v == expect) or (expect is None):
            run.append(v)
        else:
            run = [v]
            result.append(run)
        expect = v + step
    return result

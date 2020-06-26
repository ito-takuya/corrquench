# Taku Ito
# 09/10/2018
# Generate nuisance regressors for HCP data
# Use ciric et al. (2017) preprocessing approach
# aCompCor, 5 PCs of WM + Ventricles

import numpy as np
import nibabel as nib
import h5py
import os
import scipy
from scipy import signal
import time
import multiprocessing as mp

### New subjects for QC'd 352 HCP subjs
##### HCP352 QC'd data set
# Exploratory set
subjNums = ['100206','108020','117930','126325','133928','143224','153934','164636','174437','183034','194443','204521','212823','268749','322224','385450','463040','529953','587664','656253','731140','814548','877269','978578','100408','108222','118124','126426','134021','144832','154229','164939','175338','185139','194645','204622','213017','268850','329844','389357','467351','530635','588565','657659','737960','816653','878877','987074','101006','110007','118225','127933','134324','146331','154532','165638','175742','185341','195445','205119','213421','274542','341834','393247','479762','545345','597869','664757','742549','820745','887373','989987','102311','111009','118831','128632','135528','146432','154936','167036','176441','186141','196144','205725','213522','285345','342129','394956','480141','552241','598568','671855','744553','826454','896879','990366','102513','112516','118932','129028','135629','146533','156031','167440','176845','187850','196346','205826','214423','285446','348545','395756','481042','553344','599671','675661','749058','832651','899885','991267','102614','112920','119126','129129','135932','147636','157336','168745','177645','188145','198350','208226','214726','286347','349244','406432','486759','555651','604537','679568','749361','835657','901442','992774','103111','113316','120212','130013','136227','148133','157437','169545','178748','188549','198451','208327','217429','290136','352738','414229','497865','559457','615744','679770','753150','837560','907656','993675','103414','113619','120414','130114','136833','150726','157942','171330']

# Validation set
#subjNums1 = ['178950','189450','199453','209228','220721','298455','356948','419239','499566','561444','618952','680452','757764','841349','908860','103818','113922','121618','130619','137229','151829','158035','171633','179346','190031','200008','210112','221319','299154','361234','424939','500222','570243','622236','687163','769064','845458','911849','104416','114217','122317','130720','137532','151930','159744','172029','180230','191235','200614','211316','228434','300618','361941','432332','513130','571144','623844','692964','773257','857263','926862','105014','114419','122822','130821','137633','152427','160123','172938','180432','192035','200917','211417','239944','303119','365343','436239','513736','579665','638049','702133','774663','865363','930449','106521','114823','123521','130922','137936','152831','160729','173334','180533','192136','201111','211619','249947','305830','366042','436845','516742','580650','645450','715041','782561','871762','942658','106824','117021','123925','131823','138332','153025','162026','173536','180735','192439','201414','211821','251833','310621','371843','445543','519950','580751','647858','720337','800941','871964','955465','107018','117122','125222','132017','138837','153227','162329','173637','180937','193239','201818','211922','257542','314225','378857','454140','523032','585862','654350','725751','803240','872562','959574','107422','117324','125424','133827','142828','153631','164030','173940','182739','194140','202719','212015','257845','316633','381543','459453','525541','586460','654754','727553','812746','873968','966975']


def createRegressorsParallel(nproc=5):
    """
    Wrapper function to run subjects in parallels:
    Usage:
    1. Open ipython
    2. In python, write code:
        `import generateNuisanceParameters_Ciric as gnc`
        `gnc.createRegressorsParallel()`
    Code will run in parallel, with default set to 5 CPUs
    """
    pool = mp.Pool(processes=nproc)
    pool.map_async(createNuisanceRegressorsSubject,subjNums).get()
    pool.close()
    pool.join()


def createNuisanceRegressorsSubject(subj):
    """
    EDIT this function!!
    One output file per subject
    """
    #### Change parameters
    allRuns = ['rfMRI_REST1_RL', 'rfMRI_REST1_LR','rfMRI_REST2_RL', 'rfMRI_REST2_LR','tfMRI_EMOTION_RL','tfMRI_EMOTION_LR','tfMRI_GAMBLING_RL','tfMRI_GAMBLING_LR','tfMRI_LANGUAGE_RL','tfMRI_LANGUAGE_LR','tfMRI_MOTOR_RL','tfMRI_MOTOR_LR','tfMRI_RELATIONAL_RL','tfMRI_RELATIONAL_LR','tfMRI_SOCIAL_RL','tfMRI_SOCIAL_LR','tfMRI_WM_RL','tfMRI_WM_LR']
    maskdir = '/projects3/NetworkDiversity/data/hcppreprocessedmsmall/' + subj + '/masks/'
    globalmask = maskdir + subj + '_wholebrainmask_func_dil1vox.nii.gz'
    wmmask = maskdir + subj + '_wmMask_func_eroded.nii.gz'
    ventriclesmask = maskdir + subj + '_ventricles_func_eroded.nii.gz'
    outputfile = '/projects3/TaskFCMech/data/hcppreprocessedmsmall/nuisanceRegressors/' + subj + '_nuisanceRegressors.h5'
    datadir = '/projects3/ExternalDatasets2/HCPData2/HCPS1200MSMAll/' + subj + '/MNINonLinear/Results/'
    compCorComponents = 5
    spikeReg = .25
    ####

    for run in allRuns:
    
        print 'creating nuisance regressors for subject', subj, 'run:', run
        
        #### Obtain movement parameters -- this will differ across preprocessing pipelines (e.g., HCP vs. typical)
        # For all 12 movement parameters (6 regressors + derivatives)
        movementRegressors = np.loadtxt(datadir + run + '/Movement_Regressors.txt')
        # Separate the two parameters out for clarity
        motionParams = movementRegressors[:,:6]
        motionParams_deriv = movementRegressors[:,6:] # HCP automatically computes derivative of motion parameters
        
        h5f = h5py.File(outputfile,'a')
        try:
            h5f.create_dataset(run + '/motionParams',data=motionParams)
            h5f.create_dataset(run + '/motionParams_deriv',data=motionParams_deriv)
        except:
            del h5f[run + '/motionParams'], h5f[run + '/motionParams_deriv']
            h5f.create_dataset(run + '/motionParams',data=motionParams)
            h5f.create_dataset(run + '/motionParams_deriv',data=motionParams_deriv)
        h5f.close()

        #### Obtain relative root-mean-square displacement -- this will differ across preprocessing pipelines
        # A simple alternative is to compute the np.sqrt(x**2 + y**2 + z**2), where x, y, and z are motion displacement parameters
        # e.g., x = x[t] - x[t-1]; y = y[t] - y[t-1]; z = z[t] - z[t-1]
        relativeRMS = np.loadtxt(datadir + run + '/Movement_RelativeRMS.txt')
        _createMotionSpikeRegressors(relativeRMS,outputfile, run, spikeReg=spikeReg)

        inputname = datadir + run + '/' + run + '.nii.gz' 

        _createPhysiologicalNuisanceRegressors(inputname, outputfile, run, globalmask, wmmask, ventriclesmask, aCompCor=compCorComponents)
         
        


def _createMotionSpikeRegressors(relative_rms, output, run,  spikeReg=.25):
    """
    relative_rms-  time x 1 array (for HCP data, can be obtained from the txt file 'Movement_RelativeRMS.txt'; otherwise see Van Dijk et al. (2011) Neuroimage for approximate calculation
    output      -   output h5f file
    spikeReg    -   generate spike time regressors for motion spikes, using a default threshold of .25mm FD threshold
    """
    # Create h5py output
    h5f = h5py.File(output,'a')

    nTRs = relative_rms.shape[0]

    motionSpikes = np.where(relative_rms>spikeReg)[0]
    if len(motionSpikes)>0:
        spikeRegressorsArray = np.zeros((nTRs,len(motionSpikes)))

        for spike in range(len(motionSpikes)):
            spike_time = motionSpikes[spike]
            spikeRegressorsArray[spike_time,spike] = 1.0

        spikeRegressorsArray = np.asarray(spikeRegressorsArray,dtype=bool)

        try:
            h5f.create_dataset(run + '/motionSpikes',data=spikeRegressorsArray)
        except:
            del h5f[run + '/motionSpikes']
            h5f.create_dataset(run + '/motionSpikes',data=spikeRegressorsArray)

        h5f.close()

def _createPhysiologicalNuisanceRegressors(inputname, output, run, globalmask, wmmask, ventriclesmask, aCompCor=5):
    """
    inputname   -   4D input time series to obtain nuisance regressors
    output      -   output h5f file
    globalmask  -   whole brain mask to extract global time series
    wmmask      -   white matter mask (functional) to extract white matter time series
    ventriclesmask- ventricles mask (functional) to extract ventricle time series
    aCompCor    -   Create PC component time series of white matter and ventricle time series, using first n PCs
    """
    
    # Create h5py output
    h5f = h5py.File(output,'a')

    # Load raw fMRI data (in volume space)
    print 'Loading raw fMRI data'
    fMRI4d = nib.load(inputname).get_data()

    ##########################################################
    ## Nuisance time series (Global signal, WM, and Ventricles)
    print 'Obtaining standard global, wm, and ventricle signals and their derivatives'
    # Global signal
    globalMask = nib.load(globalmask).get_data()
    globalMask = np.asarray(globalMask,dtype=bool)
    globaldata = fMRI4d[globalMask].copy()
    globaldata = signal.detrend(globaldata,axis=1,type='constant')
    globaldata = signal.detrend(globaldata,axis=1,type='linear')
    global_signal1d = np.mean(globaldata,axis=0)
    # White matter signal
    wmMask = nib.load(wmmask).get_data()
    wmMask = np.asarray(wmMask,dtype=bool)
    wmdata = fMRI4d[wmMask].copy()
    wmdata = signal.detrend(wmdata,axis=1,type='constant')
    wmdata = signal.detrend(wmdata,axis=1,type='linear')
    wm_signal1d = np.mean(wmdata,axis=0)
    # Ventricle signal
    ventricleMask = nib.load(ventriclesmask).get_data()
    ventricleMask = np.asarray(ventricleMask,dtype=bool)
    ventricledata = fMRI4d[ventricleMask].copy()
    ventricledata = signal.detrend(ventricledata,axis=1,type='constant')
    ventricledata = signal.detrend(ventricledata,axis=1,type='linear')
    ventricle_signal1d = np.mean(ventricledata,axis=0)

    del fMRI4d

    ## Create derivative time series (with backward differentiation, consistent with 1d_tool.py -derivative option)
    # Global signal derivative
    global_signal1d_deriv = np.zeros(global_signal1d.shape)
    global_signal1d_deriv[1:] = global_signal1d[1:] - global_signal1d[:-1]
    # White matter signal derivative
    wm_signal1d_deriv = np.zeros(wm_signal1d.shape)
    wm_signal1d_deriv[1:] = wm_signal1d[1:] - wm_signal1d[:-1]
    # Ventricle signal derivative
    ventricle_signal1d_deriv = np.zeros(ventricle_signal1d.shape)
    ventricle_signal1d_deriv[1:] = ventricle_signal1d[1:] - ventricle_signal1d[:-1]

    ## Write to h5py
    try:
        h5f.create_dataset(run + '/global_signal',data=global_signal1d)
        h5f.create_dataset(run + '/global_signal_deriv',data=global_signal1d_deriv)
        h5f.create_dataset(run + '/wm_signal',data=wm_signal1d)
        h5f.create_dataset(run + '/wm_signal_deriv',data=wm_signal1d_deriv)
        h5f.create_dataset(run + '/ventricle_signal',data=ventricle_signal1d)
        h5f.create_dataset(run + '/ventricle_signal_deriv',data=ventricle_signal1d_deriv)
    except:
        del h5f[run + '/global_signal'], h5f[run + '/global_signal_deriv'], h5f[run + '/wm_signal'], h5f[run + '/wm_signal_deriv'], h5f[run + '/ventricle_signal'], h5f[run + '/ventricle_signal_deriv']
        h5f.create_dataset(run + '/global_signal',data=global_signal1d)
        h5f.create_dataset(run + '/global_signal_deriv',data=global_signal1d_deriv)
        h5f.create_dataset(run + '/wm_signal',data=wm_signal1d)
        h5f.create_dataset(run + '/wm_signal_deriv',data=wm_signal1d_deriv)
        h5f.create_dataset(run + '/ventricle_signal',data=ventricle_signal1d)
        h5f.create_dataset(run + '/ventricle_signal_deriv',data=ventricle_signal1d_deriv)

    
    ##########################################################
    ## Obtain aCompCor regressors using first 5 components of WM and Ventricles (No GSR!)
    ncomponents = 5
    nTRs = len(global_signal1d)
    print 'Obtaining aCompCor regressors and their derivatives'
    # WM time series
    wmstart = time.time()
    # Obtain covariance matrix, and obtain first 5 PCs of WM time series
    tmpcov = np.corrcoef(wmdata.T)
    eigenvalues, topPCs = scipy.sparse.linalg.eigs(tmpcov,k=ncomponents,which='LM')
    # Now using the top n PCs 
    aCompCor_WM = topPCs
#    wmend = time.time() - wmstart
#    print 'WM aCompCor took', wmend, 'seconds'
    
    # Ventricle time series
    ventstart = time.time()
    # Obtain covariance matrix, and obtain first 5 PCs of ventricle time series
    tmpcov = np.corrcoef(ventricledata.T)
    eigenvalues, topPCs = scipy.sparse.linalg.eigs(tmpcov,k=ncomponents,which='LM')
    # Now using the top n PCs
    aCompCor_ventricles = topPCs
#    ventricletime = time.time() - ventstart 
#    print 'Ventricle aCompCor took', ventricletime, 'seconds' 
    
    # White matter signal derivative using backwards differentiation
    aCompCor_WM_deriv = np.zeros(aCompCor_WM.shape)
    aCompCor_WM_deriv[1:,:] = np.real(aCompCor_WM[1:,:]) - np.real(aCompCor_WM[:-1,:])
    # Ventricle signal derivative
    aCompCor_ventricles_deriv = np.zeros(aCompCor_ventricles.shape)
    aCompCor_ventricles_deriv[1:,:] = np.real(aCompCor_ventricles[1:,:]) - np.real(aCompCor_ventricles[:-1,:])

    ## Write to h5py
    try:
        h5f.create_dataset(run + '/aCompCor_WM',data=aCompCor_WM)
        h5f.create_dataset(run + '/aCompCor_WM_deriv',data=aCompCor_WM_deriv)
        h5f.create_dataset(run + '/aCompCor_ventricles',data=aCompCor_ventricles)
        h5f.create_dataset(run + '/aCompCor_ventricles_deriv',data=aCompCor_ventricles_deriv)
    except:
        del h5f[run + '/aCompCor_WM'], h5f[run + '/aCompCor_WM_deriv'], h5f[run + '/aCompCor_ventricles'], h5f[run + '/aCompCor_ventricles_deriv']
        h5f.create_dataset(run + '/aCompCor_WM',data=aCompCor_WM)
        h5f.create_dataset(run + '/aCompCor_WM_deriv',data=aCompCor_WM_deriv)
        h5f.create_dataset(run + '/aCompCor_ventricles',data=aCompCor_ventricles)
        h5f.create_dataset(run + '/aCompCor_ventricles_deriv',data=aCompCor_ventricles_deriv)


    ##########################################################
    ## Load motion parameters, and calculate motion spike regressors

    h5f.close()



        


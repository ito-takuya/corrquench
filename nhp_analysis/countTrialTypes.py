# Taku Ito 
# 10/15/2018

import numpy as np
import pandas as pd

# Count number of trials per task per monkey

sessions = ['100706','100731','100818','100826','100913','101008','101028','101127','101207','110106','110115_01','100724','100802','100819','100827','100915','101009','101030','101128','101209','110107_01','110120','100725','100803','100820','100828','100917','101023','101122','101202','101210','110110_01','110121','100726','100804','100823','100907','100920','101024','101123','101203','101216','110110_02','100730','100817','100824','100910','100921','101027','101124','101206','101217','110111_01']

datadir = '/projects3/TaskFCMech/data/nhpData/'
idfile = datadir + 'monkeyToSessionID.csv'
monkeyTable = pd.read_csv(idfile,delimiter=',')

monkeyID = {}
for i in range(1, len(sessions)+1):
    if i < 10:
        sess_str = 'session_ ' + str(i)
        name_str = 'name_ ' + str(i)
    else:
        sess_str = 'session_' + str(i)
        name_str = 'name_' + str(i)

    session = str(monkeyTable[sess_str][0])
    name = monkeyTable[name_str][0]

    if session=='110111_02.mat': session = '110111_02'

    monkeyID[session] = name


n_flamap_paula = 0
n_mocol_paula = 0
n_delsac_paula = 0
n_total_paula = 0

n_flamap_rex = 0
n_mocol_rex = 0
n_delsac_rex = 0
n_total_rex = 0

for session in sessions:
    fileid = datadir + session + '_trialInfoAllTasks.csv'
    dat = pd.read_csv(fileid)
    if monkeyID[session] == 'paula':
        n_flamap_paula += np.sum(dat['task']=='flamap')
        n_delsac_paula += np.sum(dat['task']=='delsac')
        n_mocol_paula += np.sum(dat['task']=='mocol')
        n_total_paula += len(dat['task'])
    elif monkeyID[session] == 'rex':
        n_flamap_rex += np.sum(dat['task']=='flamap')
        n_delsac_rex += np.sum(dat['task']=='delsac')
        n_mocol_rex += np.sum(dat['task']=='mocol')
        n_total_rex += len(dat['task'])


print 'Paula -- total number of trials:', n_total_paula
print 'Number of mocol trials', n_mocol_paula
print 'Number of flamap trials', n_flamap_paula
print 'Number of delsac trials', n_delsac_paula

print '\n'

print 'Rex -- total number of trials:', n_total_rex
print 'Number of mocol trials', n_mocol_rex
print 'Number of flamap trials', n_flamap_rex
print 'Number of delsac trials', n_delsac_rex

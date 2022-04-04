# Training a classifier starting from ROOT files
#
# 
#
#!pip freeze ! grep uproot

import logging

import uproot
import pandas as pd
import matplotlib as plt
import numpy as np
import re


FORMAT = '%(asctime)-15s- %(levelname)s - %(name)s -%(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    print('Welcome!')


# converting from root to a format suitable for manipulation 
# fcosmic = uproot.open("OutBarTree_run6201_cosmics.root")
# tcosmic = fcosmic['Tree_BarData']


fpbar = uproot.open("OutBarTree_run6201_pbarhold.root")
tpbar = fpbar['Tree_BarData']

# prints number of entries
logger.info('Entries in file 0: ' + str(tpbar.num_entries))

# prints variable names 
logger.info(tpbar.keys())

dfpbar = tpbar.arrays(library="pd")

#  https://stackoverflow.com/questions/21752989/numpy-efficiently-avoid-0s-when-taking-logmatrix
from numpy import errstate #,isneginf,array

ibots = [ re.sub(r'[a-zA-Z]*', '', s) for s in dfpbar.filter(regex="AmpBot.*").columns.values ]

for i in ibots:
    
    with errstate(divide='ignore', invalid='ignore'):           
        dfpbar['P'+i] = np.sqrt(dfpbar['AmpBot'+i]*dfpbar['AmpTop'+i])
        dfpbar['z'+i] = np.log(dfpbar['AmpBot'+i]/dfpbar['AmpTop'+i])

    dfpbar['w'+i] = dfpbar['P'+i]*dfpbar['P'+i]
    dfpbar['iw'+i] = 1./dfpbar['P'+i]/dfpbar['P'+i]
    dfpbar['iwz'+i] = dfpbar['z'+i]*dfpbar['iw'+i]
        
    dfpbar['x'+i] = np.cos(int(i) * 2 * np.pi / len(ibots))
    dfpbar['y'+i] = np.sin(int(i) * 2 * np.pi / len(ibots))
    dfpbar['wx'+i] = dfpbar['x'+i]*dfpbar['w'+i]
    dfpbar['wy'+i] = dfpbar['y'+i]*dfpbar['w'+i]
    dfpbar['wxy'+i] = dfpbar['x'+i]*dfpbar['y'+i]*dfpbar['w'+i] 
    dfpbar['wxx'+i] = dfpbar['x'+i]*dfpbar['x'+i]*dfpbar['w'+i] 
    dfpbar['wyy'+i] = dfpbar['y'+i]*dfpbar['y'+i]*dfpbar['w'+i] 
        
    dfpbar['dt'+i] = dfpbar['TDCBot'+i]-dfpbar['TDCTop'+i]
    dfpbar['t'+i] = dfpbar['TDCBot'+i]+dfpbar['TDCTop'+i]


    
dfpbar['N']    = dfpbar.filter(regex='Amp.*').count(axis=1) 
dfpbar['Eavg'] = dfpbar.filter(regex='^P[0-9]+').mean(axis=1,skipna=True)
dfpbar['sE']   = dfpbar.filter(regex='^P[0-9]+').std(axis=1,skipna=True)/dfpbar['Eavg']
dfpbar['tavg'] = dfpbar.filter(regex='^t[0-9]+').mean(axis=1,skipna=True) 
dfpbar['st']   = dfpbar.filter(regex='^t[0-9]+').std(axis=1,skipna=True)

dfpbar['W']    = dfpbar.filter(regex='^w[0-9]+').sum(axis=1,skipna=True)

print(dfpbar.filter(regex='^w[0-9]+').columns.values)

dfpbar['IW']   = dfpbar.filter(regex='iw[0-9]+').sum(axis=1,skipna=True)

dfpbar['ZA']   = dfpbar.filter(regex='iwz[0-9]+').sum(axis=1,skipna=True)/dfpbar['IW']
dfpbar['X']   = dfpbar.filter(regex='wx[0-9]+').sum(axis=1,skipna=True)/dfpbar['W']
dfpbar['Y']   = dfpbar.filter(regex='wy[0-9]+').sum(axis=1,skipna=True)/dfpbar['W']
dfpbar['r']   = np.sqrt(dfpbar['X']**2 + dfpbar['Y']**2)

dfpbar['Sxy']   = dfpbar.filter(regex='wxy[0-9]+').sum(axis=1,skipna=True)/dfpbar['W'] - dfpbar['X']*dfpbar['Y']
dfpbar['Sxx']   = dfpbar.filter(regex='wxx[0-9]+').sum(axis=1,skipna=True)/dfpbar['W'] - dfpbar['X']*dfpbar['X']
dfpbar['Syy']   = dfpbar.filter(regex='wyy[0-9]+').sum(axis=1,skipna=True)/dfpbar['W'] - dfpbar['Y']*dfpbar['Y']

with errstate(divide='ignore', invalid='ignore'):       
    dfpbar['Corr'] = dfpbar['Sxy']/np.sqrt(dfpbar['Sxx']*dfpbar['Syy'])

# now remove the columns that refer to individual bars (i.e., names end with numbers)


logger.info(dfpbar)


logger.info(dfpbar.filter(regex="[a-zA-Z_]+$").columns)



dfpbar.fillna(value=0,inplace=True)

logger.info(dfpbar[['EventTime', 'ZA', 'N', 'Eavg', 'sE', 'st', 'r', 'Sxx', 'Syy', 'Corr']]) #dfpbar.filter(regex="[a-zA-Z_]+$").columns])



#labels = list(filter(r.match, dfpbar.columns.values))
#idx = np.array([ int(re.sub(r'P*', '', s)) for s in labels])
#energies=dfpbar[labels].to_numpy(na_value=0)

'''


# now sort labels 


# sort 

indices = energies.argsort()
logger.info(indices.shape)
logger.info(idx.shape)
logger.info(energies.shape)

newI = idx
newP = np.take_along_axis(P,indices,axis=1)


energies[indices] 
'''



#r = re.compile("AmpTop*")
#atops = list(filter(r.match, dfpbar.columns.values))
#itops = [ re.sub(r'[a-zA-Z]*', '', s) for s in atops ] 

#r = re.compile("AmpBot*")
#abots = list(filter(r.match, dfpbar.columns.values))
#ibots = [ re.sub(r'[a-zA-Z]*', '', s) for s in abots ]




#dfpbar.fillna(value=dict(zip(atops, [0]*len(atops))),inplace=True)

#dfpbar[atops] = dfpbar[atops].where(dfpbar[atops]>0,0)
#dfpbar[abots] = dfpbar[abots].where(dfpbar[abots]>0,0)



#topidx = itops.argsort()
#botidx = ibots.argsort()

#atops



#topamps = dfpbar[atops].to_numpy(na_value=1e-6)
#botamps = dfpbar[abots].to_numpy(na_value=1e-6)

#logger.info(itops.shape)
#logger.info(topamps.shape)



#logger.info(topidx.shape)


#idx = itops[topidx]
#P = np.sqrt( topamps[:,topidx] * botamps[:,botidx] )
#R = np.log( topamps[:,topidx] / botamps[:,botidx] )

#logger.info(idx.shape)
#logger.info(itops)
#logger.info(itops[topidx])

# sort by energy
#indices = P.argsort(axis=1)[:,::-1]

#logger.info(R.shape)
#Pi = np.take_along_axis(P,indices,axis=1)
#Ri = np.take_along_axis(R,indices,axis=1)
#idxi = np.take_along_axis(idx,indices,axis=0)



#logger.info(np.take_along_axis(P,indices,axis=1)[2])
#logger.info(P[2])

#logger.info(R[indices])
#logger.info(idx[indices])


#>>> A.argsort()
#array([[1, 0, 2, 3],
#       [1, 2, 0, 3],
#       [1, 0, 2, 3]])
#>>> idx=np.array([5, 2, 3, 9])
#
#>>> indices = A.argsort()[:,::-1]
#>>> 



# pass list of substitutions
#dfpbar.fillna(value=dict(zip(atops, [0]*len(atops))),inplace=True)






# this should be probably postponed , after inspection of eventtime 
'''
#reduce the EventTime ranges in the two trees; here values are hard-coded 
dfcosmic['EventTimeBin'] = pd.cut(dfcosmic['EventTime'],bins=np.linspace(10,160,101),labels=np.arange(0,100))
dfpbar['EventTimeBin'] = pd.cut(dfpbar['EventTime'],bins=np.linspace(500,650,101),labels=np.arange(0,100))

#remove out of range events (bin is NaN)
dfcosmic.dropna(subset=['EventTimeBin'],inplace=True)
dfpbar.dropna(subset=['EventTimeBin'],inplace=True)

cntcosmic = dfcosmic['EventTimeBin'].value_counts(sort=False)
bkg = np.mean(cntcosmic)

#dfcosmic['EventTimeBin'].fillna(value=0,inplace=True)
#dfpbar['EventTimeBin'].fillna(value=0,inplace=True)

dfpbar['count'] = dfpbar.groupby('EventTimeBin')['EventTimeBin'].transform('count')
dfcosmic['count'] = dfcosmic.groupby('EventTimeBin')['EventTimeBin'].transform('count')
#https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html


'''

#dfpbar.columns.to_series().str.contains("AmpTop")



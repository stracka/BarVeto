# 
# Utilities for dealing with datasets 
#
# 
#
#   # !pip freeze ! grep uproot
#

import logging

import uproot
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

FORMAT = '%(asctime)-15s- %(levelname)s - %(name)s -%(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG)
logger = logging.getLogger(__name__)


#if __name__ == '__main__':
#    print('Welcome!')
    

def df_from_root(filename,treename,columnsre=None):
    """
    converting from root to a format suitable for manipulation 
    """

    # prepare data: open file and load tree, then convert to dataframe
    logger.info(' Preparing data...')

    f_ = uproot.open(filename)
    t_ = f_[treename]
    
    logger.info('Entries in file 0: ' + str(t_.num_entries))
    logger.info(t_.keys())

    df = t_.arrays(library="pd")

    
    # cleanup and add features 
    logger.info(' Cleaning up dataset and adding features ...')

    ibots = [ re.sub(r'[a-zA-Z]*', '', s) for s in df.filter(regex="AmpBot.*").columns.values ]

    for i in ibots:

        # replace unphysical values in TDC (failed fits) with nan
        df.loc[df['TDCBot'+i] < 50, 'TDCBot'+i] = np.nan
        df.loc[df['TDCTop'+i] < 50, 'TDCTop'+i] = np.nan
        
        with np.errstate(divide='ignore', invalid='ignore'):           #  https://stackoverflow.com/questions/21752989/numpy-efficiently-avoid-0s-when-taking-logmatrix
            df['P'+i] = np.sqrt(df['AmpBot'+i]*df['AmpTop'+i])
            df['z'+i] = np.log(df['AmpBot'+i]/df['AmpTop'+i])

        df['w'+i] = df['P'+i]*df['P'+i]
        df['iw'+i] = 1./df['P'+i]/df['P'+i]
        df['iwz'+i] = df['z'+i]*df['iw'+i]
        
        df['x'+i] = np.cos(int(i) * 2 * np.pi / len(ibots))
        df['y'+i] = np.sin(int(i) * 2 * np.pi / len(ibots))
        df['wx'+i] = df['x'+i]*df['w'+i]
        df['wy'+i] = df['y'+i]*df['w'+i]
        df['wxy'+i] = df['x'+i]*df['y'+i]*df['w'+i] 
        df['wxx'+i] = df['x'+i]*df['x'+i]*df['w'+i] 
        df['wyy'+i] = df['y'+i]*df['y'+i]*df['w'+i] 
        
        df['dt'+i] = df['TDCBot'+i]-df['TDCTop'+i]
        df['t'+i] = df['TDCBot'+i]+df['TDCTop'+i]
        

    df['N']    = df.filter(regex='Amp.*').count(axis=1)
    df['logN'] = np.log(df.filter(regex='Amp.*').count(axis=1))
    df['Eavg'] = df.filter(regex='^P[0-9]+').mean(axis=1,skipna=True)
    df['sE']   = df.filter(regex='^P[0-9]+').std(axis=1,skipna=True)/df['Eavg']
    df['tavg'] = df.filter(regex='^t[0-9]+').mean(axis=1,skipna=True) 
    df['st']   = df.filter(regex='^t[0-9]+').std(axis=1,skipna=True)
        
    df['W']    = df.filter(regex='^w[0-9]+').sum(axis=1,skipna=True)
        
    #print(df.filter(regex='^w[0-9]+').columns.values)

    df['IW']   = df.filter(regex='iw[0-9]+').sum(axis=1,skipna=True)
       
    df['ZA']   = df.filter(regex='iwz[0-9]+').sum(axis=1,skipna=True)/df['IW']
    df['X']   = df.filter(regex='wx[0-9]+').sum(axis=1,skipna=True)/df['W']
    df['Y']   = df.filter(regex='wy[0-9]+').sum(axis=1,skipna=True)/df['W']
    df['r']   = np.sqrt(df['X']**2 + df['Y']**2)
    
    df['Sxy']   = df.filter(regex='wxy[0-9]+').sum(axis=1,skipna=True)/df['W'] - df['X']*df['Y']
    df['Sxx']   = df.filter(regex='wxx[0-9]+').sum(axis=1,skipna=True)/df['W'] - df['X']*df['X']
    df['Syy']   = df.filter(regex='wyy[0-9]+').sum(axis=1,skipna=True)/df['W'] - df['Y']*df['Y']

    with np.errstate(divide='ignore', invalid='ignore'):
        df['Corr'] = df['Sxy']/np.sqrt(df['Sxx']*df['Syy'])
    df['Corr'].replace([np.inf], 1, inplace=True)
    df['Corr'].replace([-np.inf], 1, inplace=True)
    df.loc[df['Corr']>1, 'Corr'] = 1
    df.loc[df['Corr']<-1, 'Corr'] = -1

    # remove outliers ; this might be a pileup event of some sort 
    logger.info('Should keep record of selection of outliers; also, this should probably be done elsewhere') 
    df.loc[df['st']>100,'st'] = np.nan
    
    
    logger.info(df)

    if (columnsre):
        return df.filter(columnsre)
    else : 
        return df



# 
# Utilities for dealing with datasets 
#
# Add function testing. 
#
#   # !pip freeze ! grep uproot
#

import logging

import uproot
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import random

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

    return df



def df_gen_col_name(colname, notinlist):
    newname = colname
    while newname in notinlist: # add random suffix if column with same name exists
        newname = ''.join(random.choice(string.ascii_lowercase))

    return newname




def df_cleanup(data,ranges=None):

    count = 0

    df = data.copy()
    
    for i in ranges.keys():

        selrange = ranges[i]
        thevars = df.filter(regex=i).columns.values        

        for j in thevars:

            count+=len(df.loc[df[j] < selrange[0]])
            count+=len(df.loc[df[j] > selrange[1]])
            
            df.loc[df[j] < selrange[0], j] = selrange[2]
            df.loc[df[j] > selrange[1], j] = selrange[3]

    logger.info(f"Found and replaced {count} out-of-range elements")
            
    return df





def df_category(data,catname,ranges=None):
    """
    the ranges are OR'ed
    """

    df = data.copy()

    df[catname] = 1

    count = 0
    for i in ranges.keys():

        selrange = ranges[i]
        thevars = df.filter(regex=i).columns.values        

        for j in thevars:

            count+=len(df.loc[df[j] < selrange[0]])
            count+=len(df.loc[df[j] > selrange[1]])
            
            df.loc[df[j] < selrange[0], catname] = 0
            df.loc[df[j] > selrange[1], catname] = 0

    logger.info(f"Found {count} out-of-range elements")
            
    return df





def df_transform_vars(data,columnsre=None):
    """ 
    juggle features 
    """

    df = data.copy()
    
    logger.info('Replacing raw features with derived ones ...')

    ibar = [ re.sub(r'[a-zA-Z]*', '', s) for s in df.filter(regex="Bar[0-9]+").columns.values ]

    for i in ibar:

        df['dt'+i] = df['TDCBot'+i]-df['TDCTop'+i]
        df['t'+i] = df['TDCBot'+i]+df['TDCTop'+i]
        
        with np.errstate(divide='ignore', invalid='ignore'):
            #  https://stackoverflow.com/questions/21752989/numpy-efficiently-avoid-0s-when-taking-logmatrix
            df['P'+i] = np.sqrt(df['AmpBot'+i]*df['AmpTop'+i])
            df['z'+i] = np.log(df['AmpBot'+i]/df['AmpTop'+i])

        df['x'+i] = np.cos(int(i) * 2 * np.pi / len(ibar))
        df['y'+i] = np.sin(int(i) * 2 * np.pi / len(ibar))

    if (columnsre):
        savedcols = []
        for patt in columnsre: 
            r = re.compile(patt)
            savedcols += list(filter(r.match, df.columns.values)) 
        logger.info(f'These columns will be saved: {savedcols}')
        return df.filter(items=savedcols)
    else:
        return df
    

def df_extra_vars(data,columnsre=None):
    """  
    these variables combine the output of several columns:
    a cleanup should be applied before calculating them
    """

    df = data.copy()
    
    ibar = [ re.sub(r'[a-zA-Z]*', '', s) for s in df.filter(regex="^P[0-9]+").columns.values ]

    for i in ibar:
            
        df['w'+i] = df['P'+i]*df['P'+i]
        df['iw'+i] = 1./df['P'+i]/df['P'+i]
        df['iwz'+i] = df['z'+i]*df['iw'+i]
        df['wx'+i] = df['x'+i]*df['w'+i]
        df['wy'+i] = df['y'+i]*df['w'+i]
        df['wxy'+i] = df['x'+i]*df['y'+i]*df['w'+i] 
        df['wxx'+i] = df['x'+i]*df['x'+i]*df['w'+i] 
        df['wyy'+i] = df['y'+i]*df['y'+i]*df['w'+i] 
        
        
    df['N']    = df.filter(regex='^P[0-9]+').count(axis=1)/2
    df['logN'] = np.log(df['N'])

    df['Eavg'] = df.filter(regex='^P[0-9]+').mean(axis=1,skipna=True)
    df['sE']   = df.filter(regex='^P[0-9]+').std(axis=1,skipna=True)/df['Eavg']
    df['tavg'] = df.filter(regex='^t[0-9]+').mean(axis=1,skipna=True) 
    df['st']   = df.filter(regex='^t[0-9]+').std(axis=1,skipna=True)
    df['dtavg']= df.filter(regex='^dt[0-9]+').mean(axis=1,skipna=True) 
    df['sdt']  = df.filter(regex='^dt[0-9]+').std(axis=1,skipna=True) # equivalent to Marta's delta_z
    logger.info(str(df['st'].max())+' '+str(df['sdt'].max()))
    
    df['W']    = df.filter(regex='^w[0-9]+').sum(axis=1,skipna=True)        
    df['IW']   = df.filter(regex='iw[0-9]+').sum(axis=1,skipna=True)
       
    df['ZA']   = df.filter(regex='iwz[0-9]+').sum(axis=1,skipna=True)/df['IW']
    df['X']   = df.filter(regex='wx[0-9]+').sum(axis=1,skipna=True)/df['W']
    df['Y']   = df.filter(regex='wy[0-9]+').sum(axis=1,skipna=True)/df['W']
    df['r']   = np.sqrt(df['X']**2 + df['Y']**2)
    
    df['Sxy']   = df.filter(regex='wxy[0-9]+').sum(axis=1,skipna=True)/df['W'] - df['X']*df['Y']
    df['Sxx']   = df.filter(regex='wxx[0-9]+').sum(axis=1,skipna=True)/df['W'] - df['X']*df['X']
    df['Syy']   = df.filter(regex='wyy[0-9]+').sum(axis=1,skipna=True)/df['W'] - df['Y']*df['Y']

    with np.errstate(divide='ignore', invalid='ignore'):
        df['Corr'] = np.abs(df['Sxy']/np.sqrt(df['Sxx']*df['Syy']))
    df['Corr'].replace([np.inf], 1, inplace=True)
    df['Corr'].replace([-np.inf], 1, inplace=True)
    df.loc[df['Corr']>1, 'Corr'] = 1
    df.loc[df['Corr']<-1, 'Corr'] = -1

    logger.info(columnsre) 
    logger.info(df)

    if (columnsre):
        savedcols = []
        for patt in columnsre: 
            r = re.compile(patt)
            savedcols += list(filter(r.match, df.columns.values)) 
        logger.info(f'These columns will be saved: {savedcols}')
        return df.filter(items=savedcols)
    else:
        return df




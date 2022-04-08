# 
# Utilities for plotting
#
# 
#
#   # !pip freeze ! grep uproot
#

import logging
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

FORMAT = '%(asctime)-15s- %(levelname)s - %(name)s -%(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG)
logger = logging.getLogger(__name__)


def plot_corr(df,thevars=None):
    """ 
    Plot correlation matrix 
    """

    df = df.fillna(value=0)

    if thevars==None:
        thevars=df.columns
        
    corr = np.abs(df[thevars].corr())
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(corr,0)] = True
    corr[mask]=np.nan
    
    plt.matshow(corr)
    for (x, y), value in np.ndenumerate(corr):
        if (y<x):
            plt.text(y, x, f"{value:.2f}", va="center", ha="center")        
    plt.show()

    corr = corr.stack().reset_index()

    logger.info(corr)

    return plt



def plot_pairs(df, thevars=None) : 
    
    df = df.fillna(value=0)

    if thevars==None:
        thevars=df.columns
        
    pairs = list(itertools.combinations(thevars,2))

    nvars = len(thevars)
    if (nvars%2):
        n = nvars-1
        m = nvars
    else:
        n = nvars
        m = nvars-1
    n=int(n/2)
        
    fig, axs = plt.subplots(nrows=n, ncols=m)
        
    for ax, ind in zip(axs.flat, pairs):
        ax.set_title('')
        ax.hist2d(x=df[ind[0]], y=df[ind[1]], bins=(50,50))
        ax.set_xlabel(ind[0])
        ax.set_ylabel(ind[1])

    return plt



def plot_diff(df1, df2, thevars=None, nrows=3, ncols=3, bins=100) :
    """ 

    """
    
    df1 = df1.fillna(value=0)
    df2 = df2.fillna(value=0)

    if thevars==None:
        thevars=[col for col in df1.columns if col in df2.columns]

    nvars = len(thevars)

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols)
    
    for ax, var in zip(axs.flat, thevars):
        _hs,_edges = np.histogram(df1[var].to_numpy(), bins=bins)
        _hb,_edges = np.histogram(df2[var].to_numpy(), bins=_edges)

        width = 1.0 * (_edges[1] - _edges[0])
        center = (_edges[:-1] + _edges[1:]) / 2
        hists = (_hs-_hb)/np.sum(_hs-_hb)
        histb = _hb/np.sum(_hb)
        
        ax.bar(center, hists, align='center', width=width)
        ax.bar(center, histb, align='center', width=width, alpha = 0.6)
        ax.set_xlabel(var)
        
    return plt
            
'''

#selection=(np.abs(dfpbar.TDCTop0)<50) #  & (dfpbar.Bar0==0)
#dfpbar[selection].plot(x='dt0',y='z0',kind='hexbin',gridsize=50) #,vmax=10)
#plt.hist(dfpbar['dt0'],bins=100) #,vmax=10)
#plt.show()

'''

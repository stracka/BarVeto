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
logging.basicConfig(format=FORMAT, level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_corr(data,thevars=None):
    """ 
    Plot correlation matrix 
    """
    
    df = data.fillna(value=0)   # this returns a copy of the original dataframe

    if thevars==None:
        thevars=df.columns
        
    corr = np.abs(df[thevars].corr())
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(corr,0)] = True
    corr[mask]=np.nan

    fig, ax = plt.subplots()
    ax.matshow(corr)

    ax.set_xticks(range(len(thevars)))
    ax.set_yticks(range(len(thevars)))
    ax.set_xticklabels(thevars)
    ax.set_yticklabels(thevars)
    
    for (x, y), value in np.ndenumerate(corr):
        if (y<x):
            ax.text(y, x, f"{value:.2f}", va="center", ha="center")        
        
    corr = corr.stack().reset_index()

    logger.info(corr)

    return plt



def plot_pairs(data, thevars=None) : 
    
    df = data.fillna(value=0)   # this returns a copy of the original dataframe

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
        
    fig, axs = plt.subplots(nrows=n, ncols=m, squeeze=False)

    for ax, ind in zip(axs.flat, pairs):
        ax.set_title('')
        ax.hist2d(x=df[ind[0]], y=df[ind[1]], bins=(50,50))
        ax.set_xlabel(ind[0])
        ax.set_ylabel(ind[1])

    return plt



def plot_diff(data, targetvar, sbvar=None, varstoplot=None, nrows=3, ncols=3, bins=100) :
    """ 
    plot dfs (signal) - dfsb (sideband) vs dfbkg (background) 
    """

    df = data.fillna(value=0)  # this returns a copy of the original dataframe 

    if (sbvar):
        dfs  = df.loc[(df[targetvar]==1) & (df[sbvar]==1)]
        dfsb = df.loc[(df[targetvar]==0) & (df[sbvar]==1)]
        dfbkg  = df.loc[(df[targetvar]==0)]
    else:
        dfs  = df.loc[(df[targetvar]==1)]
        dfsb = df.loc[(df[targetvar]==0)]
        dfbkg  = df.loc[(df[targetvar]==0)]
        
    if varstoplot==None:
        varstoplot=[col for col in df.columns]

    nvars = len(varstoplot)

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False)
    
    for ax, var in zip(axs.flat, varstoplot):
        _hs,_edges  = np.histogram(dfs[var].to_numpy(),   bins=bins)        
        _hsb,_edges = np.histogram(dfsb[var].to_numpy(),  bins=_edges)
        _hb,_edges  = np.histogram(dfbkg[var].to_numpy(), bins=_edges)            
            
        width = 1.0 * (_edges[1] - _edges[0])
        center = (_edges[:-1] + _edges[1:]) / 2

        with np.errstate(divide='ignore', invalid='ignore'):        
            hists = (_hs-_hsb)/np.sum(_hs-_hsb)
            histb = _hb/np.sum(_hb)
            np.nan_to_num(hists, copy=False, nan=0.0, posinf=0.0, neginf=0.0) 
            np.nan_to_num(histb, copy=False, nan=0.0, posinf=0.0, neginf=0.0) 

        ax.bar(center, hists, align='center', width=width)
        ax.bar(center, histb, align='center', width=width, alpha = 0.6)
        ax.set_xlabel(var)
        
    return plt



def plot_roc(y, out, sb=None) : 
    """
    ToDo: replace ns0 etc. with a formula related to a matrix of the probabilities in each bag
    """ 
        
    data = np.column_stack([y,out])
    df = pd.DataFrame(data,columns=['y','out'])

    if sb is not None:
        df['sb'] = sb
    else:
        df['sb'] = 1

    minout = df['out'].min()
    maxout = df['out'].max()
    
    seff=np.array([])
    beff=np.array([])
    yts =np.array([])
    
    ns0=len(df.loc[ (df.y==1) & (df.sb==1) ]) - len(df.loc[ (df.y==0) & (df.sb==1) ]) 
    nb0=len(df.loc[ (df.y==0) ] )
    
    for yt in np.linspace(minout,maxout,100):
        ns=len(df.loc[ (df.y==1) & (df.sb==1) & (df.out>yt)]) - len(df.loc[ (df.y==0) & (df.sb==1)  & (df.out>yt)]) 
        nb=len(df.loc[ (df.y==0) & (df.out>yt) ] )

        yts = np.append(yts, yt)
        seff = np.append(seff,ns/ns0)
        beff = np.append(beff,nb/nb0)

    brej = 1.-beff

    plt.plot(brej,seff)
    return plt





'''
df_roc.plot(x='effS',y='rateB')
plt.yscale('log')
plt.xticks(ticks=list(np.linspace(0,1,11)))
plt.xlabel("signal efficiency")
plt.ylabel("background rate (Hz)")
plt.grid()
plt.savefig('rate.png')
plt.clf()
'''




'''


#dfpbar[selection].plot(x='dt0',y='z0',kind='hexbin',gridsize=50) #,vmax=10)
#plt.hist(dfpbar['dt0'],bins=100) #,vmax=10)
#plt.show()

'''

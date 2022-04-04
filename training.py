# Training a classifier starting from ROOT files
#
# 
#
#   # !pip freeze ! grep uproot

import logging

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


FORMAT = '%(asctime)-15s- %(levelname)s - %(name)s -%(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG)
logger = logging.getLogger(__name__)


#if __name__ == '__main__':
#    print('Welcome!')
    
from dataset_utils import *

columnsre=None
# columnsre = "[a-zA-Z_]+$" # this removes the columns that refer to individual bars (i.e., names end with numbers)

dfpbar   = df_from_root("OutBarTree_run6201_pbarhold.root","Tree_BarData",columnsre)
dfcosmic = df_from_root("OutBarTree_run6201_cosmics.root","Tree_BarData",columnsre)
    
selection=(np.abs(dfpbar.dt0) < 5e-8)#  & (dfpbar.Bar0==0)
dfpbar[selection].plot(x='dt0',y='z0',kind='hexbin',gridsize=50) #,vmax=10)
plt.show()

dfpbar.fillna(value=0,inplace=True)

# now I want to make useful plots, for pbar and cosmics
# are the variables calibrated ? 



#topamps = dfpbar[atops].to_numpy(na_value=1e-6)
#botamps = dfpbar[abots].to_numpy(na_value=1e-6)

#reduce the EventTime ranges in the two trees; here values are hard-coded 
#dfcosmic['EventTimeBin'] = pd.cut(dfcosmic['EventTime'],bins=np.linspace(10,160,101),labels=np.arange(0,100))
#dfpbar['EventTimeBin'] = pd.cut(dfpbar['EventTime'],bins=np.linspace(500,650,101),labels=np.arange(0,100))

#remove out of range events (bin is NaN)
#dfcosmic.dropna(subset=['EventTimeBin'],inplace=True)
#dfpbar.dropna(subset=['EventTimeBin'],inplace=True)

#cntcosmic = dfcosmic['EventTimeBin'].value_counts(sort=False)
#bkg = np.mean(cntcosmic)

#dfpbar['count'] = dfpbar.groupby('EventTimeBin')['EventTimeBin'].transform('count')
#dfcosmic['count'] = dfcosmic.groupby('EventTimeBin')['EventTimeBin'].transform('count')
#https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html




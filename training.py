# Training a classifier starting from ROOT files
# 
#   # !pip freeze ! grep uproot


import logging
FORMAT = '%(asctime)-15s- %(levelname)s - %(name)s -%(message)s'
#logging.basicConfig(format=FORMAT, level=logging.DEBUG)
logging.basicConfig(format=FORMAT, level=logging.INFO)
logger = logging.getLogger(__name__)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

from sklearn.pipeline        import Pipeline
from sklearn.impute          import SimpleImputer  #, MissingIndicator
from sklearn.preprocessing   import StandardScaler
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, validation_curve
#cross_val_score, train_test_split, RandomizedSearchCV #https://scikit-learn.org/stable/modules/cross_validation.html
#from sklearn.metrics         import accuracy_score

    
from dataset_utils import *
from plot_utils import * 

#https://docs.python.org/3/library/argparse.html

#if __name__ == '__main__':
#    print('Welcome!')


# load datasets 
columnsre=None

dfpbar   = df_from_root("OutBarTree_run6201_pbarhold.root","Tree_BarData",columnsre)
dfcosmic = df_from_root("OutBarTree_run6201_cosmics.root","Tree_BarData",columnsre)


# add classification target
target = df_gen_col_name('y',list(dfcosmic.columns.values) + list(dfpbar.columns.values))
dfcosmic[target] = 0
dfpbar[target] = 1



# concatenate datasets
dfall = pd.concat([dfcosmic,dfpbar])
logger.info(str(dfall.shape)+' : '+str(dfpbar.shape) + ' + ' + str(dfcosmic.shape))



# cleanup raw dataset 
selections = {
    '^TDC[a-zA-Z_]*[0-9]+' : (50 , np.inf, np.nan, np.nan)
    }
dfall = df_cleanup(dfall, ranges=selections)


# transform variables 
dfall = df_transform_vars(dfall,columnsre=['^P[0-9]+','^z[0-9]+','^dt[0-9]+','^t[0-9]+', '^x[0-9]+', 'y[0-9]+', '^'+target+'$', '^EventTime$'])


# cleanup transformed dataset 
selections = {
    '^P[0-9]+' : (-np.inf , np.inf, np.nan, np.nan),
    '^z[0-9]+' : (-10 , 10 , np.nan, np.nan), 
    '^dt[0-9]+': (-1.0e-7 , 1.0e-7, np.nan, np.nan),  
    '^t[0-9]+' : (50 , np.inf, np.nan, np.nan)
    }
dfall = df_cleanup(dfall, ranges=selections)

logger.info(dfall)
#dfall.plot(x='dt0',y='z0',kind='hexbin',gridsize=50) #,vmax=10)  #plt.show()  #plt.clf()



dfall = df_pair_vars(dfall)
logger.info(dfall)


# add and shrink variables 
columnsre = ["[a-zA-Z_]+$"] # vars for bar end w/ number and are dropped
dfall = df_extra_vars(dfall,columnsre)
logger.info(dfall.loc[dfall.y==0,'ZA'].mean())

# cleanup transformed dataset 
selections = {
    '^st$' : (-1e-7, 1e-7, np.nan, np.nan),
    '^[a-zA-Z]+_tof$' : (-1e-10, 2e-8,-1e-10,2e-8),
    '^[a-zA-Z]+_vel$' : (-1e-6, 1e-6, np.nan, np.nan),
    }
dfall = df_cleanup(dfall, ranges=selections)

logger.info(dfall)



# add signalbox category
sbcat = df_gen_col_name('signalbox',list(dfall.columns.values))
signalbox = {
    '^ZA$' : (0,np.inf),
    }
dfall = df_category(dfall, catname=sbcat, ranges=signalbox)


# enrich signal sample by removing signal events not in signalbox
#dfall = dfall.loc[ (dfall[target]==0) | (( dfall[target]==1 ) & (dfall[sbcat]==1)) ]


logger.info(dfall)




# variable selection 
featlist = ['logN','Eavg','sE','r','Corr','min_tof','mean_tof','max_tof','std_tof','min_halfchord','max_halfchord','mean_halfchord','std_halfchord'] 
for ifeat in featlist:  # check the variables are in varlist, remove otherwise
    if not ifeat in list(dfall.columns.values):
        featlist.remove(ifeat)

auxlist  = ['ZA','EventTime'] 
for ifeat in auxlist:  # check the variables are in varlist, remove otherwise
    if not ifeat in list(dfall.columns.values):
        auxlist.remove(ifeat)

logger.info('all variables ' + str(dfall.columns.values))
logger.info('train on ' + str(featlist) + 'project on' + str(auxlist) )




plot_corr(dfall,featlist)
plt.savefig('correlations.png')
#plt.show()
plt.clf()

#plot_pairs(dfall,featlist)
#plt.show()
#plt.clf()


nsel=0
plot_diff( dfall.loc[dfall.N>0],
           targetvar=target, sbvar=None, #sbcat,
           ncols=5, nrows=3,
           varstoplot=featlist+auxlist ) 
plt.savefig("features.png")
#plt.show()
plt.clf()



# create iterator
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
outer_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)

estimators = [('imputer', SimpleImputer(strategy='constant', fill_value=0)), ('RF', RandomForestClassifier(random_state=0))]
pipe = Pipeline(estimators)

logger.info('Pipeline parameters:\n' + str(pipe.get_params()))

p_grid = { "RF__max_depth" : [5,None],
           "RF__n_estimators" : [100] ,
           "RF__min_samples_split" : [2,5] , 
           "RF__max_features" : [0.4,1.] }

clf = GridSearchCV(estimator=pipe, param_grid=p_grid, cv=inner_cv, scoring='roc_auc')

# in principle, could get param_name and range from p_grid
# use flat cross-validation on one 'outer split'


# prepare datasets and check for any nan feature
aux = dfall[auxlist].to_numpy()
sb = dfall[sbcat].to_numpy()

y = dfall[target].to_numpy()
n_samples = len(dfall)
itrain,itest = next(outer_cv.split(np.zeros(n_samples),y))


y_train, y_test = y[itrain], y[itest]
aux_train, aux_test = aux[itrain, :], aux[itest, :]
sb_train, sb_test = sb[itrain], sb[itest]


X = dfall[['logN']].to_numpy()
X_train, X_test = X[itrain, :], X[itest, :]
y_out3 = X_test[:,0]


#featlist = ['logN','Eavg','sE','r','Corr','max_tof','max_halfchord','std_halfchord','min_vel','max_vel','mean_vel','std_vel']
#featlist = ['logN','Eavg','sE','r','Corr','mean_tof','min_halfchord','max_halfchord','mean_halfchord','std_halfchord']
featlist = ['logN','Eavg','sE','r','Corr']
X = dfall[featlist].to_numpy()
X_train, X_test = X[itrain, :], X[itest, :]

# get the best estimator on this train/test
clf.fit(X_train,y_train)
logger.info("Tuned best params: {}".format(clf.best_params_))

for var, imp in zip(featlist, clf.best_estimator_.steps[1][1].feature_importances_):
    logger.info(f'{var}: {imp}')

y_out1 = clf.predict_proba(X_test)[:,1]
#y_pred = clf.predict(X_test)
    


#featlist = ['logN','Eavg','sE','r','Corr','min_tof','max_tof','min_halfchord','max_halfchord','std_halfchord','min_vel','max_vel'] 
featlist = ['logN','Eavg','sE','r','Corr','min_halfchord','max_halfchord','mean_halfchord','std_halfchord']
X = dfall[featlist].to_numpy()
X_train, X_test = X[itrain, :], X[itest, :]

# get the best estimator on this train/test
clf.fit(X_train,y_train)
logger.info("Tuned best params: {}".format(clf.best_params_))

for var, imp in zip(featlist, clf.best_estimator_.steps[1][1].feature_importances_):
    logger.info(f'{var}: {imp}')

y_out2 = clf.predict_proba(X_test)[:,1]
#y_pred = clf.predict(X_test)


plot_roc(y_test,y_out3,sb_test)
plot_roc(y_test,y_out1,sb_test)
plot_roc(y_test,y_out2,sb_test)


ax = plt.gca()
ax.set_xlim([0.95, 1.0])
plt.savefig('roc.png')
plt.clf()


init_rate = len(dfcosmic)/(dfcosmic.EventTime.max() - dfcosmic.EventTime.min())
logger.info(f'Initial rate: {init_rate} Hz')

plot_rate(y_test,y_out3,init_rate,sb_test)
plot_rate(y_test,y_out1,init_rate,sb_test)
plot_rate(y_test,y_out2,init_rate,sb_test)

plt.savefig('rate.png')
plt.clf()



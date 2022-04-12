# Training a classifier starting from ROOT files
# 
# in principle: 
#    prepare and collate data (root to df)
#    partition data (here)
#    add features
#    tune features (scale, normalize: learned on _training_ data)
#    
#   # !pip freeze ! grep uproot

#https://docs.python.org/3/library/argparse.html

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


# add and shrink variables 
columnsre = ["[a-zA-Z_]+$"] # vars for bar end w/ number and are dropped
dfall = df_extra_vars(dfall,columnsre)


# cleanup transformed dataset 
selections = {
    '^st$' : (-1e-7, 1e-7, np.nan, np.nan),
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
dfall = dfall.loc[ (dfall[target]==0) | (( dfall[target]==1 ) & (dfall[sbcat]==1)) ]


logger.info(dfall)




# variable selection 

featlist = ['logN','Eavg','sE','r','Corr'] 
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
           targetvar=target, sbvar=sbcat,
           ncols=3, nrows=2, 
           varstoplot=featlist ) 
plt.savefig("features.png")
#plt.show()
plt.clf()





# prepare datasets and check for any nan feature
X = dfall[featlist].to_numpy()
y = dfall[target].to_numpy()

aux = dfall[auxlist].to_numpy()
sb = dfall[sbcat].to_numpy()

logger.info(str(X.shape)+ ' ' + str(y.shape))
logger.info('feature set contains nan? '+str(np.isnan(X).any()) )


# create iterator
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
outer_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)

estimators = [('imputer', SimpleImputer(strategy='constant', fill_value=0)), ('RF', RandomForestClassifier(random_state=0))]
pipe = Pipeline(estimators)

logger.info('Pipeline parameters:\n' + str(pipe.get_params()))

p_grid = { "RF__max_depth" : [5,10,None],
           "RF__n_estimators" : [100] ,
           "RF__min_samples_split" : [2,5] , 
           "RF__max_features" : [0.4,1.] }

clf = GridSearchCV(estimator=pipe, param_grid=p_grid, cv=inner_cv, scoring='roc_auc')

# in principle, could get param_name and range from p_grid
# use flat cross-validation on one 'outer split'

itrain,itest = next(outer_cv.split(X,y))
X_train, X_test = X[itrain, :], X[itest, :]
y_train, y_test = y[itrain], y[itest]

aux_train, aux_test = aux[itrain, :], aux[itest, :]
sb_train, sb_test = sb[itrain], sb[itest]

# get the best estimator on this train/test
clf.fit(X_train,y_train)
logger.info("Tuned best params: {}".format(clf.best_params_))


for var, imp in zip(featlist, clf.best_estimator_.steps[1][1].feature_importances_):
    logger.info(f'{var}: {imp}')


y_out = clf.predict_proba(X_test)[:,1]
y_pred = clf.predict(X_test)

    
plot_roc(y_test,y_out,sb_test)
plt.savefig('roc.png')
plt.clf()


'''
init_rate = len(dfcosmic)/(dfcosmic.EventTime.max() - dfcosmic.EventTime.min())
logger.info(f'Initial rate: {init_rate} Hz')
'''
'''

_a,_bins,_c = plt.hist(df_test.loc[(df_test.yUR==1) & (df_test.ZA>=0),'y_out'],density=True,bins=100)
_a,_b,_c = plt.hist(df_test.loc[(df_test.yUR==0) & (df_test.ZA>=0),'y_out'],density=True,alpha=0.6,bins=_bins)
plt.savefig('y_out.png')
plt.clf()
'''


'''
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
cm = confusion_matrix(y_test, y_pred)
logger.info(cm)
cm_display = ConfusionMatrixDisplay(cm).plot()
plt.show()
'''

'''
ns=len(df_test.loc[(df_test.y==1) & (df_test.y_out>yt)])-len(df_test.loc[(df_test.y==0) & (df_test.ZA>=0) & (df_test.y_out>yt)])
nb=len(df_test.loc[(df_test.y==0) & (df_test.y_out>yt)])
'''
    
'''
param_name = "RF__n_estimators"
param_range = [1,2,5,10,100]
train_scores, test_scores = validation_curve(
    estimator=clf.best_estimator_,
    X=X,
    y=y,
    param_name=param_name,
    param_range=param_range,
    cv=inner_cv, 
    scoring='roc_auc') 

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)


plt.title("Validation Curve with RF")
plt.xlabel(r"$n_tree$")
plt.ylabel("Score")
plt.ylim(0.0, 1.1)
lw = 2
plt.semilogx(
    param_range, train_scores_mean, label="Training score", color="darkorange", lw=lw
) 
plt.fill_between(
    param_range,
    train_scores_mean - train_scores_std,
    train_scores_mean + train_scores_std,
    alpha=0.2,
    color="darkorange",
    lw=lw,
)
plt.semilogx(
    param_range, test_scores_mean, label="Cross-validation score", color="navy", lw=lw
)
plt.fill_between(
    param_range,
    test_scores_mean - test_scores_std,
    test_scores_mean + test_scores_std,
    alpha=0.2,
    color="navy",
    lw=lw,
)
plt.legend(loc="best")
plt.show()
'''


#https://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html#sphx-glr-auto-examples-model-selection-plot-randomized-search-py

## Nested cross-validation for model selection: the inner cv generator runs on the 'split samples' determined by the outer cv generator that acts on the input sample
#cv_score = cross_val_score(clf, X=X, y=y, cv=outer_cv, scoring='roc_auc')
#logger.info("Scores for each run of cross-validation: {}".format(cv_score))
## https://machinelearningmastery.com/nested-cross-validation-for-machine-learning-with-python/
## https://arxiv.org/pdf/1809.09446.pdf


'''



ypred = clf.predict(X_train)
ypred2 = clf.predict(X_test)

logger.info(classification_report(y_train, ypred))
logger.info(classification_report(y_test, ypred2))



    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)
'''

'''
def plot_learning_curve(
    estimator,
    title,
    X,
    y,
    axes=None,
    ylim=None,
    cv=None,
    n_jobs=None,
    train_sizes=np.linspace(0.1, 1.0, 5),
):


'''

    
'''
plt.figure(figsize=(9, 6))
param_range1 = [i / 10000.0 for i in range(1, 11)]
    param_range2 = [{0: 1, 1: 6}, {0: 1, 1: 4}, {0: 1, 1: 5.5}, {0: 1, 1: 4.5}, {0: 1, 1: 5}]

    train_sizes, train_scores, test_scores = learning_curve(
        estimator=rg_cv.best_estimator_, X=X_train, y=y_train,
        train_sizes=np.arange(0.1, 1.1, 0.1), cv=cv, scoring='f1', n_jobs=- 1)

    plot_learning_curve(train_sizes, train_scores, test_scores, title='Learning curve for Logistic Regression')

    train_scores, test_scores = validation_curve(
        estimator=rg_cv.best_estimator_, X=X_train, y=y_train, param_name="clf__C", param_range=param_range1,
        cv=cv, scoring="f1", n_jobs=-1)

    plot_validation_curve(param_range1, train_scores, test_scores, title="Validation Curve for C", alpha=0.1)

    train_scores, test_scores = validation_curve(
        estimator=rg_cv.best_estimator_, X=X_train, y=y_train, param_name="clf__class_weight", param_range=param_range2,
        cv=cv, scoring="f1", n_jobs=-1)

    plot_validation_curve(param_range2, train_scores, test_scores, title="Validation Curve for class_weight", alpha=0.1)

'''

    

#>>> clf.predict(X)  # predict classes of the training data
#>>> clf.predict([[4, 5, 6], [14, 15, 16]])  # predict classes of new data


#selection=(np.abs(dfpbar.dt0) < 5e-8)#  & (dfpbar.Bar0==0)


#reduce the EventTime ranges in the two trees; here values are hard-coded 
#dfcosmic['EventTimeBin'] = pd.cut(dfcosmic['EventTime'],bins=np.linspace(10,160,101),labels=np.arange(0,100))
#dfpbar['EventTimeBin'] = pd.cut(dfpbar['EventTime'],bins=np.linspace(500,650,101),labels=np.arange(0,100))

#remove out of range events (bin is NaN)
#dfcosmic.dropna(subset=['EventTimeBin'],inplace=True)
#dfpbar.dropna(subset=['EventTimeBin'],inplace=True)

#dfpbar['count'] = dfpbar.groupby('EventTimeBin')['EventTimeBin'].transform('count')
#dfcosmic['count'] = dfcosmic.groupby('EventTimeBin')['EventTimeBin'].transform('count')
#https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html




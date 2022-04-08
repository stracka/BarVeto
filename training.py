# Training a classifier starting from ROOT files
# 
# in principle: 
#    prepare and collate data (root to df)
#    partition data (here)
#    add features
#    tune features (scale, normalize: learned on _training_ data)
#    
#   # !pip freeze ! grep uproot

import logging
FORMAT = '%(asctime)-15s- %(levelname)s - %(name)s -%(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG)
logger = logging.getLogger(__name__)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
dfcosmic['y'] = 0
dfpbar['y']   = 1


# reduce variables 
columnsre = "[a-zA-Z_]+$"  # remove columns referring to bars (i.e., names end with numbers)
varlist  = list(dfpbar.filter(regex=columnsre).columns.values)

featlist = varlist.copy()
try:
    for ivar in ['y','ZA','EventTime','tavg','Sxy','Sxx','Syy','W','IW','X','Y','N','AcosCorr']:
        featlist.remove(ivar)
    
except ValueError:
    logger.info("Error: some variables do not exist")
    
logger.info('selected ' + str(featlist) + ' out of ' + str(varlist) )


# reduce and concatenate the two datasets 
dfall = pd.concat([dfcosmic[varlist],dfpbar[varlist]])
logger.info(str(dfall.shape)+' : '+str(dfpbar.shape) + ' + ' + str(dfcosmic.shape))



# prepare datasets and check for any nan feature
X = dfall[featlist].to_numpy()
y = dfall['y'].to_numpy()
logger.info(str(X.shape)+ ' ' + str(y.shape))
logger.info('feature set contains nan? '+str(np.isnan(X).any()) )

plot_corr(dfall,featlist)
plt.show()

plot_pairs(dfall,featlist)
plt.show()

plot_diff(dfall[dfall.y==1],dfall[dfall.y==0],['ZA']+featlist)
plt.show()





# create iterator
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
outer_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)

estimators = [('imputer', SimpleImputer(strategy='constant', fill_value=0)), ('RF', RandomForestClassifier(random_state=0))]
pipe = Pipeline(estimators)

logger.info('Pipeline parameters:\n' + str(pipe.get_params()))

p_grid = { "RF__max_depth" : [5],
           "RF__n_estimators" : [100] ,
           "RF__min_samples_split" : [5] ,              
           "RF__max_features" : [0.4,0.99] }

clf = GridSearchCV(estimator=pipe, param_grid=p_grid, cv=inner_cv, scoring='roc_auc')


# in principle, could get param_name and range from p_grid
# use flat cross-validation on one 'outer split'

itrain,itest = next(outer_cv.split(X,y))
X_train, X_test = X[itrain, :], X[itest, :]
y_train, y_test = y[itrain], y[itest]

# get the best estimator on this train/test
clf.fit(X_train,y_train)
logger.info("Tuned best params: {}".format(clf.best_params_))

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
logger.info(cm)

cm_display = ConfusionMatrixDisplay(cm).plot()
plt.show()


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
#plt.show()



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

'''
# plotting as a function of train size
def plot_learning_curve(train_sizes, train_scores, test_scores, title, alpha=0.1):
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    plt.plot(train_sizes, train_mean, label='train score', color='blue', marker='o')
    plt.fill_between(train_sizes, train_mean + train_std,
                     train_mean - train_std, color='blue', alpha=alpha)
    plt.plot(train_sizes, test_mean, label='test score', color='red', marker='o')

    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, color='red', alpha=alpha)
    plt.title(title)
    plt.xlabel('Number of training points')
    plt.ylabel('F-measure')
    plt.grid(ls='--')
    plt.legend(loc='best')
    plt.show()

# plotting as a function of parameter
def plot_validation_curve(param_range, train_scores, test_scores, title, alpha=0.1):
    param_range = [x[1] for x in param_range] 
    sort_idx = np.argsort(param_range)
    param_range=np.array(param_range)[sort_idx]
    train_mean = np.mean(train_scores, axis=1)[sort_idx]
    train_std = np.std(train_scores, axis=1)[sort_idx]
    test_mean = np.mean(test_scores, axis=1)[sort_idx]
    test_std = np.std(test_scores, axis=1)[sort_idx]
    plt.plot(param_range, train_mean, label='train score', color='blue', marker='o')
    plt.fill_between(param_range, train_mean + train_std,
                 train_mean - train_std, color='blue', alpha=alpha)
    plt.plot(param_range, test_mean, label='test score', color='red', marker='o')
    plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, color='red', alpha=alpha)
    plt.title(title)
    plt.grid(ls='--')
    plt.xlabel('Weight of class 2')
    plt.ylabel('Average values and standard deviation for F1-Score')
    plt.legend(loc='best')
    plt.show()
'''
    

#>>> clf.predict(X)  # predict classes of the training data
#>>> clf.predict([[4, 5, 6], [14, 15, 16]])  # predict classes of new data


#selection=(np.abs(dfpbar.dt0) < 5e-8)#  & (dfpbar.Bar0==0)
#dfpbar[selection].plot(x='dt0',y='z0',kind='hexbin',gridsize=50) #,vmax=10)
#plt.show()

#reduce the EventTime ranges in the two trees; here values are hard-coded 
#dfcosmic['EventTimeBin'] = pd.cut(dfcosmic['EventTime'],bins=np.linspace(10,160,101),labels=np.arange(0,100))
#dfpbar['EventTimeBin'] = pd.cut(dfpbar['EventTime'],bins=np.linspace(500,650,101),labels=np.arange(0,100))

#remove out of range events (bin is NaN)
#dfcosmic.dropna(subset=['EventTimeBin'],inplace=True)
#dfpbar.dropna(subset=['EventTimeBin'],inplace=True)

#dfpbar['count'] = dfpbar.groupby('EventTimeBin')['EventTimeBin'].transform('count')
#dfcosmic['count'] = dfcosmic.groupby('EventTimeBin')['EventTimeBin'].transform('count')
#https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html




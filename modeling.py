__author__ = 'p_cohen'

import pandas as pd
import collections
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model.ridge import RidgeCV
from sklearn.linear_model import Lasso
from sklearn import svm
import numpy as np
import math
from sklearn import ensemble, preprocessing
import time
from sklearn.feature_selection import SelectKBest, f_regression
import sklearn as skl
from sklearn.feature_extraction import DictVectorizer


############### Define Globals ########################
CLN_PATH = '/home/vagrant/caterpillar-peter/Clean/'
############### Define Functions ########################
def create_val_and_train(train, seed, ids, split_rt = .20):
    """
        this will create a validate and train

        ids: this is the level of randomization, so if you want to
        randomize countries, rather than cities, you would
        set this to 'countries'

        split_rate: pct of data to assign as validation
    """
    np.random.seed(seed)
    # Get vector of de-dupped values of ids
    id_dat = pd.DataFrame(train[ids].drop_duplicates())
    # creating random vector to split train val on
    vect_len = len(id_dat.ix[:, 0])
    id_dat['rand_vals'] = (np.array(np.random.rand(vect_len,1)))
    train = pd.merge(train, id_dat, on=ids)
    # splits train into modeling and validating portions
    trn_for_mods = train[train["rand_vals"] > split_rt]
    trn_for_val = train[train["rand_vals"] <= split_rt]
    return trn_for_mods, trn_for_val

#### __author__ = 'benhamner'

def rmsle(actual, predicted):
    """
    Computes the root mean squared log error.
    This function computes the root mean squared log error between two lists
    of numbers.
    Parameters
    ----------
    actual : list of numbers, numpy array
             The ground truth value
    predicted : same type as actual
                The predicted value
    Returns
    -------
    score : double
            The root mean squared log error between actual and predicted
    """
    sle_val = (np.power(np.log(np.array(actual)+1) -
               np.log(np.array(predicted)+1), 2))
    msle_val = np.mean(sle_val)
    return np.sqrt(msle_val)

def lasso_var_select(df, feats):
    """
    Uses the lasso to select from a list of features
    :param df: Dataframe to select features using (must have 'target')
    :param feats: list of features from which to select
    :return: Selected list of features

    """
    lass = Lasso(alpha = .0005)
    lass.fit(df[feats], df['target'])
    # Keep only lassoed vars (assigned non zero coefficient)
    lassoed_vars = []
    for i in range(0, len(lass.coef_)):
        if lass.coef_[i] != 0:
            lassoed_vars.append(feats[i])
    return lassoed_vars

######################################################

### Load data ####
all_data = pd.read_csv(CLN_PATH + "full_data.csv")
non_test = all_data[all_data.is_test == 0]
test = all_data[all_data.is_test != 0]

### Create list of features
feats = list(all_data.columns.values)
non_feats = ['id', 'is_test', 'tube_assembly_id', 'cost']
for var in non_feats:
    feats.remove(var)

### Run models
avg_score = 0
num_loops = 6
test['cost'] = 0
for cv_fold in range(0, num_loops):
    # Create trn val samples
    trn, val = create_val_and_train(non_test, cv_fold, 'tube_assembly_id', .2)
    # recode target variable
    trn['target'] = trn.cost.apply(lambda x: math.log(x+1))
    # Use lasso to select variables
    lassoed_vars = lasso_var_select(trn, feats)
    # fit random forest
    frst = RandomForestRegressor(n_estimators=50, n_jobs=8)
    frst.fit(trn[lassoed_vars], trn['target'])
    # Predict and rescale predictions
    val['raw_preds'] = frst.predict(val[lassoed_vars])
    val['preds'] = val['raw_preds'].apply(lambda x: math.exp(x)-1)
    score = rmsle(val['cost'], val['preds'])
    # for i in range(0, len(frst.feature_importances_)):
    #     print "Feature %s has importance: %s" % (feats[i],
    #                                              frst.feature_importances_[i])
    print "Score is: %s" % score
    avg_score += score/num_loops
    # Predict onto test
    nm = 'preds'+str(cv_fold)
    test[nm] = frst.predict(test[lassoed_vars])
    test[nm] = test[nm].apply(lambda x: math.exp(x)-1)
    test['cost'] += test[nm]/num_loops
avg_score

# Export test preds
test['id'] = test['id'].apply(lambda x: int(x))
test[['id', 'cost']].to_csv(CLN_PATH+'randomforest from first data build.csv', index=False)

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

######################################################

### Load data ####
all_data = pd.read_csv(CLN_PATH + "full_data.csv")
non_test = all_data[all_data.is_test == 0]
test = all_data[all_data.is_test != 0]

### Create list of features
feats = list(all_data.columns.values)
non_feats = ['id', 'is_test', 'tube_assembly_id', 'cost',
             'boss_orientation_median', 'boss_orientation_max',
             'boss_orientation_min', 'boss_groove_median', 'boss_groove_min',
             'boss_groove_max', 'float_orientation_median', 'float_orientation_max',
             'float_orientation_min', 'float_bolt_pattern_long_max', 'float_bolt_pattern_wide_max',
             'float_bolt_pattern_wide_min', 'nut_orientation_median', 'nut_orientation_min',
             'nut_orientation_max', 'nut_blind_hole_median',
             'nut_blind_hole_min', 'nut_blind_hole_max',
             'adaptor_nominal_size_2_median', 'adaptor_nominal_size_2_max',
             'adaptor_nominal_size_2_min', 'adaptor_length_2_median', 'adaptor_length_2_max',
             'adaptor_length_2_min', 'adaptor_length_1_median', 'adaptor_length_1_max',
             'adaptor_length_1_min', 'adaptor_adaptor_angle_median',
             'adaptor_adaptor_angle_max', 'adaptor_adaptor_angle_min',
             'threaded_length_4_median', 'threaded_length_4_min',
             'threaded_length_4_max', 'threaded_thread_size_4_median',
             'threaded_thread_size_4_max', 'threaded_thread_size_4_min',
             'threaded_thread_pitch_4_median', 'threaded_thread_pitch_4_min',
             'threaded_thread_pitch_4_max', 'threaded_nominal_size_4_max',
             'threaded_nominal_size_4_min', 'threaded_nominal_size_4_median',
             'threaded_adaptor_angle_median', 'threaded_adaptor_angle_min',
             'threaded_adaptor_angle_max', 'sleeve_plating_max',
             'sleeve_plating_min', 'sleeve_plating_median',
             'sleeve_orientation_median', 'sleeve_orientation_min',
             'sleeve_orientation_max', 'hfl_orientation_median',
             'hfl_orientation_max', 'hfl_orientation_min']
for var in non_feats:
    feats.remove(var)

### Run models
avg_score = 0
num_loops = 6
test['cost'] = 0
for i in range(0, num_loops):
    # Create trn val samples
    trn, val = create_val_and_train(non_test, i, 'tube_assembly_id', .2)
    # recode target variable
    trn['target'] = trn.cost.apply(lambda x: math.log(x+1))
    X = trn[feats]
    val_feats = val[feats]
    # Use lasso to select variables
    lass = Lasso(alpha = .001)
    lass.fit(X, trn['target'])
    lasso_vars = [x != 0 for x in lass.coef_]
    # fit random forest
    frst = RandomForestRegressor(n_estimators=1000, n_jobs=8)
    frst.fit(X.ix[:, lasso_vars], trn['target'])
    # Predict and rescale predictions
    val['preds'] = frst.predict(val_feats.ix[:, lasso_vars])
    val['preds'] = val['preds'].apply(lambda x: math.exp(x)-1)
    score = rmsle(val['cost'], val['preds'])
    # for i in range(0, len(frst.feature_importances_)):
    #     print "Feature %s has importance: %s" % (feats[i],
    #                                              frst.feature_importances_[i])
    print "Score is: %s" % score
    avg_score += score/num_loops
    # Predict onto test
    test_for_pred = test[feats]
    nm = 'preds'+str(i)
    test[nm] = frst.predict(test_for_pred.ix[:, lasso_vars])
    test[nm] = test[nm].apply(lambda x: math.exp(x)-1)
    test['cost'] += test[nm]/num_loops

avg_score

# Export test preds
test['id'] = test['id'].apply(lambda x: int(x))
test[['id', 'cost']].to_csv(CLN_PATH+'randomforest from first data build.csv', index=False)

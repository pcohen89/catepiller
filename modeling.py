__author__ = 'p_cohen'

############## Import packages ########################
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso
import numpy as np
import math
import sys
sys.path.append('/home/vagrant/xgboost/wrapper')
import xgboost as xgb

############### Define Globals ########################
CLN_PATH = '/home/vagrant/caterpillar-peter/Clean/'
SUBM_PATH = '/home/vagrant/caterpillar-peter/Submissions/'

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

def write_preds(df, mod, cv_fold, features, is_test=1):
    """
    This writes predictions froma  model into the test data
    :param df: test observations
    :return:
    """
    nm = 'preds'+str(cv_fold)
    df[nm] = mod.predict(df[features])
    df[nm] = df[nm].apply(lambda x: math.exp(x)-1)
    if is_test == 1:
        df['cost'] += df[nm]/num_loops
    return df

def write_xgb_preds(df, xgb_data, mod, cv_fold, is_test=1):
    """
    This writes predictions froma  model into the test data
    :param df: test observations
    :return:
    """
    nm = 'preds'+str(cv_fold)
    df[nm] = mod.predict(xgb_data)
    df[nm] = df[nm].apply(lambda x: math.exp(x)-1)
    if is_test == 1:
        df['cost'] += df[nm]/num_loops
    return df

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

# good parameter:
# gradient boosting: alpha=.16, estimators=150, depth=5
# Random forest 1000 trees
# SVR: rbf, small epsilons (.001) seem to be the best, but not that good

### Set parameters
avg_score = 0
num_loops = 6
start_num = 18
is_xgb = 1
is_forest = 0
is_boost = 0
test['cost'] = 0
param = {'max_depth': 6, 'eta': .095, 'silent': 1}

### Run models
for cv_fold in range(start_num, start_num+num_loops):
    # Create trn val samples
    trn, val = create_val_and_train(non_test, cv_fold, 'tube_assembly_id', .2)
    # recode target variable to log(x+1) in trn and val
    for df in [trn, val]:
        df['target'] = df['cost'].apply(lambda x: math.log(x+1))
    # Gradient boosting
    if is_xgb:
        xgb_trn = xgb.DMatrix(np.array(trn[feats]),
                              label=np.array(trn['target']))
        xgb_val = xgb.DMatrix(np.array(val[feats]),
                              label=np.array(val['target']))
        xgb_test = xgb.DMatrix(np.array(test[feats]))
        xboost = xgb.train(param.items(), xgb_trn, 1500)
        # Predict and rescale predictions
        val = write_xgb_preds(val, xgb_val, xboost, cv_fold, is_test=0)
        # Predict onto test
        test = write_xgb_preds(test, xgb_test, xboost, cv_fold, is_test=1)
    if is_forest:
        # Use lasso to select variables
        lassoed_vars = lasso_var_select(trn, feats)
        mod = RandomForestRegressor(n_estimators=50, n_jobs=8)
        mod.fit(trn[lassoed_vars], trn['target'])
        for i in range(0, len(mod.feature_importances_)):
            print "Feature %s has importance: %s" % (lassoed_vars[i],
                                                mod.feature_importances_[i])
    if is_boost:
        mod = GradientBoostingRegressor(learning_rate=.02, n_estimators=150,
                                         max_depth=1)
        mod.fit(trn[lassoed_vars], trn['target'])
    if is_boost | is_forest:
        val = write_preds(val, mod, cv_fold, lassoed_vars, is_test=0)
    score = rmsle(val['cost'], val['preds'+str(cv_fold)])
    print score

# Export test preds
test['id'] = test['id'].apply(lambda x: int(x))
test[['id', 'cost']].to_csv(SUBM_PATH+'stacking first attempt.csv', index=False)

num_loops = 6
start_num = 12
param = {'max_depth': 6, 'eta': .05, 'silent': 1}
for eta in [ .085, .09, .095, .10, .105]:
    avg_score = 0
    param['eta'] = eta
    for cv_fold in range(start_num, start_num+num_loops):
        # Create trn val samples
        trn, val = create_val_and_train(non_test, cv_fold, 'tube_assembly_id', .2)
        # recode target variable to log(x+1)
        trn['target'] = trn.cost.apply(lambda x: math.log(x+1))
        val['target'] = val.cost.apply(lambda x: math.log(x+1))
        # Gradient boosting
        xgb_trn = xgb.DMatrix(np.array(trn[feats]), label=np.array(trn['target']))
        xgb_val = xgb.DMatrix(np.array(val[feats]), label=np.array(val['target']))
        xgb_test = xgb.DMatrix(np.array(test[feats]))
        xboost = xgb.train(param.items(), xgb_trn, 1500)
        # Predict and rescale predictions
        val = write_xgb_preds(val, xgb_val, xboost, cv_fold, is_test=0)
        score = rmsle(val['cost'], val['preds'+str(cv_fold)])
        avg_score += score/num_loops
    print "the score for eta=%s is %s" % (eta, avg_score)

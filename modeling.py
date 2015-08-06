#__author__ = 'p_cohen'

############## Import packages ########################
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
import numpy as np
import math
import sys
sys.path.append('/home/vagrant/xgboost/wrapper')
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

############### Define Globals ########################
CLN_PATH = '/home/vagrant/caterpillar-peter/Clean/'
SUBM_PATH = '/home/vagrant/caterpillar-peter/Submissions/'

############### Define Functions ########################
def create_val_and_train(df, seed, ids, split_rt=.20):
    """
        Creates two samples (generally used to create
        train and validation samples)

        Parameters
        ----------------------------------------------------
        ids: this is the level of randomization, so if you want to
        randomize countries, rather than cities, you would
        set this to 'countries'

        split_rate: pct of data to assign as validation

        Output
        ----------------------------------------------------
        trn_for_mods (1-split_rate of df), trn_for_val (split_rate of data)


    """
    np.random.seed(seed)
    # Get vector of de-dupped values of ids
    id_dat = pd.DataFrame(df[ids].drop_duplicates())
    # Create random vector to split train val on
    vect_len = len(id_dat.ix[:, 0])
    id_dat['rand_vals'] = (np.array(np.random.rand(vect_len,1)))
    df = pd.merge(df, id_dat, on=ids)
    # split data into two dfs
    trn_for_mods = df[df.rand_vals > split_rt]
    trn_for_val = df[df.rand_vals <= split_rt]
    # drop rand_vals
    trn_for_val = trn_for_val.drop('rand_vals', axis=1)
    trn_for_mods = trn_for_mods.drop('rand_vals', axis=1)
    return trn_for_mods, trn_for_val

def rmsle(actual, predicted):
    """
    original author = 'benhamner'
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

def write_preds(df, mod, name, features):
    """
    Writes predictions from a model into a dataframe, transforms them
    according to e^x - 1

    Parameters
    -----------
    df: data to build predictions into and from
    mod: a predictive model
    name: name of predictions
    features: features used in model

    Output
    ----------
    dataframe with predictions labeled as 'preds'+name
    """
    nm = 'preds'+str(name)
    df[nm] = mod.predict(df[features])
    df[nm] = df[nm].apply(lambda x: math.exp(x)-1)
    return df

def write_xgb_preds(df, xgb_data, mod, pred_nm, is_test=0):
    """
    This writes predictions from an XGBOOST model into the data

    Parameters
    --------------
    df: pandas dataframe to predict into
    xgb_data: XGB dataframe (built from same data as df,
             with features used by mod)
    mod: XGB model used for predictions
    pred_nm: prediction naming convention

    Output
    --------------
    data frame with predictions

    """
    # Create name for predictions column
    nm = 'preds'+str(pred_nm)
    # Predict and rescale (rescales to e^pred - 1)
    df[nm] = mod.predict(xgb_data)
    #df[nm] = df[nm].apply(lambda x: math.exp(x)-1)
    df[nm] = np.power(df[nm], 16)
    # Create an average prediction across folds for actual submission
    if is_test == 1:
        df['cost'] += df[nm]/num_loops
    return df

############### Load data ######################
# Load data
all_data = pd.read_csv(CLN_PATH + "full_data.csv")
non_test = all_data[all_data.is_test == 0]
unique_tube = 1

test = all_data[all_data.is_test != 0]

# Create list of features
feats = list(all_data.columns.values)
non_feats = ['id', 'is_test', 'tube_assembly_id', 'cost']
for var in non_feats:
    feats.remove(var)

############ Run unrebalanced xgb ################
# Set parameters
avg_score = 0
num_loops = 6
start_num = 12
test['cost'] = 0
param = {'max_depth': 6, 'eta': .04, 'silent': 1, 'subsample': .8}
# Run models (looping through different train/val splits)
for cv_fold in range(start_num, start_num+num_loops):
    # Create trn val samples
    trn, val = create_val_and_train(non_test, cv_fold, 'tube_assembly_id', .2)
    trn['target'] = np.power(trn['cost'], .0625)
    trn['one'] = 1
    grouped = trn.groupby('tube_assembly_id')
    counts = grouped.one.sum().reset_index()
    counts = counts.rename(columns={'one': 'ob_weight'})
    trn = trn.merge(counts, on='tube_assembly_id')
    # Gradient boosting
    xgb_trn = xgb.DMatrix(np.array(trn[feats]),
                          label=np.array(trn['target']),
                          weight=np.array(trn.ob_weight))
    xgb_val = xgb.DMatrix(np.array(val[feats]))
    xgb_test = xgb.DMatrix(np.array(test[feats]))
    xboost = xgb.train(param.items(), xgb_trn, 2000)
    # Predict and rescale predictions
    val = write_xgb_preds(val, xgb_val, xboost, str(cv_fold), is_test=0)
    test = write_xgb_preds(test, xgb_test, xboost, str(cv_fold), is_test=1)
    # Save score
    score = rmsle(val['cost'], val['preds'+str(cv_fold)])
    avg_score += score/num_loops
    print score
avg_score

# Export test preds
test['id'] = test['id'].apply(lambda x: int(x))
test[['id', 'cost']].to_csv(SUBM_PATH+'rebalanced max.csv', index=False)

############ Run rebalanced xgb ################

# Set parameters

num_loops = 6
start_num = 12
test['cost'] = 0
for eta in [.004, .008, .012]:
    avg_score = 0
    param = {'max_depth': 8, 'eta': eta, 'silent': 1, 'subsample': .75}
    # Run models (looping through different train/val splits)
    for cv_fold in range(start_num, start_num+num_loops):
        # Create trn val samples
        trn, val = create_val_and_train(non_test, cv_fold, 'tube_assembly_id', .2)
        val['preds'+str(cv_fold)] = 0
        test['preds'+str(cv_fold)] = 0
        # recode target variable to log(x+1) in trn and val
        for folds in range(1, 4):
            trn['rand'] = (np.array(np.random.rand(len(trn.quantity),1)))
            grouped = trn.groupby('tube_assembly_id')
            tube_maxquant = grouped.rand.max().reset_index()
            mrg_vars = ['tube_assembly_id', 'rand']
            trn1 = trn.merge(tube_maxquant, how='inner', on=mrg_vars)
            trn1['target'] = np.power(trn1['cost'], .0625)
            # Gradient boosting
            xgb_trn = xgb.DMatrix(np.array(trn1[feats]), label=np.array(trn1['target']))
            xgb_val = xgb.DMatrix(np.array(val[feats]))
            xgb_test = xgb.DMatrix(np.array(test[feats]))
            xboost = xgb.train(param.items(), xgb_trn, 5000)
            # Predict and rescale predictions
            loop_nm = str(cv_fold)+'_'+str(folds)
            val = write_xgb_preds(val, xgb_val, xboost, loop_nm, is_test=0)
            test = write_xgb_preds(test, xgb_test, xboost, loop_nm, is_test=0)
            # Combine predictions
            nm = 'preds'+str(cv_fold)
            val[nm] += val[nm+'_'+str(folds)]/3
            test[nm] += test[nm+'_'+str(folds)]/3
            #print "mini loop is %s" % (rmsle(val['cost'], val[nm+'_'+str(folds)]))
        # Save score
        score = rmsle(val['cost'], val['preds'+str(cv_fold)])
        avg_score += score/num_loops
        #print score
    print avg_score

# Export test preds
test['cost'] = test[['preds12', 'preds13', 'preds14',
                     'preds15', 'preds16', 'preds17']].mean(axis=1)
test['id'] = test['id'].apply(lambda x: int(x))
test[['id', 'cost']].to_csv(SUBM_PATH+'rebalanced max.csv', index=False)

############ Browse feature importances ################

# Code for browsing feature importances
for cv_fold in range(1, 2):
    # Create trn val samples
    trn, val = create_val_and_train(non_test, cv_fold, 'tube_assembly_id', .2)
    # recode target variable to log(x+1)
    trn['target'] = trn.cost.apply(lambda x: math.log(x+1))
    val['target'] = val.cost.apply(lambda x: math.log(x+1))
    # Gradient boosting
    frst = RandomForestRegressor(n_estimators=100, n_jobs=8)
    frst.fit(trn[feats], trn['target'])
    outputs = pd.DataFrame({'feats': feats,
                           'weight': frst.feature_importances_})
    print outputs.sort(columns='weight', ascending=False)

__author__ = 'p_cohen'

############## Import packages ########################
# Workaround for pycharm bug
from __builtin__ import list, range, len, str, set, any, int
import pandas as pd
import numpy as np
import math
import sys
sys.path.append('/home/vagrant/xgboost/wrapper')
import xgboost as xgb

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import Lasso, RidgeCV
from sklearn.tree import DecisionTreeRegressor
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
    id_dat['rand_vals'] = (np.array(np.random.rand(vect_len, 1)))
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
    df[nm] = np.power(df[nm], 16)
    return df


def write_xgb_preds(df, xgb_data, mod, pred_nm, scale, is_test=0):
    """
    This writes predictions from an XGBOOST model into the data

    Parameters
    --------------
    df: pandas dataframe to predict into
    xgb_data: XGB dataframe (built from same data as df,
             with features used by mod)
    mod: XGB model used for predictions
    pred_nm: prediction naming convention
    scale: (float) this is the power to raise predictions to pred^scale

    Output
    --------------
    data frame with predictions

    """
    # Create name for predictions column
    nm = 'preds'+str(pred_nm)
    # Predict and rescale (rescales to e^pred - 1)
    df[nm] = mod.predict(xgb_data)
    # df[nm] = df[nm].apply(lambda x: math.exp(x)-1)
    df[nm] = np.power(df[nm], scale)
    # Create an average prediction across folds for actual submission
    if is_test == 1:
        df['cost'] += df[nm]/num_loops
    return df


def gen_weights(df):
    """
    This creates weights based on the number of rows per tube assembly

    :param df: data frame to add weights to
    :return: dataframe with wieghts
    """
    df['one'] = 1
    grouped = df.groupby('tube_assembly_id')
    counts = grouped.one.sum().reset_index()
    counts = counts.rename(columns={'one': 'ob_weight'})
    counts['ob_weight'] = counts['ob_weight']
    df = df.merge(counts, on='tube_assembly_id')
    df = df.drop('one', axis=1)
    return df

def outcome_transfactor(int):
    """
    This creates slightly different functional forms for outcome

    :param int: integer used to control functional form (think of it as a
    random number
    :return: transformation factor (power)
    """
    power_up, power_down = 16, 1.0/16
    if int % 2 == 0:
        power_up, power_down = 17, 1.0/17
    if int % 3 == 0:
        power_up, power_down = 15, 1.0/15
    return power_up, power_down

def create_firststage_preds(train, valid, testing):
    """
    This handles the first stage of a true stacking procedure using
    random forests to create first stage predictions in the train, test,
    and validation. Splits train into two sections, run random forest
    on both and predicts from one half into other (and visa versa). Then
    random forest is run on whole model and predicted into both validation
    and test.
    """
    np.random.seed(42)
    # Get vector of de-dupped values of ids
    id_dat = pd.DataFrame(train['tube_assembly_id'].drop_duplicates())
    # Create random vector to split train val on
    vect_len = len(id_dat.ix[:, 0])
    id_dat['rand_vals'] = (np.array(np.random.rand(vect_len, 1)))
    df = pd.merge(train, id_dat, on='tube_assembly_id')
    # Create model for both halves of df
    frst1 = RandomForestRegressor(n_estimators=300, n_jobs=7)
    is_first_half = df.rand_vals > .5
    is_scnd_half = df.rand_vals < .5
    frst1.fit(df.ix[is_first_half, feats], df.ix[is_first_half, 'target'])
    frst2 = RandomForestRegressor(n_estimators=300, n_jobs=7)
    frst2.fit(df.ix[is_scnd_half, feats], df.ix[is_scnd_half, 'target'])
    # Predict frst1 onto forst2 data set and visa versa
    train['forest'] = 0
    train['forest'][is_scnd_half] = frst1.predict(df.ix[is_scnd_half, feats])
    train['forest'][is_first_half] = frst2.predict(df.ix[is_first_half, feats])
    # Create forest in full data for validation and test
    frst = RandomForestRegressor(n_estimators=300, n_jobs=7)
    frst.fit(df[feats], df.target)
    valid['forest'] = frst.predict(valid[feats])
    testing['forest'] = frst.predict(testing[feats])
    # Create model for both halves of df
    rdg1 = RidgeCV(alphas=[.5, .75, 1, 1.25])
    rdg2 = RidgeCV(alphas=[.5, .75, 1, 1.25])
    rdg1.fit(df.ix[is_first_half, feats], df.ix[is_first_half, 'target'])
    rdg2.fit(df.ix[is_scnd_half, feats], df.ix[is_scnd_half, 'target'])
    # Predict frst1 onto forst2 data set and visa versa
    train['ridge'] = 0
    train['ridge'][is_scnd_half] = rdg1.predict(df.ix[is_scnd_half, feats])
    train['ridge'][is_first_half] = rdg2.predict(df.ix[is_first_half, feats])
    # Create forest in full data for validation and test
    rdg = RidgeCV(alphas=[.5, .75, 1, 1.25])
    rdg.fit(df[feats], df.target)
    valid['ridge'] = rdg.predict(valid[feats])
    testing['ridge'] = rdg.predict(testing[feats])


############### Load data ######################
# Load data
all_data = pd.read_csv(CLN_PATH + "full_data.csv")
non_test = all_data[all_data.is_test == 0]
test = all_data[all_data.is_test != 0]
# Create list of features
feats = list(all_data.columns.values)
non_feats = ['id', 'is_test', 'tube_assembly_id', 'cost']
for var in non_feats:
    feats.remove(var)

# ########### Run unrebalanced xgb ################
# Set parameters
avg_score = 0
num_loops = 15
start_num = 42
loop = 1
current_sum = 0.0
test['cost'] = 0
param = {'max_depth': 8, 'eta': .027,  'silent': 1, 'subsample': .75,
         'colsample_bytree': .75, 'gamma': .00025}
# Run models (looping through different train/val splits)
for cv_fold in range(start_num, start_num+num_loops):
    # Create trn val samples
    trn, val = create_val_and_train(non_test, cv_fold, 'tube_assembly_id', .2)
    power_up, power_down = outcome_transfactor(cv_fold)
    trn['target'] = np.power(trn['cost'], power_down)
    trn = gen_weights(trn)
    # Gradient boosting
    xgb_trn = xgb.DMatrix(np.array(trn[feats]), label=np.array(trn['target']),
                          weight=np.array(trn.ob_weight))
    xgb_val = xgb.DMatrix(np.array(val[feats]))
    xgb_test = xgb.DMatrix(np.array(test[feats]))
    xboost = xgb.train(param.items(), xgb_trn, 2500)
    # Predict and rescale predictions
    cv_str = str(cv_fold)
    val = write_xgb_preds(val, xgb_val, xboost, cv_str, power_up, is_test=0)
    test = write_xgb_preds(test, xgb_test, xboost, cv_str, power_up, is_test=1)
    # Save score
    score = rmsle(val['cost'], val['preds'+cv_str])
    avg_score += score/num_loops
    current_sum += score
    print "Loop %s score is : %s" % (loop, score)
    print "Current average score is %s" % (current_sum/loop)
    loop += 1

############ TEST of a simple true stacking concept ######
# Set parameters
num_loops = 4
start_num = 42
loop = 1
current_sum = 0.0
test['cost'] = 0
param = {'max_depth': 8, 'eta': .022,  'silent': 1, 'subsample': .65,
         'colsample_bytree': .55, 'gamma': .00025}
for eta in [.021, .022, .023, .024, .025, .0255]:
    avg_score = 0
    print "eta is %s" % eta
    param['eta'] = eta
    # Run models (looping through different train/val splits)
    for cv_fold in range(start_num, start_num+num_loops):
        # Create list of features
        feats = list(all_data.columns.values)
        non_feats = ['id', 'is_test', 'tube_assembly_id', 'cost']
        for var in non_feats:
            feats.remove(var)
        # Create trn val samples
        trn, val = create_val_and_train(non_test, cv_fold, 'tube_assembly_id', .2)
        # Create functional form of outcome
        power_up, power_down = outcome_transfactor(cv_fold)
        trn['target'] = np.power(trn['cost'], power_down)
        trn = gen_weights(trn)
        # Create first stage predictions
        create_firststage_preds(trn, val, test)
        feats.append('ridge')
        feats.append('forest')
        # Gradient boosting
        xgb_trn = xgb.DMatrix(np.array(trn[feats]), label=np.array(trn['target']),
                              weight=np.array(trn.ob_weight))
        xgb_val = xgb.DMatrix(np.array(val[feats]))
        xgb_test = xgb.DMatrix(np.array(test[feats]))
        xboost = xgb.train(param.items(), xgb_trn, 2500)
        # Predict and rescale predictions
        cv_str = str(cv_fold)
        val = write_xgb_preds(val, xgb_val, xboost, cv_str, power_up, is_test=0)
        test = write_xgb_preds(test, xgb_test, xboost, cv_str, power_up, is_test=1)
        # Save score
        score = rmsle(val['cost'], val['preds'+cv_str])
        avg_score += score/num_loops
        current_sum += score
        print "Loop %s score is : %s" % (loop, score)
        print "Current average score is %s" % (current_sum/loop)
        loop += 1

print avg_score

############################################################################

# Export test preds
test['id'] = test['id'].apply(lambda x: int(x))
test[['id', 'cost']].to_csv(SUBM_PATH+'2500 trees with 15 folds and minor gamma.csv', index=False)

# ########### Browse feature importances ################
# Code for browsing feature importances
cv_fold = 12
# Create trn val samples
trn, val = create_val_and_train(non_test, cv_fold, 'tube_assembly_id', .2)
# recode target variable to log(x+1)
trn['target'] = np.power(trn['cost'], .0625)
val['target'] = np.power(trn['cost'], .0625)
# Gradient boosting
frst = RandomForestRegressor(n_estimators=300, n_jobs=4)
frst.fit(trn[feats], trn['target'])
outputs = pd.DataFrame({'feats': feats,
                        'weight': frst.feature_importances_})
outputs = outputs.sort(columns='weight', ascending=False)
val = write_preds(val, frst, cv_fold, feats)
# Score loop
score = rmsle(val['cost'], val['preds'+str(cv_fold)])
print "Score for %s trees is: %s" % (12, score)
print outputs

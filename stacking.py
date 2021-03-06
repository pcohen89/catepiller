__author__ = 'p_cohen'
from __builtin__ import list, range, len, str, set, any, int, min

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge, RidgeCV
import numpy as np
import sys
sys.path.append('/home/vagrant/xgboost/wrapper')
import xgboost as xgb

############### Define Globals ########################
DATA_PATH = 'C:/Git_repos/catepiller/Original/'
CLN_PATH = 'C:/Git_repos/catepiller/Clean/'
SUBM_PATH = 'C:/Git_repos/catepiller/Submissions/'
############### Define Functions ########################
def create_val_and_train(df, seed, ids, split_rt = .20):
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
    #### __author__ = 'benhamner'
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
    df[nm] = np.power(df[nm], 16)
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

def create_feat_list(df, non_features):
    feats = list(df.columns.values)
    for var in non_features:
        feats.remove(var)
    return feats


######################################################
# Load data
all_data = pd.read_csv(CLN_PATH + "full_data.csv")
non_test = all_data[all_data.is_test == 0]
test = all_data[all_data.is_test != 0]

# Set hyper parameters of process
types = ['boss', 'adaptor', 'elbow', 'float', 'hfl', 'nut', 'other', 'sleeve',
         'straight', 'threaded']

################
# Globals
################
non_feats = ['id', 'is_test', 'tube_assembly_id', 'cost']

all_feats = all_data.columns.values
avg_score = 0
first_loop = 0
num_loops = 15
start_num = 42
# Run bagged models
for cv_fold in range(start_num, start_num+num_loops):
    param = {'max_depth': 6, 'eta': .135, 'silent': 1, 'subsample': .7,
             'colsample_bytree': .75}
    # Create trn val samples
    trn, val = create_val_and_train(non_test, cv_fold, 'tube_assembly_id', .2)
    # recode target variable to log(x+1) in trn and val
    trn['target'] = np.power(trn['cost'], .0625)
    val['target'] = np.power(val['cost'], .0625)
    # Separate samples for first stage and second stage
    feat_trn, mod_trn = create_val_and_train(trn, cv_fold, 'tube_assembly_id', .15)
    feat_trn = gen_weights(feat_trn)
    # Create list of second stage modeling features
    stage2_feats = create_feat_list(all_data, non_feats)
    # Gradient boosting (choose TWO comp types a generate models on using those
    # and core features)
    for i in range(0, len(types)):
        for j in range(i+1, len(types)):
            for k in range(j+1, len(types)):
                # Initialize the base features for first stage
                stg1_feats = ['annual_usage', 'bracket_pricing', 'tube_wall',
                              'min_order_quantity', 'quote_date',
                              'supplier', 'tube_length', 'adjusted_unique_cnt',
                              'tube_material_id', 'tube_diameter',
                              'tube_num_bends', 'tube_bend_radius',
                              'tube_end_a_1x', 'tube_end_a_2x', 'quantity',
                              'tube_end_x_1x', 'tube_end_x_2x', 'tube_end_a',
                              'tube_end_x', 'tube_num_boss', 'supplier_freq',
                              'tube_num_bracket', 'tube_other','dayofyear',
                              'comp_weight_sum', 'first_year_appeared_cnt',
                              'comp_tot_cnt', 'specs_cnt', 'adjusted_wt',
                              'is_min_order_quantity', 'ext_as_pct',
                              'len_x_dai', 'dia_over_len', 'wall_over_diam',
                              'unq_cnt', 'thick_cnt', 'orient_cnt', 'month',
                              'bend_per_length', 'radius_per_bend', 'year',
                              'ann_use_ove_q', 'length_x_wall', 'dayofweek'
                              ]
                # Add all feats that match either component type
                # Note: see data building for how shorten this if statement
                for feat in all_feats:
                    if ((types[j] in feat) | (types[i] in feat) |
                            (types[k] in feat)):
                        stg1_feats.append(feat)
                # Create xgboost data sets
                xgb_feat_trn = xgb.DMatrix(np.array(feat_trn[stg1_feats]),
                                           label=np.array(feat_trn.target),
                                           weight=np.array(feat_trn.ob_weight))
                xgb_mod_trn = xgb.DMatrix(np.array(mod_trn[stg1_feats]),
                                          label=np.array(mod_trn['target']))
                xgb_val = xgb.DMatrix(np.array(val[stg1_feats]))
                xgb_test = xgb.DMatrix(np.array(test[stg1_feats]))
                # Fit xgboost
                xboost = xgb.train(param.items(), xgb_feat_trn, 1000)
                # Create scaled predictions
                nm1 = 'frststage' + types[i] + types[j] + types[k]
                val = write_xgb_preds(val, xgb_val, xboost, nm1)
                mod_trn = write_xgb_preds(mod_trn, xgb_mod_trn, xboost, nm1)
                test = write_xgb_preds(test, xgb_test, xboost, nm1)
                # Add prediction to stage 2 features
                stage2_feats.append('preds'+nm1)
                score = rmsle(val['cost'], val['preds'+nm1])
                # Create ridge feats
                model = Ridge(alpha=3)
                model = model.fit(feat_trn[stg1_feats], feat_trn['target'])
                # Predict and rescale predictions
                nm2 = 'frststage_rdg' + types[i] + types[j] + types[k]
                val = write_preds(val, model, nm2, stg1_feats)
                mod_trn = write_preds(mod_trn, model, nm2, stg1_feats)
                test = write_preds(test, model, nm2, stg1_feats)
                # Store prediction variable name
                stage2_feats.append('preds'+nm2)
                score_rdg = rmsle(val['cost'], val['preds'+nm2])
                # Report score of loop
                label1 = "For the %s - %s - %s fold, score "
                label2 = "is %s for boost and %s for forest"
                label = label1 + label2
                print label % (types[i], types[j], types[k], score,
                               score_rdg)
    # Fit second stage model
    model = RandomForestRegressor(n_estimators=2000, n_jobs=8)
    model.fit(mod_trn[stage2_feats], mod_trn.target.values)
    val = write_preds(val, model, cv_fold, stage2_feats)
    test = write_preds(test, model, cv_fold, stage2_feats)
    # Score loop
    score = rmsle(val['cost'], val['preds'+str(cv_fold)])
    print "Score for fold %s is: %s" % (cv_fold, score)
    avg_score += score/num_loops
print avg_score


test['cost'] = test[[u'preds42', u'preds43', u'preds44', u'preds45', u'preds46']].mean(axis=1)


# Export test preds
test['id'] = test['id'].apply(lambda x: int(x))
test[['id', 'cost']].to_csv(SUBM_PATH+'stack w col samp 5 folds new var.csv', index=False)


######################################################
# New Stacking Concept
#####################################################
all_data = pd.read_csv(CLN_PATH + "full_data.csv")
non_test = all_data[all_data.is_test == 0]
test = all_data[all_data.is_test != 0]

# Set hyper parameters of process
avg_score = 0
first_loop = 0
num_loops = 1
start_num = 42
# Create list of second stage modeling features
stage1_feats = create_feat_list(all_data, non_feats)
stage2_feats = create_feat_list(all_data, non_feats)
for frststage_mod in ['rdg', 'xgb', 'frst']:
    stage2_feats.append('preds_' + frststage_mod)
# Run bagged models
for cv_fold in range(start_num, start_num+num_loops):
    param = {'max_depth': 6, 'eta': .135, 'silent': 1, 'subsample': .7,
             'colsample_bytree': .5}
    # Create trn val samples
    trn, val = create_val_and_train(non_test, cv_fold, 'tube_assembly_id', .2)
    # recode target variable to log(x+1) in trn and val
    trn['target'] = np.power(trn.cost, .0625)
    val['target'] = np.power(val.cost, .0625)
    # Separate samples for first stage and second stage
    feat_trn, mod_trn = create_val_and_train(trn, cv_fold, 'tube_assembly_id', .25)
    feat_trn = gen_weights(feat_trn)
    # Gradient boosting
    # Create xgboost data sets
    xgb_feat_trn = xgb.DMatrix(np.array(feat_trn[stg1_feats]),
                               label=np.array(feat_trn.target),
                               weight=np.array(feat_trn.ob_weight))
    xgb_mod_trn = xgb.DMatrix(np.array(mod_trn[stg1_feats]),
                              label=np.array(mod_trn.target))
    xgb_val = xgb.DMatrix(np.array(val[stg1_feats]))
    xgb_test = xgb.DMatrix(np.array(test[stg1_feats]))
    # Fit xgboost
    xboost = xgb.train(param.items(), xgb_feat_trn, 100)
    # Create scaled predictions
    val = write_xgb_preds(val, xgb_val, xboost, '_xgb')
    mod_trn = write_xgb_preds(mod_trn, xgb_mod_trn, xboost, '_xgb')
    test = write_xgb_preds(test, xgb_test, xboost, '_xgb')
    # Create ridge feats
    alphas = [.5, 1.5, 3, 4.5]
    model = RidgeCV(alphas=alphas).fit(feat_trn[stg1_feats], feat_trn.target)
    # Predict and rescale predictions
    val = write_preds(val, model, '_rdg', stg1_feats)
    mod_trn = write_preds(mod_trn, model, '_rdg', stg1_feats)
    test = write_preds(test, model, '_rdg', stg1_feats)
    # Create forest feats
    frst = RandomForestRegressor(n_estimators=200, n_jobs=8)
    frst.fit(feat_trn[stg1_feats], feat_trn.target)
    # Predict and rescale predictions
    val = write_preds(val, frst, '_frst', stg1_feats)
    mod_trn = write_preds(mod_trn, frst, '_frst', stg1_feats)
    test = write_preds(test, frst, '_frst', stg1_feats)
    # Score models
    score_xgb = rmsle(val.cost, val.preds_xgb)
    score_rdg = rmsle(val.cost, val.preds_rdg)
    score_frst = rmsle(val.cost, val.preds_frst)
    # Report score of loop
    label1 = "XGB score is %s;"
    label2 = "Ridge score is %s;"
    label3 = "Forest score is %s;"
    label = label1 + label2 + label3
    print label % (score_xgb, score_rdg, score_frst)
    # Fit second stage model
    model = RandomForestRegressor(n_estimators=200, n_jobs=8)
    model.fit(mod_trn[stage2_feats], mod_trn.target.values)
    val = write_preds(val, model, cv_fold, stage2_feats)
    test = write_preds(test, model, cv_fold, stage2_feats)
    # Score loop
    score = rmsle(val['cost'], val['preds'+str(cv_fold)])
    print "Score for fold %s is: %s" % (cv_fold, score)
    avg_score += score/num_loops
    outputs = pd.DataFrame({'feats': stage2_feats,
                        'weight': model.feature_importances_})
    outputs = outputs.sort(columns='weight', ascending=False)
    print outputs
print avg_score


test['cost'] = test[[u'preds42', u'preds43', u'preds44', u'preds45', u'preds46']].mean(axis=1)


# Export test preds
test['id'] = test['id'].apply(lambda x: int(x))
test[['id', 'cost']].to_csv(SUBM_PATH+'stack w col samp 5 folds new var.csv', index=False)
#'threeway vars with bill vars.csv'

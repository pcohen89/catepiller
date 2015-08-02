__author__ = 'p_cohen'

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
import numpy as np
import math
import sys
sys.path.append('/home/vagrant/xgboost/wrapper')
import xgboost as xgb

############### Define Globals ########################
CLN_PATH = '/home/vagrant/caterpillar-peter/Clean/'
SUBM_PATH = '/home/vagrant/caterpillar-peter/Submissions/'
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
    df[nm] = df[nm].apply(lambda x: math.exp(x)-1)
    # Create an average prediction across folds for actual submission
    if is_test == 1:
        df['cost'] += df[nm]/num_loops
    return df

######################################################
### Load data ####
all_data = pd.read_csv(CLN_PATH + "full_data.csv")
non_test = all_data[all_data.is_test == 0]
test = all_data[all_data.is_test != 0]

### Set hyper parameters of process
types = ['boss', 'adaptor', 'elbow', 'float', 'hfl', 'nut', 'other', 'sleeve',
         'straight', 'threaded']
all_feats = all_data.columns.values
avg_score = 0
first_loop = 0
num_loops = 6
start_num = 12
# Run (sort of) cross validated models
for cv_fold in range(start_num, start_num+num_loops):
    param = {'max_depth':6, 'eta': .14, 'silent': 1, 'subsample': .7}
    # Create trn val samples
    trn, val = create_val_and_train(non_test, cv_fold, 'tube_assembly_id', .2)
    # recode target variable to log(x+1) in trn and val
    for df in [trn, val]:
        df['target'] = df['cost'].apply(lambda x: math.log(x+1))
    # Separate samples for first stage and second stage
    feat_trn, mod_trn = create_val_and_train(trn, cv_fold, 'tube_assembly_id', .15)
    # Create list of second stage modeling features
    stage2_feats = list(all_data.columns.values)
    non_feats = ['id', 'is_test', 'tube_assembly_id', 'cost']
    for var in non_feats:
        stage2_feats.remove(var)
    # Gradient boosting (choose TWO comp types a generate models on using those
    # and core features)
    for i in range(0, len(types)):
        for j in range(i+1, len(types)):
            for k in range(j+1, len(types)):
                # Initialize the base features for first stage
                stage1_feats = ['annual_usage', 'bracket_pricing',
                                'min_order_quantity', 'quantity', 'quote_date',
                                'supplier', 'tube_wall', 'tube_length',
                                'tube_material_id', 'tube_diameter',
                                'tube_num_bends', 'tube_bend_radius',
                                'tube_end_a_1x', 'tube_end_a_2x',
                                'tube_end_x_1x', 'tube_end_x_2x', 'tube_end_a',
                                'tube_end_x', 'tube_num_boss',
                                'tube_num_bracket', 'tube_other',
                                'year', 'month', 'dayofyear', 'comp_weight_sum',
                                'comp_tot_cnt', 'specs_cnt',
                                'is_min_order_quantity', 'ext_as_pct'
                ]
                # Add all feats that match either component type
                for feat in all_feats:
                    if ((types[j] in feat) | (types[i] in feat) | (types[k] in feat)):
                        stage1_feats.append(feat)
                if first_loop == 0:
                    print stage1_feats
                    first_loop = 1
                # Create xgboost data sets
                xgb_feat_trn = xgb.DMatrix(np.array(feat_trn[stage1_feats]),
                                      label=np.array(feat_trn['target']))
                xgb_mod_trn = xgb.DMatrix(np.array(mod_trn[stage1_feats]),
                                      label=np.array(mod_trn['target']))
                xgb_val = xgb.DMatrix(np.array(val[stage1_feats]))
                xgb_test = xgb.DMatrix(np.array(test[stage1_feats]))
                # Fit xgboost
                xboost = xgb.train(param.items(), xgb_feat_trn, 1000)
                # Create scaled predictions
                nm = 'frststage' + types[i] + types[j]  + types[k]
                val = write_xgb_preds(val, xgb_val, xboost, nm)
                mod_trn = write_xgb_preds(mod_trn, xgb_mod_trn, xboost, nm)
                test = write_xgb_preds(test, xgb_test, xboost, nm)
                # Add prediction to stage 2 features
                stage2_feats.append('preds'+nm)
                score = rmsle(val['cost'], val['preds'+nm])
                # Create ridge feats
                model = Ridge(alpha=3)
                model = model.fit(feat_trn[stage1_feats], feat_trn['target'])
                # Predict and rescale predictions
                nm = 'frststage_rdg' + types[i] + types[j] + types[k]
                val = write_preds(val, model, nm, stage1_feats)
                mod_trn = write_preds(mod_trn, model, nm, stage1_feats)
                test = write_preds(test, model, nm, stage1_feats)
                # Store prediction variable name
                stage2_feats.append('preds'+nm)
                score_rdg = rmsle(val['cost'], val['preds'+nm])
                # Report score of loop
                label1 = "For the %s - %s - %s fold, score"
                label2 = "is %s for boost and %s for forest"
                print label1 + label2 % (types[i], types[j],
                                         types[k], score, score_rdg)
    # Fit second stage model
    model = RandomForestRegressor(n_estimators=2000, n_jobs=8)
    model.fit(mod_trn[stage2_feats], mod_trn.target.values)
    val = write_preds(val, model, cv_fold, stage2_feats)
    test = write_preds(test, model, cv_fold, stage2_feats)
    # Score loop
    score = rmsle(val['cost'], val['preds'+str(cv_fold)])
    print "Score for fold %s is: %s" % (str(param['eta']), score)
    avg_score += score/num_loops
print avg_score


test['cost'] = test[['preds12', 'preds13', 'preds14', 'preds15', 'preds16', 'preds17']].mean(axis=1)
test[['preds12', 'preds13', 'preds14', 'preds15', 'preds16', 'preds17']].corr()

# Export test preds
test['id'] = test['id'].apply(lambda x: int(x))
test[['id', 'cost']].to_csv(SUBM_PATH+'threeway vars with subsamp.csv', index=False)

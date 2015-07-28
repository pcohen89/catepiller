__author__ = 'p_cohen'

import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from sklearn.preprocessing import StandardScaler
import numpy as np
import math

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

def write_nn_preds(df, mod, cv_fold, features, is_test=0):
    """
    This writes predictions from a model into a dataframe
    :param df: test observations
    :return:
    """
    vals = mod.predict(df)
    return vals

######################################################
### Load data ####
all_data = pd.read_csv(CLN_PATH + "full_data.csv")
non_test = all_data[all_data.is_test == 0]
test = all_data[all_data.is_test != 0]

### Set hyper parameters of process
types = ['boss', 'adaptor', 'elbow', 'float', 'hfl', 'nut', 'other', 'sleeve',
         'straight', 'threaded']
feats = list(non_test.columns.values)
non_feats = ['id', 'is_test', 'tube_assembly_id', 'cost']
for var in non_feats:
    feats.remove(var)
# Try to keep only continuous variables?
for var in feats:
    if ((len(all_data[var].drop_duplicates()) < 6) |
            ('id' in var) |
            ('max' in var) ):
        feats.remove(var)
len(feats)
feats = ['annual_usage', 'min_order_quantity', 'quantity', 'quote_date', 'supplier',
 'supplier_freq', 'tube_diameter', 'tube_wall', 'tube_length', 'tube_num_bends',
 'tube_bend_radius', 'tube_end_a_2x', 'tube_end_x_2x', 'tube_end_a', 'tube_end_x',
 'tube_num_boss', 'tube_other', 'tube_pc_tube_end_nomatch', 'bill_of_materials_quantity_1',
 'bill_of_materials_quantity_2', 'bill_of_materials_quantity_3', 'bill_of_materials_quantity_4',
 'bill_of_materials_quantity_5', 'reshaped_specs_spec1', 'reshaped_specs_spec2',
 'reshaped_specs_spec3', 'reshaped_specs_spec4', 'reshaped_specs_spec5',
 'reshaped_specs_spec6', 'reshaped_specs_spec7', 'reshaped_specs_spec9',
 'reshaped_specs_s_0004', 'reshaped_specs_s_0007', 'reshaped_specs_s_0057',
 'reshaped_specs_s_0070', 'reshaped_specs_s_0024', 'reshaped_specs_s_0009',
 'reshaped_specs_s_0029', 'reshaped_specs_s_0067','comp_tot_cnt',
 'specs_cnt', 'elbow_groove_max', 'other_weight_median', 'other_count',
 'straight_thickness_median', 'sleeve_weight_median', 'nut_weight_median',
 'ext_as_pct',
 'comp_weight_sum',
 'apprx_density',
 'length_x_wall',
 'radius_per_bend',
 'bend_per_length',
 'reshaped_specs_s_0047', 'reshaped_specs_s_0016',
 'reshaped_specs_s_0088', 'reshaped_specs_s_0065',
 'reshaped_specs_s_0082', 'reshaped_specs_s_0076',
 'tube_end_form_forming_x', 'tube_end_form_forming_y',  'year',
 'month',
 'dayofyear',]
avg_score = 0
first_loop = 0
num_loops = 1
start_num = 12
# Run (sort of) cross validated models
for cv_fold in range(start_num, start_num+num_loops):
    # Create trn val samples
    trn, val = create_val_and_train(non_test, cv_fold, 'tube_assembly_id', .2)
    # recode target variable to log(x+1) in trn and val
    for df in [trn, val]:
        df['target'] = df['cost'].apply(lambda x: math.log(x+1))
    # Separate samples for first stage and second stage
    model = Sequential()
    model.add(Dense(len(feats), 256))
    model.add(Activation('relu'))
    model.add(Dropout(.2))
    model.add(Dense(256, 256))
    model.add(Activation('relu'))
    model.add(Dropout(.2))
    model.add(Dense(256, 1))
    #sgd = SGD(lr=0.15, momentum=.02, nesterov=True)
    # Rescale data
    scaler = StandardScaler()
    scaler.fit(all_data[feats])
    X = scaler.transform(trn[feats])
    val_X = scaler.transform(val[feats])
    test_X = scaler.transform(test[feats])
    # Fit model
    model.compile(loss='mse', optimizer='rmsprop')
    model.fit(np.array(X), np.array(trn.target.values), batch_size=32,
              nb_epoch=45, verbose=2, validation_split=0.15)
    # Write predictions
    val['preds_final'] = model.predict(val_X)
    val['preds_final'] = val['preds_final'].apply(lambda x: math.exp(x)-1)
    test['cost'] = model.predict(test_X)
    test['cost'] = test['cost'].apply(lambda x: math.exp(x)-1)
    # Score loop
    score = rmsle(val['cost'], val['preds_final'])
    print "Score for loop %s is: %s" % (cv_fold, score)
    avg_score += score/num_loops
avg_score

prds = ['preds12', 'preds13', 'preds14', 'preds15', 'preds16', 'preds17']
test['cost'] = test['cost'].apply(lambda x: min(x, 1000))
test['cost'] = test['cost'].apply(lambda x: max(x, .5))
test[].corr()

# Export test preds
test['id'] = test['id'].apply(lambda x: int(x))
test[['id', 'cost']].to_csv(SUBM_PATH+'uhhh neural network.csv', index=False)

# Figure out what is breaking
for feat in feats:
    topic = "%s: min: %s, max: %s"
    print topic % (feat, (val[feat].min()-val[feat].mean())/ val[feat].std(), (val[feat].max()-val[feat].mean())/val[feat].std())
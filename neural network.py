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
        this will create a validate and train

        ids: this is the level of randomization, so if you want to
        randomize countries, rather than cities, you would
        set this to 'countries'

        split_rate: pct of data to assign as validation
    """
    np.random.seed(seed)
    # Get vector of de-dupped values of ids
    id_dat = pd.DataFrame(df[ids].drop_duplicates())
    # creating random vector to split train val on
    vect_len = len(id_dat.ix[:, 0])
    id_dat['rand_vals'] = (np.array(np.random.rand(vect_len,1)))
    df = pd.merge(df, id_dat, on=ids)
    # splits train into modeling and validating portions
    trn_for_mods = df[df.rand_vals > split_rt]
    trn_for_val = df[df.rand_vals <= split_rt]
    # drop rand_vals
    for sub_df in [trn_for_mods, trn_for_val]:
        sub_df = sub_df.drop('rand_vals', axis=1)
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
    #model.add(Dropout(.2))
    model.add(Dense(len(feats), 256))
    model.add(Activation('relu'))
    model.add(Dropout(.2))
    model.add(Dense(256, 256))
    model.add(Activation('relu'))
    model.add(Dropout(.2))
    model.add(Dense(256, 1))
    # Gradient boosting (choose TWO comp types a generate models on using those
    # and core features)
    # Fit second stage model
    sgd = SGD(lr=0.15, momentum=.02, nesterov=True)
    # Rescale data
    scaler = StandardScaler()
    X = trn[feats]
    scaler.fit(X)
    X = scaler.transform(X)
    model.compile(loss='mean_squared_error', optimizer='rmsprop')
    model.fit(X,
              trn.target.values,validation_split=0.15)
    # Write predictions
    val_X = scaler.transform(val[feats])
    test_X = scaler.transform(test[feats])
    val['preds_final'] = 0
    val['preds_final'] = write_nn_preds(val_X, model, '_final', feats, is_test=0)
    val['preds_final'] = val['preds_final'].apply(lambda x: math.exp(x)-1)
    test['preds_final'] = write_nn_preds(test_X, model, '_final', feats, is_test=0)
    test['cost'] = test['preds_final'].apply(lambda x: math.exp(x)-1)
    # Score loop
    score = rmsle(val['cost'], val['preds_final'])
    print "Score for loop %s is: %s" % (cv_fold, score)
    avg_score += score/num_loops
avg_score

test['cost'] = test[['preds12', 'preds13', 'preds14', 'preds15', 'preds16', 'preds17']].mean(axis=1)
test[['preds12', 'preds13', 'preds14', 'preds15', 'preds16', 'preds17']].corr()

# Export test preds
test['id'] = test['id'].apply(lambda x: int(x))
test[['id', 'cost']].to_csv(SUBM_PATH+'uhhh neural network.csv', index=False)
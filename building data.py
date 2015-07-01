__author__ = 'p_cohen'

import pandas as pd
import collections
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model.ridge import RidgeCV
from sklearn.linear_model import RandomizedLasso
from sklearn import svm
import numpy as np
import time
from sklearn.feature_selection import SelectKBest, f_regression
import sklearn as skl
from sklearn.feature_extraction import DictVectorizer
import gc

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

def rename_nonId(x, merge_id, nm):
    if x == merge_id:
        return x
    return nm + '_' + x


for comp_type in comp_types:
    dfs_to_merge['comp_'+comp_type] = 'component_id'

def merge_on_assembly(df):
    """
    This function will merge all of the data sets related to the Kaggle
    competition for Caterpillar which can be merged simply using the
    tube_assembly_id

    """
    # Create list of csvs to merge
    csvs_to_merge = ['tube', 'bill_of_materials', 'specs']
    # Name merge var
    merge_var = 'tube_assembly_id'
    # Loop over csvs to merge
    for nm in csvs_to_merge:
        # Read in csv
        merge_df = pd.read_csv(DATA_PATH + nm +'.csv')
        # Rename columns to format tablename_column_name
        merge_df.rename(columns=lambda x: rename_nonId(x, merge_var, nm),
                        inplace=True)
        # Merge
        df = pd.merge(df, merge_df, on=merge_var, how='left')
    return df

def merge_on_tube_end(df):
    """
    This function handles merges related to tube ends, merges onto main data
    set twice, once for each side (side a and side x)
    """
    # Create list of csvs to merge
    nm = "tube_end_form"
    # Name merge var
    merge_var = "end_form_id"
    # Merge to data about both ends of pipe
    for end in ['_a', '_x']:
        # Read in csv
        merge_df = pd.read_csv(DATA_PATH + nm +'.csv')
        # Rename columns to format tablename_column_name
        merge_df.rename(columns=lambda c: nm + end + '_' + c, inplace=True)
        # Merge
        df = pd.merge(df, merge_df, how='left', left_on='tube_end'+end,
                      right_on=nm + end + '_' + merge_var)
    return df

def clean_component_data(comp_dict):
    """
    Reads in and cleans a component data set
    """
    for name, feat_dict in comp_dict.iteritems():
        df = pd.read_csv(DATA_PATH + 'comp_' + name + '.csv')
        for i in range(0, len(feat_dict)):
            if feat_dict[i]=='num':
                mean = df.ix[:,i].mean()
                df.ix[:,i] = df.ix[:,i].fillna(value=mean)
        return df

clean_component_data(comp_dict)


def add_component_vars(df, comp):
    """
    This merges a comp_* table onto bill_of_materials and takes aggregated
    statistics of each field
    """
    merge_var = "component_id"

    # Counter for number of times merge matches

    for slot in range(1, 9):
        # Read in csv
        merge_df = pd.read_csv(DATA_PATH + 'comp_' + comp + '.csv')
        # Rename columns to format tablename_column_name
        merge_df.rename(columns=lambda c: comp + str(slot) + '_' + c,
                        inplace=True)
        # Merge
        left_merge_var = 'bill_of_materials_' + merge_var + '_' + str(slot)
        right_merge_var = comp + str(slot) + '_' + merge_var
        df = pd.merge(df, merge_df, how='left',
                      left_on=left_merge_var,
                      right_on=right_merge_var)
    for var in ['length', 'thread_size', 'weight', 'diameter', 'seat_angle', 'hex_nut_size', 'thread_pitch']:
        var_list = []
        for i in range(1, 9):
            var_list.append(comp + str(i) + '_' + var)
        df[comp+'_'+var] = df[var_list].mean(axis=1)
        df = df.drop(var_list, axis=1)
    return df

add_component_vars(non_test, 'nut')

def merge_all_components(df):
    """
    This merges all possible component types onto all component "slots" in
    bill_of_materials
    :param df:
    :return:
    """
    for i in range(1, 9):
        df = merge_on_component(df, i, comp_types)
    return df

######################################################
tube = pd.read_csv(DATA_PATH + 'tube.csv')
tube = tube.rename(columns=lambda x: 'tube_'+x)
for x in range(1, len(tube.columns.values)):
    tube.columns[x] = 'tube_' + tube.columns.values[x]

############### Define Globals ########################
DATA_PATH = '/home/vagrant/caterpillar-peter/Original/'
CLN_PATH = '/home/vagrant/caterpillar-peter/Clean/'

######################################################
# Data dictionary for cleaning comp tables
comp_dict = {
    'boss': ['id', 'id', 'cat', 'cat', 'cat', 'cat', 'num', 'num', 'num', 'bin', 'num', 'num', 'bin', 'bin', 'num']
}


# Load train and test data
non_test = pd.read_csv(DATA_PATH + 'train_set.csv')
test = pd.read_csv(DATA_PATH + 'test_set.csv')
comp_types = ['adaptor', 'boss', 'elbow', 'float', 'hfl', 'nut', 'other',
              'sleeve', 'straight', 'tee', 'threaded']
comp_types = ['adaptor',]
# Merge on needed data sets
non_test = merge_on_assembly(non_test)
non_test = merge_on_tube_end(non_test)
non_test = add_component_vars(non_test, 'nut')

# Split non_test into train and validation samples
trn, val = create_val_and_train(non_test, 42, 'tube_assembly_id')

trn
val
non_test

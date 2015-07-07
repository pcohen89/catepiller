__author__ = 'p_cohen'

import pandas as pd
import numpy as np
from sklearn import ensemble, preprocessing

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
            # Cleaning steps for categorical or binary variables
            if ((feat_dict[i]=='cat') | (feat_dict[i]=='bin')):
                lbl = preprocessing.LabelEncoder()
                lbl.fit(list(df.ix[:,i]))
                df.ix[:,i] = lbl.transform(df.ix[:,i])
        # Spot fix nut (numeric column with string values
        if name == 'nut':
            df.ix[df.thread_size == 'M12', 'thread_size'] = -1
            df.ix[df.thread_size == 'M10', 'thread_size'] = -2
            df.ix[df.thread_size == 'M8', 'thread_size'] = -3
            df.ix[df.thread_size == 'M6', 'thread_size'] = -4
        if name == 'threaded':
            df.ix[df.nominal_size_1 == 'See Drawing', 'nominal_size_1'] = -1
            df['nominal_size_4'] = -4
        df.to_csv(CLN_PATH + 'comp_' + name + '.csv', index=False)

def aggregate_compslots(df, comp, comp_var_list):
    """
    If a given component type has multiple instances in one asssembly (i.e an
    assembly has three nuts on it) then this function creates summary statistics
    that aggregate field values for each instance of that component type
    """
    # Count how many times this assembly matches this comp type
    comp_ids = []
    for comp_slot in range(1, 9):
        comp_ids.append(comp + str(comp_slot) + '_component_id')
    df[comp+"_count"] = df[comp_ids].count(axis=1)
    # Create list of all component type columns
    cols = pd.read_csv(CLN_PATH + 'comp_' + comp + '.csv').columns.values
    # Loop through list of columns
    for i in range(0, len(cols)):
        var = cols[i]
        # Create list of variables to aggregate
        var_list = []
        for comp_slot in range(1, 9):
            var_list.append(comp + str(comp_slot) + '_' + var)
        # Save the root of the naming convention for aggregated variables
        base_nm = comp + '_' + var
        # use dictionary to check if column is numeric
        if ((comp_var_list[i] == 'num') or (comp_var_list[i] == 'bin')):
            # Aggregate list
            df[base_nm+"_median"] = df[var_list].mean(axis=1)
        if comp_var_list[i] != 'id':
            # Store max and min values of variable across types
            df[base_nm+"_max"] = df[var_list].max(axis=1)
            df[base_nm+"_min"] = df[var_list].min(axis=1)
        # drop unaggregated variables
        df = df.drop(var_list, axis=1)
    return df

def add_component_vars(df, comp, comp_var_list):
    """
    This merges a comp_* table onto bill_of_materials and takes aggregated
    statistics of each field
    """
    merge_var = "component_id"
    # Try merging component data against each component 'slot'
    for slot in range(1, 9):
        # Read in csv
        merge_df = pd.read_csv(CLN_PATH + 'comp_' + comp + '.csv')
        # Rename columns to format tablename_column_name
        merge_df.rename(columns=lambda c: comp + str(slot) + '_' + c,
                        inplace=True)
        # Merge
        left_merge_var = 'bill_of_materials_' + merge_var + '_' + str(slot)
        right_merge_var = comp + str(slot) + '_' + merge_var
        df = pd.merge(df, merge_df, how='left', left_on=left_merge_var,
                      right_on=right_merge_var)

    df = aggregate_compslots(df, comp, comp_var_list)
    return df

def reduce_num_levels(df, col, min_obs):
    """ Reduces the number of levels in a given variable """
    # Group by the variable of interest
    grouped = df.groupby(col)
    # Take counts of the different levels in that variable
    df_counts = grouped['tube_assembly_id'].count().reset_index()
    # Merge counts onto original data
    df = df.merge(df_counts, on=col)
    # Set all levels with few counts to Other
    df[col][df[0]<min_obs] = -1
    return df[col]

def clean_merged_df(df):
    """
    Once normalized tables have been merged, this function will clean for
    variable construction and then modeling
    """
    # Remove all ids other than tube_assembly_id
    cols = list(df.columns.values)
    # Separate date variable
    df['year'] = df.quote_date.apply(lambda x: x[0:4])
    df['month'] = df.quote_date.apply(lambda x: x[5:7])
    df['dayofyear'] = df.quote_date.apply(lambda x: x[8:10])
    df.drop('quote_date', axis=1)
    # Create list of categorical and numeric vars
    num_cols = df._get_numeric_data().columns.values
    for col in num_cols:
        mean = df[col].mean()
        df[col] = df[col].fillna(value=mean)
    # Encode categorical as numeric levels
    cat_cols = list(set(cols)-set(num_cols)-set(['tube_assembly_id',]))
    for col in cat_cols:
        df[col] = df[col].fillna(value="-1")
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(df[col]))
        df[col] = lbl.transform(df.ix[:,col])
    return df

def build_vars(df):
    """ This builds some miscellaneous variables for modeling """
    all_cols = df.columns.values
    weight_cols = []
    quant_col = []
    for col in all_cols:
        if 'weight_median' in col:
            weight_cols.append(col)
        if 'materials_quantity' in col:
            quant_col.append(col)
    df['comp_weight_sum'] = df[weight_cols].sum(axis=1)
    df['comp_tot_cnt'] = df[quant_col].sum(axis=1)
    # Reduce number of levels in categorical vars
    #for col in all_cols:
    #    num_levels = len(df[col].drop_duplicates())
    #    # Treat variables with fewer than 500 levels as categorical
    #    if num_levels < 150:
    #        # Recode levels with fewer than 300 obs as -1
    #        df[col] = reduce_num_levels(df, col, 50)
    return df

############### Define Globals ########################
DATA_PATH = '/home/vagrant/caterpillar-peter/Original/'
CLN_PATH = '/home/vagrant/caterpillar-peter/Clean/'

######################################################
# Data dictionary for cleaning comp tables
# Data dictionary for cleaning comp tables
comp_dict = {
    'boss': [
        'id', 'cat', 'cat', 'cat', 'cat', 'cat', 'num', 'num',
        'num', 'bin', 'num', 'num', 'bin', 'bin', 'num'
    ],
    'adaptor': [
        'id', 'cat', 'num', 'num', 'cat', 'cat', 'num', 'num', 'num', 'num', 'cat',
        'cat', 'num', 'num', 'num', 'num', 'num', 'bin', 'bin', 'num'
    ],
    'elbow': [
        'id', 'cat', 'num', 'num', 'num', 'num', 'num', 'num', 'num',
        'cat', 'cat', 'num', 'bin', 'bin', 'bin', 'num'
    ],
    'float': [
        'id', 'cat', 'num', 'num', 'num', 'bin', 'num'
    ],
    'hfl': [
        'id', 'cat', 'num', 'cat', 'cat', 'cat', 'bin', 'bin', 'num'
    ],
    'nut': [
        'id', 'cat', 'num', 'num', 'num', 'num', 'num', 'num',
        'bin', 'bin', 'num'
    ],
    'other': [
        'id', 'cat', 'num'
    ],
    'sleeve': [
        'id', 'cat', 'cat', 'num', 'num', 'num', 'bin', 'bin', 'bin', 'num'
    ],
    'straight': [
        'id', 'cat', 'num', 'num', 'num', 'num', 'num', 'cat', 'bin', 'bin',
        'bin', 'num'
    ],
    'threaded': [
        'id', 'cat', 'num', 'num', 'num', 'cat', 'cat', 'num', 'num', 'num', 'num',
        'cat', 'cat', 'num', 'num', 'num', 'num', 'cat', 'cat', 'num', 'num',
        'num', 'num', 'cat', 'cat', 'num', 'num', 'num', 'num', 'bin', 'bin',
        'num'
    ]
}


# Load train and test data
non_test = pd.read_csv(DATA_PATH + 'train_set.csv')
non_test['is_test'] = 0
test = pd.read_csv(DATA_PATH + 'test_set.csv')
test['is_test'] = 1
# Append test and train data
all_data = non_test.append(test)

# Clean component data
clean_component_data(comp_dict)

# Merge on needed data sets
all_data_wassembly = merge_on_assembly(all_data)
all_data_wtubeend = merge_on_tube_end(all_data_wassembly)
for name, field_dict in comp_dict.iteritems():
    all_data_wtubeend = add_component_vars(all_data_wtubeend, name, field_dict)
cleaned_all_data = clean_merged_df(all_data_wtubeend)
all_data_complete = build_vars(cleaned_all_data)
all_data_complete.to_csv(CLN_PATH + "full_data.csv", index=False)


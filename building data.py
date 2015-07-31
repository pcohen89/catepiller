__author__ = 'p_cohen'

import pandas as pd
import math
from sklearn import ensemble, preprocessing

############### Define Functions ########################
def merge_noncomp(df, nm, left_merge, right_merge):
    """
    Merges the non component data sets onto the main training df
    :param df: Main data set we are merging variables onto
    :param nm: name of csv to merge on
    :param left_merge: (str) left merge variable
    :param right_merge: (str) right merge variable
    :return: df
    """
    # Read in csv
    merge_df = pd.read_csv(DATA_PATH + nm +'.csv')
    # Rename columns to format tablename_column_name (if not merge variable)
    columns = merge_df.columns.values
    for col in columns:
        if col != right_merge:
            merge_df.rename(columns={col: nm + '_' + col}, inplace=True)
    # Merge
    df = pd.merge(df, merge_df, left_on=left_merge,
                  right_on=right_merge, how='left')
    return df

def clean_component_data(comp_dict):
    """
    Reads in and cleans a Caterpiller tube component data sets according to
    the columns types (numeric, character, binary, etc.) stored in feat_list
    Some particular dfs such as adaptor have special cleaning steps

    feat_list : (dict) in format 'name of table' : column 1 type, column 2 type,
    column 3 type, .....

    """
    # Loop through
    for name, feat_list in comp_dict.iteritems():
        df = pd.read_csv(DATA_PATH + 'comp_' + name + '.csv')
        if name == 'adaptor':
            df_type = pd.read_csv(CLN_PATH + 'type_connection.csv')
            id_root = 'connection_type_id'
            df = df.merge(df_type, left_on=id_root + '_1', right_on=id_root)
            df = df.merge(df_type, left_on=id_root + '_2', right_on=id_root)
            drop_vars = ['connection_type_id_x', 'connection_type_id_y',
                         'name_x', 'name_y']
            for var in drop_vars:
                df = df.drop(var, axis=1)
        for i in range(0, len(feat_list)):
            # Cleaning steps for categorical or binary variables
            if ((feat_list[i]=='cat') | (feat_list[i]=='bin')):
                lbl = preprocessing.LabelEncoder()
                lbl.fit(list(df.ix[:,i]))
                df.ix[:,i] = lbl.transform(df.ix[:,i])
        # Spot fix nut (numeric column with string values
        if name == 'nut':
            # Encode as missing number
            for str_ in ['M12', 'M10', 'M8', 'M6']:
                df.ix[df.thread_size == str_, 'thread_size'] = -1*int(str_[1:])
        if name == 'threaded':
            df.ix[df.nominal_size_1 == 'See Drawing', 'nominal_size_1'] = -1
            df['nominal_size_4'] = -4
        df.to_csv(CLN_PATH + 'comp_' + name + '.csv', index=False)

def clean_type_connection(path, outpath):
    """
    :param path: path where type_connection.csv is located
    :param outpath: path to save new type_connection.csv
    :return:
    """
    # Read data
    df_connect = pd.read_csv(path+'type_connection.csv')
    # Build variables
    df_connect['has_flare'] = df_connect['name'].apply(lambda x: 'Flare' in x)
    df_connect['has_flange'] = df_connect['name'].apply(lambda x: 'Flange' in x)
    df_connect['has_metric'] = df_connect['name'].apply(lambda x: 'Metric' in x)
    df_connect['has_sae'] = df_connect['name'].apply(lambda x: 'SAE' in x)
    # Export
    df_connect.to_csv(outpath + 'type_connection.csv', index=False)

def clean_specs(path):
    """
    Re-shapes specs, because spec1 = SP-004 and spec2 = SP-004 intuitively
    should mean the same thing but because of an artifact of the storage
    they will be treated as totally unrelated.

    :param specs: Specs.csv data set from caterpiller comp
    :param path: path to save new specs file to
    """
    # Read in csv
    df = pd.read_csv(DATA_PATH + 'specs.csv')
    # Initialize list of all values that appear
    spec_vals = df.spec1.drop_duplicates()
    for spec_num in range(2, 11):
        spec_vals = spec_vals.append(df['spec'+str(spec_num)].drop_duplicates())
    spec_vals = list(spec_vals)
    # Check to see if nan
    spec_vals = [s for s in spec_vals if s == s]
    # Make binaries to represent whether tube has that spec
    for spec in spec_vals:
        temp_list = []
        # Create true false for whether a column has a spec
        for spec_col in range(1, 11):
            cnt = str(spec_col)
            temp_list.append('temp'+cnt)
            df['temp'+cnt] = df['spec'+cnt].apply(lambda x: spec in str(x))
        # Record whether any column contains spec
        df['s_'+spec[3:]] = df[temp_list].max(axis=1)
        # Drop column if spec value is very rare
        if df['s_'+spec[3:]].mean() < .0025:
            df = df.drop('s_'+spec[3:], axis=1)
        df = df.drop(temp_list, axis=1)
    df.to_csv(DATA_PATH + 'reshaped_specs.csv', index=False)

def aggregate_compslots(df, comp, comp_var_list):
    """
    This function adds component data to the main df using matches from
    bill of materials.
    If a given component type has multiple instances in one assembly (i.e an
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
        # Define merging variables
        left_merge_var = 'bill_of_materials_' + merge_var + '_' + str(slot)
        right_merge_var = comp + str(slot) + '_' + merge_var
        # Merge
        df = pd.merge(df, merge_df, how='left', left_on=left_merge_var,
                      right_on=right_merge_var)
    df = aggregate_compslots(df, comp, comp_var_list)
    return df

def clean_merged_df(df):
    """
    Once normalized tables have been merged, this function will clean for
    variable construction and then modeling
    """
    # Remove all ids other than tube_assembly_id
    cols = list(df.columns.values)
    # Separate date variable (using old python, built in date functions are
    # out of date and I can't find old docs)
    df['year'] = df.quote_date.apply(lambda x: x[0:4])
    df['month'] = df.quote_date.apply(lambda x: x[5:7])
    df['dayofyear'] = df.quote_date.apply(lambda x: x[8:10])
    # Create list of categorical and numeric vars
    num_cols = df._get_numeric_data().columns.values
    for col in num_cols:
        mean = df[col].mean()
        df[col] = df[col].fillna(value=mean)
    # Treat non numeric cols as categoricals
    cat_cols = list(set(cols)-set(num_cols)-set(['tube_assembly_id',]))
    # Encode categoricals as numerics
    for col in cat_cols:
        # Fill column missings with -1
        df[col] = df[col].fillna(value="-1")
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(df[col]))
        df[col] = lbl.transform(df.ix[:,col])
    return df

def add_supp_var(df):
    """
    Creates a few supplier variables
    :param df: dataframe to create variables in
    :return: dataframe with supplier variable
    """
    grpd = df.groupby('supplier')
    cnts_by_supplier = grpd['tube_assembly_id'].count().reset_index()
    cnts_by_supplier = cnts_by_supplier.rename(columns={0: 'supplier_freq'})
    df = df.merge(cnts_by_supplier, on='supplier')
    return df

def build_vars(df):
    """ This builds some miscellaneous variables that I
    manually determined could be useful for modeling """
    all_cols = df.columns.values
    summ_cols = {'weight_median': [], 'materials_quantity': [], 'specs_': [],
                 'unique': [], 'thick': [], 'orient': [], 'plating': []}
    # Create lists of columns with similar naming conventions
    for col in all_cols:
        for key, val in summ_cols.iteritems():
            if key in col:
                val.append(col)
    # Create miscellaenous variables
    df['comp_weight_sum'] = df[summ_cols['weight_median']].sum(axis=1)
    df['apprx_density'] = df.comp_weight_sum/ (df.tube_length + .01)
    df['length_x_wall'] = df.tube_length * df.tube_wall
    df['radius_per_bend'] = df.tube_bend_radius/(df.tube_num_bends + .01)
    df['bend_per_length'] = df.tube_bend_radius/(df.tube_length + .01)
    df['ext_over_overall'] = (df.elbow_extension_length_max /
                                df.elbow_overall_length_max + .01)
    df['thick_ove_len'] = (df.elbow_thickness_max /
                                df.elbow_overall_length_max + .01)
    df['comp_tot_cnt'] = df[summ_cols['materials_quantity']].sum(axis=1)
    df['specs_cnt'] = df[summ_cols['specs_']].sum(axis=1)
    df['plate_cnt'] = df[summ_cols['plating']].sum(axis=1)
    df['unq_cnt'] = df[summ_cols['unique']].sum(axis=1)
    df['thick_cnt'] = df[summ_cols['thick']].sum(axis=1)
    df['orient_cnt'] = df[summ_cols['orient']].sum(axis=1)
    df['is_min_order_quantity'] = df['min_order_quantity'] > 0
    df['ext_as_pct'] = df.elbow_extension_length_min/df.elbow_overall_length_min
    df = df.fillna(-1)
    # Drop variables with no variation
    print len(df.columns.values)
    no_variation = ['sleeve_ori', 'sleeve_component_type',
                    'comp_component_type_id', 'boss_orient', 'nut_orient']
    for feat in all_cols:
        if any(x in feat for x in no_variation):
            df = df.drop(feat, axis=1)
    print len(df.columns.values)
    return df


############### Define Globals ########################
DATA_PATH = '/home/vagrant/caterpillar-peter/Original/'
CLN_PATH = '/home/vagrant/caterpillar-peter/Clean/'

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
    'float': ['id', 'cat', 'num', 'num', 'num', 'bin', 'num'],
    'hfl': ['id', 'cat', 'num', 'cat', 'cat', 'cat', 'bin', 'bin', 'num'],
    'nut': [
        'id', 'cat', 'num', 'num', 'num', 'num', 'num', 'num',
        'bin', 'bin', 'num'
    ],
    'other': ['id', 'cat', 'num'],
    'sleeve': [
        'id', 'cat', 'cat', 'num', 'num', 'num', 'bin', 'bin', 'bin', 'num'
    ],
    'straight': [
        'id', 'cat', 'num', 'num', 'num', 'num', 'num', 'cat', 'bin', 'bin',
        'bin', 'num'
    ],
    'tee': [
        'id', 'cat', 'num', 'num', 'num', 'num', 'num', 'num', 'cat',
        'cat', 'cat', 'cat', 'cat', 'num'
    ],
    'threaded': [
        'id', 'cat', 'num', 'num', 'num', 'cat', 'cat', 'num', 'num', 'num',
        'num', 'cat', 'cat', 'num', 'num', 'num', 'num', 'cat', 'cat', 'num',
        'num', 'num', 'num', 'cat', 'cat', 'num', 'num', 'num', 'num', 'bin',
        'bin', 'num'
    ]
}

#############################################################################
############### Run live code ###############################################
#############################################################################

# Load train and test data
non_test = pd.read_csv(DATA_PATH + 'train_set.csv')
non_test['is_test'] = 0
test = pd.read_csv(DATA_PATH + 'test_set.csv')
test['is_test'] = 1
# Append test and train data
all_data = non_test.append(test)
# Create supplier variable
all_data = add_supp_var(all_data)
# Clean data sets
clean_type_connection(DATA_PATH, CLN_PATH)
clean_component_data(comp_dict)
clean_specs(DATA_PATH)
# After cleaning, adaptor has more columns
comp_dict['adaptor'] = [
    'id', 'cat', 'num', 'num', 'cat', 'cat', 'num', 'num', 'num', 'num', 'cat',
    'cat', 'num', 'num', 'num', 'num', 'num', 'bin', 'bin', 'num',
    'bin', 'bin', 'bin', 'bin', 'bin', 'bin', 'bin', 'bin',
]
# Merge on needed data sets
tube_merge_csvs = ['tube', 'bill_of_materials', 'reshaped_specs']
for csv in tube_merge_csvs:
    all_data = merge_noncomp(all_data, csv, 'tube_assembly_id',
                             'tube_assembly_id')
# Merge on tube ends, matching to both sides
for tube_end in ['a', 'x']:
    all_data = merge_noncomp(all_data, "tube_end_form",
                             "tube_end_" + tube_end, "end_form_id")
# Rename data after adding datasets
all_data_wtubeend = all_data
# Iteratively merge on each component data set
for name, field_dict in comp_dict.iteritems():
    all_data_wtubeend = add_component_vars(all_data_wtubeend, name, field_dict)
# Clean the resultant dataframe
cleaned_all_data = clean_merged_df(all_data_wtubeend)
# Build modeling vars
all_data_complete = build_vars(cleaned_all_data)
# Export
all_data_complete.to_csv(CLN_PATH + "full_data.csv", index=False)


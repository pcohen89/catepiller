__author__ = 'p_cohen'

from __builtin__ import list, range, len, str, set, any, int

import pandas as pd
from sklearn import preprocessing
import numpy as np
from datetime import datetime

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
    # Loop through components
    for name, feat_list in comp_dict.iteritems():
        df = pd.read_csv(DATA_PATH + 'comp_' + name + '.csv')
        # Execute adaptor specific cleaning steps
        if name == 'adaptor':
            # Merge on types
            df_type = pd.read_csv(CLN_PATH + 'type_connection.csv')
            id_root = 'connection_type_id'
            df = df.merge(df_type, left_on=id_root + '_1', right_on=id_root)
            df = df.merge(df_type, left_on=id_root + '_2', right_on=id_root)
            # Drop duplicative variables
            drop_vars = ['connection_type_id_x', 'connection_type_id_y',
                         'name_x', 'name_y']
            for var in drop_vars:
                df = df.drop(var, axis=1)
        for i in range(0, len(feat_list)):
            # Cleaning steps for categorical or binary variables
            if ((feat_list[i]=='cat') | (feat_list[i]=='bin')):
                lbl = preprocessing.LabelEncoder()
                lbl.fit(list(df.ix[:, i]))
                df.ix[:, i] = lbl.transform(df.ix[:, i])
        # Spot fix nut (numeric column with string values)
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
    # Extract common wordes from connection names
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
            df[base_nm+"_median"] = df[var_list].median(axis=1)
        if comp_var_list[i] != 'id':
            # Store max and min values of variable across types
            df[base_nm+"_max"] = df[var_list].max(axis=1)
            df[base_nm+"_min"] = df[var_list].min(axis=1)
        # drop unaggregated variables
        df = df.drop(var_list, axis=1)
    return df

def gen_bill_vars(bill_path, comp_path):
    """
    This file extracts information from component files differently than
    previous attempts, tries to aggregate information from multiple components
    e.g. sum of weight of all components

    :param bill_path: path to bill of materials
    :param comp_path: path to component files
    :param cln_path: path to output dataframe
    :return: dataframe with tube assembly ids and new variables
    """
    bill = pd.read_csv(bill_path)
    # Create count of frequency of each of the components for each assmb
    bill = make_comp_popfreqs(bill)
    types = ['boss', 'adaptor', 'elbow', 'float', 'hfl', 'nut',
             'other', 'sleeve', 'straight', 'tee', 'threaded']
    # Store new column names
    new_cols = ['adjusted_wt', 'adjusted_unique_cnt', 'overall_len_sum',
                'groove_cnt', 'thickness_sum', 'plating_cnt']
    for type in types:
        new_cols.append(type+'_cnt')
    # Initialize new columns
    for col in new_cols:
        bill[col] = 0
    # Recode NA to zero in bill of materials for quantity and component_id
    # columns
    for slot in range(1, 9):
        quant = 'quantity_' + str(slot)
        comp_id = 'component_id_' + str(slot)
        bill[quant] = bill[quant].fillna(value=0)
        bill[comp_id] = bill[comp_id].fillna(value="NA")
    # Create count of each component set
    bill = create_freq_of_compset(bill)
    cols_to_keep = bill.columns.values
    # Analyze each component type
    for comp in types:
        # Create and prepare component dataframe
        comp_df = create_comp_for_billvars(comp, comp_path)
        # Loop over bill slots, merge comp df to each
        for slot in range(1, 9):
            bill = merge_compslot_for_billvars(comp_df, bill, slot,
                                               cols_to_keep, comp)
    # Append extra vars
    for ex_var in ['tube_assembly_id', 'compset_cnt']:
        new_cols.append(ex_var)
    return bill[new_cols]


def create_comp_for_billvars(comp, comp_path):
    """
    Creates and preps a component for merges onto bill
    """
    component = pd.read_csv(comp_path+'comp_'+comp+".csv")
    # Create a indicator for determining whether row merged
    component['is_merged'] = 1
    # code nulls to zero
    component.ix[component.weight.isnull(), 'weight'] = 0
    # Code feature to boolean, make sure they exist
    vars_to_create_no = ['unique_feature', 'groove', 'plating']
    for var in vars_to_create_no:
        # If variable isn't in this component, create all no's
        if var not in component:
            component[var] = 'No'
        # Cast to boolean
        component[var] = component[var] == 'Yes'
    # Code overall length to zero if missing
    vars_to_create_zero = ['overall_length', 'thickness']
    for var in vars_to_create_zero:
        if var not in component:
            component[var] = 0
    return component


def merge_compslot_for_billvars(comp_df, main_df, slot, cols_to_keep, comp):
    """
    This function merges a particular component data set to each component
    slot in bill of materials and aggregates the information from each into
    the aggregation variables

    :param comp_df: (dataframe) component data set
    :param main_df: (dataframe) main data set with bill of materials data and
    aggregated variables
    :param slot: (int) controls which bill of materials slot to merge the
    component data set to
    :param cols_to_keep: (list) variables that should be returned in main_df
    :return: main_df with aggregated variables that represent the newly merged
    info
    """
    # Store the stemmed merge variable
    merge_nm = 'component_id'
    # Merge component on to a single slot
    main_df = main_df.merge(comp_df, how="left",
                      left_on=merge_nm + '_' +str(slot),
                      right_on=merge_nm)
    # code is_merged as zero if record didn't merge
    zero_vars = ['is_merged', 'weight', 'unique_feature',
                 'overall_length', 'groove', 'plating', 'thickness']
    for zero_var in zero_vars:
        main_df.ix[main_df[zero_var].isnull(), zero_var] = 0
    # Increment the number of pieces for the component
    quant_mergd = main_df.is_merged * main_df['quantity_'+str(slot)]
    main_df[comp+'_cnt'] += quant_mergd
    new_features = {
        'adjusted_wt': 'weight', 'groove_cnt': 'groove', 'plating_cnt':
        'plating', 'adjusted_unique_cnt': 'unique_feature',
        'overall_len_sum': 'overall_length', 'thickness_sum': 'thickness'
     }
    for new_name, base_var in new_features.iteritems():
        main_df[new_name] += (quant_mergd * main_df[base_var])
    # Drop component specific vars
    main_df = main_df.ix[:, cols_to_keep]
    return main_df


def create_freq_of_compset(df):
    """
    Creates a count of number of times tube's component set appears in
    data
    """
    comp_ids = ['component_id_1', 'component_id_2', 'component_id_3',
                'component_id_4', 'component_id_5', 'component_id_6',
                'component_id_7', 'component_id_8']
    # Create counts by component set
    grouped = df.groupby(comp_ids)
    counts = grouped.tube_assembly_id.count().reset_index()
    counts = counts.rename(columns={0: 'compset_cnt'})
    df = df.merge(counts, on=comp_ids)
    return df

def make_comp_popfreqs(bill):
    """
    This summarizes how frequently each component
    :param bill: bill_of_materials df
    :return: bill_of materials with a measure of how frequently each
     tube assembly's component appear in the data
    """
    slot1 = bill[['quantity_1', 'component_id_1']]
    slot2 = bill[['quantity_2', 'component_id_2']]
    slot3 = bill[['quantity_3', 'component_id_3']]
    slot4 = bill[['quantity_4', 'component_id_4']]
    slots = [slot1, slot2, slot3, slot4]
    for slot in slots:
        slot.columns = ['quant', 'comp']
    # append all of the column slots
    all_comps = pd.concat([slot1, slot2, slot3, slot4], ignore_index=True)
    # drop NAs
    all_comps = all_comps[np.isfinite(all_comps.quant)]
    # Get sums of frequencies
    grouped = all_comps.groupby('comp')
    counts = grouped.sum().reset_index()
    # Merge counts on to each compslot
    bill['freqs_for_comps'] = 0
    for slot in ['1', '2', '3', '4']:
        bill = bill.merge(counts, how='left',
                          left_on='component_id_'+str(slot), right_on='comp')
        bill['freqs_for_comps'] += bill.quant
        bill = bill.drop('quant', axis=1)
        bill = bill.drop('comp', axis=1)
    return bill


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
    df['dayofweek'] = df.quote_date.apply(
        lambda x: datetime.strptime(x, "%Y-%m-%d").weekday()
    )
    # Create list of categorical and numeric vars
    col_types = df.dtypes.reset_index()
    num_col_rows = col_types[col_types[0] != 'object']
    num_cols = list(num_col_rows['index'])
    for col in num_cols:
        if col == 'quote_date':
            continue
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
        df[col] = lbl.transform(df.ix[:, col])
    return df

def make_onehot_data(df, cat_cols, freq_thresh=.004):
    """
    This adds dummies to a data frame to represent a categorical variable
    as a set of dummy variables

    :param df: dataframe to encode one hot variables into
    :param cols: (dict) key - column in df to onehot encode, value - naming
    convention for the dummies
    :param freq_thresh: threshold for % of time a dummy needs to be 1 to be
    included
    :return: df with dummies
    """
    # loop through columns and nameing conventions
    for col, name in cat_cols.iteritems():
        # Create all dummies for column of interest
        onehot = pd.get_dummies(df[col], prefix=name)
        # Calc means of dummiers
        means = onehot.mean().reset_index()
        # Drop dummies that appear infrequently
        means = means[means[0]> freq_thresh]
        cols_to_keep = list(means['index'])
        onehot = onehot[cols_to_keep]
        # Append columnwise
        df = pd.concat([df, onehot], axis=1)
        df.drop(col, axis=1)
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
    df['apprx_density'] = df.comp_weight_sum/(df.tube_length + .01)
    df['adj_apprx_density'] = df.adjusted_wt/(df.tube_length + .01)
    df['length_x_wall'] = df.tube_length * df.tube_wall
    df['radius_per_bend'] = df.tube_bend_radius/(df.tube_num_bends + .01)
    df['bend_per_length'] = df.tube_bend_radius/(df.tube_length + .01)
    df['dia_over_len'] = df.tube_diameter/(df.tube_length + .01)
    df['wall_over_diam'] = df.tube_wall/(df.tube_diameter + .01)
    df['len_x_dai'] = df.tube_length * df.tube_diameter
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
    df['ann_use_ove_min'] = df.annual_usage / (df.min_order_quantity + 1)
    df['ann_use_ove_q'] = df.annual_usage / (df.quantity + 1)
    df['is_min_order_quantity'] = df['min_order_quantity'] > 0
    df['ext_as_pct'] = (df.elbow_extension_length_max/
                        df.elbow_overall_length_max)
    df['data_bug1'] = ((df.min_order_quantity > 0) &
                       (df.bracket_pricing == 1))
    df = df.fillna(-1)
    # Drop variables with no variation
    no_variation = ['sleeve_ori', 'sleeve_component_type',
                    'comp_component_type_id', 'boss_orient', 'nut_orient']
    for feat in all_cols:
        if any(x in feat for x in no_variation):
            df = df.drop(feat, axis=1)
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
all_data_onehot = pd.DataFrame(all_data)
# Create a small oneHot data set (old pandas has a much less capable
#  get_dummies
columns_for_onehot = {'tube_material_id': 'tube_mat',
                      'bill_of_materials_component_id_1': 'comp1',
                      'bill_of_materials_component_id_2': 'comp2',
                      'bill_of_materials_component_id_3': 'comp3',
                      'bill_of_materials_component_id_4': 'comp4'
                     }
all_data_onehot = make_onehot_data(all_data_onehot, columns_for_onehot)
cleaned_all_data_onehot = clean_merged_df(all_data_onehot)
# Iteratively merge on each component data set
for name, field_dict in comp_dict.iteritems():
    all_data_wtubeend = add_component_vars(all_data_wtubeend, name, field_dict)
# Clean the resultant dataframe
cleaned_all_data = clean_merged_df(all_data_wtubeend)
# Merge and create bill vars
bill = gen_bill_vars(DATA_PATH+'bill_of_materials.csv', DATA_PATH)
cleaned_all_data = cleaned_all_data.merge(bill, on='tube_assembly_id')
# Build modeling vars
all_data_complete = build_vars(cleaned_all_data)
# Export
all_data_complete.to_csv(CLN_PATH + "full_data.csv", index=False)


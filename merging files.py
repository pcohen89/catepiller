__author__ = 'p_cohen'

###############Import packages ########################
import pandas as pd

############### Define Globals ########################
SUBM_PATH = '/home/vagrant/caterpillar-peter/Submissions/'

############### Define Functions ########################
def subm_correl(subm1, subm2, id, target):
    """
    Measures correlation between to Kaggle submissions
    """
    subm1 = pd.read_csv(SUBM_PATH + subm1)
    subm2 = pd.read_csv(SUBM_PATH + subm2)
    subm2 = subm2.rename(columns={target: 'target2'})
    merged_df = subm1.merge(subm2, on=id)
    return merged_df.corr()

def merge_subms(subm_dict, path, name, target):
    """
    :param subm_dict: Dictionary of dfs to merge, where key is csv name and
    value is weight (values must sum to 1 to keep outcome in original range
    :param path: path to submission folder
    :param name: name of new file
    :param target: outcome variable of submission
    :return:
    """
    subm = pd.read_csv(path+'template.csv')
    for csv, weight in subm_dict.iteritems():
        # Read in a new csv
        score = pd.read_csv(path+csv)
        # rename target to avoid merge issues
        score = score.rename(columns={target: 'target2'})
        # Merge files to be averaged
        subm = subm.merge(score, on='id')
        subm[target] += weight * subm['target2']
        subm = subm.drop('target2', 1)
    subm.to_csv(path+name, index=False)


####################### Run Code #########################

# Merge submissions
subs_to_merge = {
    '4000 trees xgb w spec vars.csv': .6,
    '1500 trees xgb.csv': .03,
    '2500 trees xgb.csv': .03,
    '2500 trees xgb w spec vars.csv': .03,
    '2500 trees xgb w extra vars.csv': .03,
    'threeway vars with forest.csv': .17,
    'stacking with xgboost second stage all vars.csv': .01,
    'stacking with all vars in forest.csv': .01,
    'xgboost from first data build.csv': .01,
    'stacking with higher eta.csv': .01,
    'stacking with ridge vars.csv': .01,
    'cv stack with new vars frst second stage internal 243.csv': .01,
    'stacking first attempt.csv': .01,
    'cv stack.csv': .01,
    'boost from first data build.csv': .01,
    'xgboost with deep trees.csv': .01,
    'randomforest from first data build.csv': .01
}

total_weight = 0
for key, val in subs_to_merge.iteritems():
    total_weight += val
print "The total weight should be 1.0, it is: %s" % (total_weight)
merge_subms(subs_to_merge, SUBM_PATH, 'blend m.csv', 'cost')

subm_correl('blend m.csv',
            '4000 trees xgb w spec vars.csv', 'id', 'cost')


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
    '4000 trees power bill vars 2nd set depth 8.csv': .04,
    'upweighted 4000 trees.csv': .12,
    'downweighted 4000 trees.csv': .11,
    '4000 trees power bill vars.csv': .04,
    '4000 trees power no log.csv': .02,
    '2500 trees power bill vars.csv': .05,
    'weight 2500 trees.csv': .1,
    'downweighted 2500 trees.csv': .03,
    'threeway vars with bill vars.csv': .32,
    'threeway vars with subsamp.csv': .06,
    'threeway vars with forest.csv': .01,
    '1500 trees xgb.csv': .01,
    '2500 trees xgb.csv': .01,
    '2500 trees xgb w spec vars.csv': .01,
    '2500 trees xgb w extra vars.csv': .01,
    'stacking with xgboost second stage all vars.csv': .01,
    'stacking with all vars in forest.csv': .01,
    'xgboost from first data build.csv': .01,
    'stacking with higher eta.csv': .01,
    'stacking with ridge vars.csv': .01,
    'cv stack with new vars frst second stage internal 243.csv': .01
}

total_weight = 0
for key, val in subs_to_merge.iteritems():
    total_weight += val
print "The total weight should be 1.0, it is: %s" % (total_weight)
merge_subms(subs_to_merge, SUBM_PATH, 'blend w.csv', 'cost')

subm_correl('downweighted 4000 trees.csv',
            'upweighted 4000 trees.csv', 'id', 'cost')


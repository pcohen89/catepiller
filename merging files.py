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

def check_weight_and_merge(dict, name):
    """
    :param dict: file, weight pairs
    :param name: name of resulting blended file
    :return: blended file saved to server
    """
    total_weight = 0
    for key, val in dict.iteritems():
        total_weight += val
    print "The total weight should be 1.0, it is: %s" % (total_weight)
    merge_subms(dict, SUBM_PATH, name, 'cost')


####################### Run Code #########################
old_files_to_merge = {
    'threeway vars with subsamp.csv': .08,
    'threeway vars with forest.csv': .1,
    '1500 trees xgb.csv': .1,
    '2500 trees xgb.csv': .1,
    '2500 trees xgb w spec vars.csv': .1,
    '2500 trees xgb w extra vars.csv': .07,
    'stacking with xgboost second stage all vars.csv': .07,
    'stacking with all vars in forest.csv': .08,
    'xgboost from first data build.csv': .08,
    'stacking with higher eta.csv': .07,
    'stacking with ridge vars.csv': .07,
    'cv stack with new vars frst second stage internal 243.csv': .08
}

files2500_to_merge = {
    '2500 trees power bill vars.csv': .08,
    'weight 2500 trees.csv': .31,
    '2500 with new folds.csv': .38,
    'downweighted 2500 trees.csv': .23,

}

hightrees_to_merge = {
    '4000 trees power bill vars 2nd set depth 8.csv': .27,
    'upweighted 4000 trees.csv': .07,
    'upweighted 6500 trees no compset.csv': .13,
    '6500 with new folds.csv': .33,
    'downweighted 4000 trees.csv': .04,
    'downweighted 6500 trees w compset.csv': .05,
    '4000 trees power bill vars.csv': .05,
    '4000 trees power no log.csv': .06,
}

stacking_to_merge = {
  'upweight stack extra stage1 vars.csv': .4,
  'threeway vars with bill vars.csv': .2,
  'stack w new folds.csv': .4
}

check_weight_and_merge(old_files_to_merge, 'old files.csv')
check_weight_and_merge(files2500_to_merge, '2500 files.csv')
check_weight_and_merge(hightrees_to_merge, 'high tree files.csv')
check_weight_and_merge(stacking_to_merge, 'stacking files.csv')

# Merge submissions
subs_to_merge = {
    'high tree files.csv': .4,
    'stacking files.csv': .35,
    '2500 files.csv': .15,
    'old files.csv': .1
}

total_weight = 0
for key, val in subs_to_merge.iteritems():
    total_weight += val
print "The total weight should be 1.0, it is: %s" % (total_weight)
merge_subms(subs_to_merge, SUBM_PATH, 'blend z.csv', 'cost')

subm_correl('blend y.csv',
            'blend z.csv', 'id', 'cost')


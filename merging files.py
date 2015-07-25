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
        score = pd.read_csv(path+csv)
        score = score.rename(columns={'cost': 'target2'})
        subm = subm.merge(score, on='id')
        subm[target] += weight * subm['target2']
        subm = subm.drop('target2', 1)
    subm.to_csv(path+name, index=False)


####################### Run Code #########################

# Merge submissions
submissions_to_merge = {'1500 trees xgb.csv': .20,
                        '2500 trees xgb.csv': .1,
                        '2500 trees xgb w spec vars.csv': .1,
                        '2500 trees xgb w extra vars.csv': .05,
                        'stacking with all vars in forest.csv': .05,
                        'stacking with three comp.csv': .1,
                        'xgboost from first data build.csv': .05,
                        'stacking with higher eta.csv': .10,
                        'stacking with ridge vars.csv': .05,
                        'cv stack with new vars frst second stage internal 243.csv': .05,
                        'stacking first attempt.csv': .03,
                        'cv stack.csv': .03,
                        'boost from first data build.csv': .03,
                        'xgboost with deep trees.csv': .03,
                        'randomforest from first data build.csv': .03
                        }

total_weight = 0
for key, val in submissions_to_merge.iteritems():
    total_weight += val
print total_weight
merge_subms(submissions_to_merge, SUBM_PATH,
            'blend j.csv', 'cost')

subm_correl('uhhh neural network.csv',
            '2500 trees xgb w spec vars.csv', 'id', 'cost')
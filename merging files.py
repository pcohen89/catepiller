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
    subm2 = subm2.rename(columns={'cost': 'target2'})
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
        subm[target] += weight * score[target]
    subm.to_csv(path+name, index=False)


####################### Run Code #########################

# Merge submissions
submissions_to_merge = {'xgboost from first data build.csv': .4,
                        'cv stack with new vars frst second stage internal 243.csv': .1,
                        'stacking with all vars in forest.csv': .4,
                        'stacking first attempt.csv': .02,
                        'cv stack.csv': .02,
                        'boost from first data build.csv': .02,
                        'xgboost with deep trees.csv': .02,
                        'randomforest from first data build.csv': .02}

merge_subms(submissions_to_merge, SUBM_PATH,
            'blend after adding all vars to second stage.csv', 'cost')
subm_correl('cv stack with new vars frst second stage internal 243.csv',
            'stacking with all vars in forest.csv', 'id', 'cost')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import *
from sklearn.multiclass import *
from sklearn.linear_model import *
from sklearn.naive_bayes import *
from sklearn.neural_network import *
from sklearn.svm import *
from sklearn.model_selection import *
from sklearn import metrics

#Importing team data
team_data = pd.read_csv('cbb21.csv', header=0,index_col='TEAM')
#win_perc = pd.Series(team_data['W']/team_data['G'])
#team_data['win%'] = win_perc.values
team_data = pd.DataFrame.drop(team_data, 'W',1)
team_data = pd.DataFrame.drop(team_data, 'G',1)

# Importing Matchup Data
game_data = pd.read_csv('2021Matchups.csv', header=0)
game_data['result'] = game_data['result'].str[0]
#game_data = game_data[game_data['result']=='W']
game_data['team'] = game_data['team'].str.replace('+',' ')
game_data = game_data.set_index('team')
#game_data['winner'] = game_data['team']

# Joining Data
match_up_data = game_data.join(team_data,how="inner")
match_up_data = match_up_data.reset_index().set_index('opponent')
match_up_data = match_up_data.join(team_data,how="inner",rsuffix='_OPP')
match_up_data = match_up_data.rename(columns={'index':'TEAM'})
match_up_data = match_up_data.reset_index().set_index('TEAM').rename(columns={'index':'opponent'})
match_up_data = pd.DataFrame.drop(match_up_data,['CONF','CONF_OPP','date','opponent'],1)

# Variable Selection
def VariableSelection():
    vs_data = match_up_data.dropna()
    array = vs_data.values
    print(len(array[0]))
    Y = array[:,0]
    Y = Y.astype('str')
    vs_data = pd.DataFrame.drop(vs_data,['result','SEED','SEED_OPP'],1)
    array = vs_data.values
    X = array[:,0:37]
    names = list(vs_data.columns)
    vs_dict = {}
    for i in range(len(names)):
        vs_dict[names[i]] = []
    for i in range(100):
        model = ExtraTreesClassifier(n_estimators=100)
        model.fit(X, Y)
        importance = list(model.feature_importances_)
        for j in range(len(names)):
            vs_dict[names[j]].append(importance[j])
    for i in range(len(names)):
        vs_dict[names[i]] = np.mean(vs_dict[names[i]])
    vs_dict = {k: v for k, v in sorted(vs_dict.items(), key=lambda item: item[1], reverse=True)}
    print(vs_dict)
# Most important are WAB_OPP, WAB, BARTHAG, BARTHAG_OPP, ADJOE, ADJOE_OPP

# Model Selection


# Training


# Testing

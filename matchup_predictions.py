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
win_perc = pd.Series(team_data['W']/team_data['G'])
team_data['win%'] = win_perc.values
team_data = pd.DataFrame.drop(team_data, 'W',1)
team_data = pd.DataFrame.drop(team_data, 'G',1)

# Importing Matchup Data
game_data = pd.read_csv('2021Matchups.csv', header=0)
game_data['result'] = game_data['result'].str[0]
game_data = game_data[game_data['result']=='W']
game_data['team'] = game_data['team'].str.replace('+',' ')
game_data['winner'] = game_data['team']

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
team_data = pd.read_csv('cbb.csv', header=0,index_col='TEAM')
print(team_data['YEAR'].unique())
win_perc = pd.Series(team_data['W']/team_data['G'])
team_data['win%'] = win_perc.values
team_data = pd.DataFrame.drop(team_data, 'W',1)
team_data = pd.DataFrame.drop(team_data, 'G',1)
team_data_2021 = team_data[team_data['YEAR']=='2021']
print(team_data_2021)

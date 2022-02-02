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
log_model = LogisticRegression(solver='sag',max_iter=10000)
extra_trees_model = ExtraTreesClassifier(n_estimators=100,random_state=1)
random_forest_model = RandomForestClassifier(n_estimators=100,random_state=1)
NN_model = MLPClassifier((512,256,),activation='logistic',solver='adam',max_iter=1000)
nb_model = GaussianNB()
svm_model = SVC(probability=True)

model_list = [nb_model,log_model,svm_model,random_forest_model,extra_trees_model,NN_model]
def ModelSelection():
    p_dict = {}
    model_select_data = pd.DataFrame.drop(match_up_data,['SEED','SEED_OPP'],1).dropna()
    model_select_x = pd.DataFrame.drop(model_select_data,'result',1)
    model_select_y = model_select_data['result']
    for model in model_list:
        model.fit(model_select_x,model_select_y)
        performance = cross_val_score(model,model_select_x,model_select_y, cv=5,scoring='f1_weighted')
        p_dict[model] = np.average(performance)
    print(p_dict)
    best_model = max(p_dict, key=lambda k: p_dict[k])
    print(best_model)

    print(p_dict.values())
# All models have >70% cross_val_score
# Best in order are LogisticRegression (75.3%), SVC (74.6%), MLPClassifier (73.8%), GaussianNB (72.7%), RandomForestClassifier (71.2%), ExtraTreesClassifier (70.8%)

# Training
training_data = pd.DataFrame.drop(match_up_data,['SEED','SEED_OPP'],1).dropna()
training_x = pd.DataFrame.drop(training_data,'result',1)
#training_x = pd.DataFrame.drop(training_x,['WAB_OPP', 'WAB', 'BARTHAG', 'BARTHAG_OPP'],1)
training_y = training_data['result']
log_model.fit(training_x,training_y)

# Testing on 2022 Games
team_data_2022 = pd.read_csv('cbb22.csv', header=0,index_col='Team').dropna()
#win_perc = pd.Series(team_data['W']/team_data['G'])
#team_data['win%'] = win_perc.values
team_data_2022 = pd.DataFrame.drop(team_data_2022, 'Rec',1)
team_data_2022 = pd.DataFrame.drop(team_data_2022, 'G',1)

game_data_2022 = pd.read_csv('2022Matchups.csv', header=0)
game_data_2022['result'] = game_data_2022['result'].str[0]
#game_data = game_data[game_data['result']=='W']
game_data_2022['team'] = game_data_2022['team'].str.replace('+',' ')
game_data_2022 = game_data_2022.set_index('team')
game_data_2022 = game_data_2022[game_data_2022['result']!='+']
game_data_2022 = game_data_2022[game_data_2022['result']!='-']
game_data_2022 = game_data_2022[game_data_2022['result']!='0']
#game_data['winner'] = game_data['team']

match_up_data_2022 = game_data_2022.join(team_data_2022,how="inner")
match_up_data_2022 = match_up_data_2022.reset_index().set_index('opponent')
match_up_data_2022 = match_up_data_2022.join(team_data_2022,how="inner",rsuffix='_OPP')
match_up_data_2022 = match_up_data_2022.rename(columns={'index':'TEAM'})
match_up_data_2022 = match_up_data_2022.reset_index().set_index('TEAM').rename(columns={'index':'opponent'})

test = pd.DataFrame.drop(match_up_data_2022,['Conf','Conf_OPP','date','opponent'],1)
test_x = pd.DataFrame.drop(test,'result',1)
#test_x = pd.DataFrame.drop(test_x,['WAB_OPP', 'WAB', 'Barthag', 'Barthag_OPP'],1)
test_y = test['result']
pred_y = log_model.predict(test_x)
# pred_prob_y = log_model.predict_proba(test_x).to_list()
# for i in range(len(pred_prob_y)):
#     for j in range(len(pred_prob_y[i])):
#         pred_prob_y[i][j] = round(pred_prob_y[i][j],3)
probs = match_up_data_2022[['opponent','result']]
probs['predicted'] = pd.Series(pred_y).values
# probs['predicted probabilities'] = pd.Series(pred_prob_y).values
# probs[['2ND','W','E8','F4','R32','R64','S16']] = probs['predicted probabilities'].str.split(',',expand=True)
pd.set_option("display.max_rows", 68, "display.max_columns", 5)
print('The percent of correct picks are :', round(len(probs[probs['result']==probs['predicted']])/len(probs),5)*100,'%')
print(probs[probs['result']!=probs['predicted']])
probs[probs['result']!=probs['predicted']].to_csv("2022GamePredictions.csv")
pd.reset_option("display.max_rows", "display.max_columns")

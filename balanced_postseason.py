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
from sklearn.tree import *
from sklearn import metrics

#Cleaning data
#change the number of wins to win percentage
data = pd.read_csv('cbb.csv', header=0)
data['POSTSEASON'] = data['POSTSEASON'].fillna(0.0)
data = pd.DataFrame.drop(data, 'SEED',1)
data['POSTSEASON'] = np.where(data['POSTSEASON']!=0,1,data['POSTSEASON'])
win_perc = pd.Series(data['W']/data['G'])
data['win%'] = win_perc.values
data = pd.DataFrame.drop(data, 'W',1)
data = pd.DataFrame.drop(data, 'G',1)
#print(data)

#redo the above but with teams as indices so that VariableSelection() will work:
data_teams_as_index = pd.read_csv('cbb.csv', header=0,index_col='TEAM')
data_teams_as_index['POSTSEASON'] = data_teams_as_index['POSTSEASON'].fillna(0.0)
data_teams_as_index = pd.DataFrame.drop(data_teams_as_index, 'SEED',1)
data_teams_as_index['POSTSEASON'] = np.where(data_teams_as_index['POSTSEASON']!=0,1,data_teams_as_index['POSTSEASON'])
win_perc = pd.Series(data_teams_as_index['W']/data_teams_as_index['G'])
data_teams_as_index['win%'] = win_perc.values
data_teams_as_index = pd.DataFrame.drop(data_teams_as_index, 'W',1)
data_teams_as_index = pd.DataFrame.drop(data_teams_as_index, 'G',1)
#Visualization:
#make a bunch of plots and shit--> plot categorically (by conference)
#conf_avg = data.groupby("CONF").mean()


#Prepare data for creating model
conf_list = []
for var in data['CONF']:
    if 'CONF'+'_'+var not in conf_list:
        conf_list.append('CONF'+'_'+var)
        conf = pd.Series(np.where(data['CONF']==var,1,0))
        data['CONF'+'_'+var] = conf.values

#Variable Selection:
def VariableSelection():
    vs_data = pd.DataFrame.drop(data_teams_as_index,'CONF',1)
    vs_data = pd.DataFrame.drop(vs_data,'YEAR',1)
    array = vs_data.values
    Y = array[:,17]
    Y = Y.astype('int')
    vs_data = pd.DataFrame.drop(vs_data,'POSTSEASON',1)
    array = vs_data.values
    X = array[:,0:51]
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

#It appears WAB, BARTHAG, win%, ADJOE, ADJDE are most important
print("The following features were deemed most important for making our model: WAB, BARTHAG, win%, ADJOE, ADJDE.\n")

#Training:
print('Training data:')
data_final = data.dropna() #do i need this?
data_final = data_final[data_final['YEAR']<2019]
data_final = pd.DataFrame.drop(data_final,'CONF',1)
data_final = pd.DataFrame.drop(data_final,'TEAM',1)
X = data_final.loc[:, data_final.columns != 'POSTSEASON']
y = data_final.loc[:, data_final.columns == 'POSTSEASON'].astype('int')
from imblearn.over_sampling import *
os = SMOTE(random_state=0)
columns = X.columns
os_data_X,os_data_y=os.fit_sample(X, y)
os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
os_data_y= pd.DataFrame(data=os_data_y,columns=['POSTSEASON'])
training_x = os_data_X[['WAB', 'BARTHAG', 'win%', 'ADJOE', 'ADJDE']]
training_y = os_data_y.astype('int')

model = SVC(probability=True)
#model = ExtraTreesClassifier(n_estimators=100)
#model = MLPClassifier((512,512,),activation='logistic',solver='adam',max_iter=10000)
#model = RandomForestClassifier(n_estimators=100)
#model = LogisticRegression(solver='lbfgs')
#model = GradientBoostingClassifier(n_estimators=50,random_state=1)
model.fit(training_x,training_y)

#pred_train_y = model.predict(training_x)
#count_misclassified = (training_y != pred_train_y).sum()
#print('Misclassified training samples: {}'.format(count_misclassified))
#accuracy = metrics.accuracy_score(training_y, pred_train_y)
#print('Accuracy: {:.2f}\n'.format(accuracy))
#print(training_y,pred_train_y)

#Testing:
print('Test data:')
test = data[data['YEAR']==2019]
test = pd.DataFrame.drop(test,'YEAR',1)
test_x = test[['WAB', 'BARTHAG', 'win%', 'ADJOE', 'ADJDE']]
test_y = test['POSTSEASON'].astype('int')


#Results:
pred_prob_y = model.predict_proba(test_x)

#combine pred_prob_y with 1)conf 2)actual classification for each team
final_data = test[['TEAM', 'CONF', 'POSTSEASON']]
final_data = final_data.assign(prob_of_0=pred_prob_y[:,0],prob_of_1=pred_prob_y[:,1])
#print(final_data)

#somehow sort every team by conference, then out of each conference take the ones with
#max estimated probability of being classified as a 1. remove those teams from the pool of all teams, and then take
#the next 36 highest probabilities as the last 36 teams to make playoffs
print('Based on highest probability of making playoffs and the rules that the #1 team from each conference gets an automatic bid:')
sorted_final_data = final_data.sort_values('prob_of_1', ascending=False)
grouped_final_data = sorted_final_data.groupby('CONF')

#these are the automatic bids to playoffs (#1 team from each conference):
automatic_bids = grouped_final_data.head(1)

#now find the rest of the teams...
#remove teams from the pool that already have bids (take the not first value from every conf)
remaining_teams = pd.DataFrame.drop(sorted_final_data, index = automatic_bids.index)

#choose next 36 highest
leftover_bids = remaining_teams.nlargest(36, 'prob_of_1')

#combine all the teams that got bids
pred_playoff_teams = pd.concat([automatic_bids, leftover_bids])
pred_playoff_teams = pred_playoff_teams.assign(prediction_2=1)
print(pred_playoff_teams.sum())
#print('Based on highest probability of making playoffs and the rules that the #1 team from each conference gets an automatic bid, I predict that the following teams will make playoffs:')
#print(pred_playoff_teams)

#add a column to orinal data with my predicted 1 or 0
merge_me = pred_playoff_teams['prediction_2'].astype('int').to_frame()
merged_final_data = final_data.merge(merge_me, how='left', left_index=True, right_index=True)
merged_final_data['prediction_2'] = merged_final_data['prediction_2'].fillna(0.0)
merged_final_data['prediction_2'] = merged_final_data['prediction_2'].astype('int')
#print(merged_final_data)

# #next need to compare these to the predicted teams to get accuracy
my_pred_y = merged_final_data['prediction_2']
actual_y = merged_final_data['POSTSEASON']
count_misclassified = (actual_y != my_pred_y).sum()
print('Misclassified test samples: {}'.format(count_misclassified))
accuracy = metrics.accuracy_score(test_y, my_pred_y)
print('Accuracy: {:.2f}\n'.format(accuracy))

#Just for comparison go thru test data for model not taking into account NCAA rules about automatic bids and whatnot:
print('Not taking into account NCAA rules about automatic bids:')
pred_y = model.predict(test_x)
#model_pred_data = final_data.assign(prediction_1 = pred_y)
count_misclassified = (actual_y != pred_y).sum()
print('Misclassified test samples: {}'.format(count_misclassified))
accuracy = metrics.accuracy_score(test_y, pred_y)
print('Accuracy: {:.2f}\n'.format(accuracy))
#print(model_pred_data)
#print(pred_y.sum()) #hm ok so it only predicted 62 teams making it... is there a way to make it predict 68?

#ok combine it all for some sorta summary
print('After predicting with and without considering automatic bids, here are the final results (POSTSEASON are the actual results, prediction_1 does not consider automatic bids, prediction_2 considers automatic bids):')
data_summary = final_data.assign(prediction_1 = pred_y, prediction_2 = my_pred_y)
data_summary = data_summary[['TEAM', 'POSTSEASON', 'prediction_1', 'prediction_2']]
print(data_summary)

# cnf_matrix = metrics.confusion_matrix(test_y,pred_y)
# class_names=['0','1']
# fig, ax = plt.subplots()
# tick_marks = np.arange(len(class_names))
# sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="coolwarm" ,fmt='g')
# ax.xaxis.set_label_position("top")
# plt.title('Confusion matrix', y=1.1)
# plt.ylabel('Actual label')
# plt.xlabel('Predicted label')
# plt.xticks(tick_marks, class_names)
# plt.yticks(tick_marks, class_names)
# plt.show()

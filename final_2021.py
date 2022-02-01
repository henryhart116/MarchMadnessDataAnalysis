import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import *
from sklearn.multiclass import *
from sklearn.linear_model import *
from sklearn.discriminant_analysis import *
from sklearn.neural_network import *
from sklearn.svm import *
from sklearn.model_selection import *
from sklearn.tree import *
from sklearn.naive_bayes import *
from sklearn.neighbors import *
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
#Importing data
data = pd.read_csv('cbb-2015-2021.csv', header=0,index_col='TEAM')
data = pd.DataFrame.drop(data,'WAS',1)
win_perc = pd.Series(data['W']/data['G'])
data['win%'] = win_perc.values
data = pd.DataFrame.drop(data, 'W',1)
data = pd.DataFrame.drop(data, 'G',1)
data = pd.DataFrame.drop(data,'CONF',1)

#Visualization:
seed_avg = data.groupby("SEED").mean()

def VariableSelection():
    vs_data = data.dropna()
    vs_data = pd.DataFrame.drop(vs_data,'YEAR',1)
    vs_data = pd.DataFrame.drop(vs_data,'SEED',1)
    array = vs_data.values
    Y = array[:,17]
    vs_data = pd.DataFrame.drop(vs_data,'POSTSEASON',1)
    array = vs_data.values
    X = array[:,0:52]
    names = list(vs_data.columns)
    vs_dict = {}
    for i in range(len(names)):
        vs_dict[names[i]] = []
    for i in range(100):
        model = GradientBoostingClassifier(n_estimators=100)
        model.fit(X, Y)
        importance = list(model.feature_importances_)
        for j in range(len(names)):
            vs_dict[names[j]].append(importance[j])
    for i in range(len(names)):
        vs_dict[names[i]] = np.mean(vs_dict[names[i]])
    vs_dict = {k: v for k, v in sorted(vs_dict.items(), key=lambda item: item[1], reverse=True)}
    # create plot
    plt.bar(vs_dict.keys(), vs_dict.values())
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.title('Importance of each feature')
    plt.xticks(fontsize=6)
    # Show graphic
    plt.show()
    # vs_df = pd.DataFrame.from_dict(vs_dict, orient="index")
    # vs_df.to_csv(r'E:\documents\4.2\cx\variables.csv', header = True)
    print(vs_dict)

#Model Selection:
log_model = LogisticRegression(solver='sag',max_iter=10000)
random_forest_model = RandomForestClassifier(n_estimators=100)
LDA_model = LinearDiscriminantAnalysis(solver='lsqr')
nb_model = GaussianNB()
gradient_boosting = GradientBoostingClassifier(n_estimators=1000,learning_rate=.01)
NN_model = MLPClassifier((512,256,),activation='logistic',solver='adam',max_iter=1000)

model_list = [nb_model,LDA_model,log_model,gradient_boosting,random_forest_model,gradient_boosting]
p_dict = {}
model_select_data = data.dropna()
model_select_data = pd.DataFrame.drop(model_select_data,'YEAR',1)
model_select_x = pd.DataFrame.drop(model_select_data,'POSTSEASON',1)
model_select_x = pd.DataFrame.drop(model_select_x,'SEED',1)
#model_select_x = model_select_x[['BARTHAG','3P_D','ORB','WAB','TORD','WAS_SEED_AVG']]
model_select_y = model_select_data['POSTSEASON']
for model in model_list:
    model.fit(model_select_x,model_select_y)
    performance = cross_val_score(model,model_select_x,model_select_y, cv=5,scoring='f1_weighted')
    p_dict[model] = np.average(performance)
print(p_dict)
best_model = max(p_dict, key=lambda k: p_dict[k])
print(best_model)

# print(p_dict.values())
# x_labels = ['LDA','Naive-Bayes','Random Forest','Extra Trees','Logistic Regression','Decision Tree','SVM']
# plt.bar(x_labels, sorted(p_dict.values(), reverse= True))
# plt.xlabel('Model')
# plt.ylabel('Performance')
# plt.title('Performance of each model based on F1 score')
# plt.xticks(fontsize=6)
# # Show graphic
# plt.show()

#Training
training = data.dropna()
training = training[training['YEAR'] != 2021]
training = pd.DataFrame.drop(training,'YEAR',1)
training_x = pd.DataFrame.drop(training,'POSTSEASON',1)
training_x = pd.DataFrame.drop(training_x,'SEED',1)
#training_x = training[['BARTHAG','3P_D','ORB','WAB','TORD','WAS_SEED_AVG']]
training_y = training['POSTSEASON']
scaler.fit(training_x)
training_x = scaler.transform(training_x)

from imblearn.over_sampling import *
from imblearn.pipeline import make_pipeline
os = RandomOverSampler(random_state=0)
pipeline = make_pipeline(os, gradient_boosting)

pipeline.fit(training_x,training_y)

pred_train_y = pipeline.predict(training_x)
count_misclassified = (training_y != pred_train_y).sum()
print('Misclassified samples: {}'.format(count_misclassified))
accuracy = metrics.accuracy_score(training_y, pred_train_y)
f1_score = metrics.f1_score(training_y,pred_train_y,average='weighted')
print('Accuracy: {:.2f}'.format(accuracy))
print('F1 Score: {:.2f}\n'.format(f1_score))

#Testing:
test = data[data['YEAR']==2021]
test = pd.DataFrame.drop(test,'YEAR',1)
test_x = pd.DataFrame.drop(test,'POSTSEASON',1)
test_x = pd.DataFrame.drop(test_x,'SEED',1)

#test_x = test[['BARTHAG','3P_D','ORB','WAB','TORD','WAS_SEED_AVG']]
# test_y = test['POSTSEASON']
test_x = scaler.transform(test_x)
pred_y = pipeline.predict(test_x)

'POSSIBLY PREDIT 2020 TOURNAMENT WITH BOTH SET OF CODES!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'

#Results:
# count_misclassified = (test_y != pred_y).sum()
# print('Misclassified samples: {}'.format(count_misclassified))
# accuracy = metrics.accuracy_score(test_y, pred_y)
# f1_score = metrics.f1_score(test_y,pred_y,average='weighted')
# print('Accuracy: {:.2f}'.format(accuracy))
# print('F1 Score: {:.2f}\n'.format(f1_score))
#for i in range(len(pred_prob_y)):
#    for j in range(len(pred_prob_y[i])):
#        pred_prob_y[i][j] = round(pred_prob_y[i][j],3)
#    pred_prob_y[i].append(pred_y[i])
#print(pred_prob_y)
#results = test[['POSTSEASON','SEED']]
#results['predicted'] = pd.Series(pred_y).values
#print(results)
#cnf_matrix = metrics.confusion_matrix(test_y,pred_y)
#class_names=['2ND','W','E8','F4','R32','R64','R68','S16']
#fig, ax = plt.subplots()
#tick_marks = np.arange(len(class_names))
#sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="coolwarm" ,fmt='g')
#ax.xaxis.set_label_position("top")
#plt.title('Confusion matrix', y=1.1)
#plt.ylabel('Actual label')
#plt.xlabel('Predicted label')
#plt.xticks(tick_marks, class_names)
#plt.yticks(tick_marks, class_names)
#plt.show()

# Using probabilities to predict finish in tournament
pred_prob_y = pipeline.predict_proba(test_x)
final_data = test[['SEED']]
final_data = final_data.assign(prob_Champions=np.around(pred_prob_y[:,1],3),
    prob_2ND=np.around(pred_prob_y[:,1]+pred_prob_y[:,0],3),
    prob_F4=np.around(pred_prob_y[:,1]+pred_prob_y[:,0]+pred_prob_y[:,3],3),
    prob_E8=np.around(pred_prob_y[:,1]+pred_prob_y[:,0]+pred_prob_y[:,3]+pred_prob_y[:,2],3),
    prob_S16=np.around(pred_prob_y[:,1]+pred_prob_y[:,0]+pred_prob_y[:,3]+pred_prob_y[:,2]+pred_prob_y[:,7],3),
    prob_R32=np.around(pred_prob_y[:,1]+pred_prob_y[:,0]+pred_prob_y[:,3]+pred_prob_y[:,2]+pred_prob_y[:,7]+pred_prob_y[:,4],3),
    prob_R64=np.around(pred_prob_y[:,1]+pred_prob_y[:,0]+pred_prob_y[:,3]+pred_prob_y[:,2]+pred_prob_y[:,7]+pred_prob_y[:,4]+pred_prob_y[:,5],3),
    prob_R68=np.around(pred_prob_y[:,1]+pred_prob_y[:,0]+pred_prob_y[:,3]+pred_prob_y[:,2]+pred_prob_y[:,7]+pred_prob_y[:,4]+pred_prob_y[:,5]+pred_prob_y[:,6],3))
#print(final_data)
prob_list = ["prob_Champions","prob_2ND","prob_F4","prob_E8","prob_S16","prob_R32","prob_R64","prob_R68"]
counter = 1
remaining_teams = final_data
pred_finish = pd.DataFrame()
prediction = []
for i in range(8):
    sorted_final_data = remaining_teams.sort_values(prob_list[i],ascending=False)
    if prob_list[i] == "prob_R68":
        counter = 4
    round_winners = sorted_final_data.head(counter)
    remaining_teams = pd.DataFrame.drop(remaining_teams,index=round_winners.index)
    for j in range(counter):
        prediction.append(prob_list[i][5:])
    if prob_list[i] != "prob_Champions":
        counter = counter*2
    pred_finish = pd.concat([pred_finish,round_winners])
pred_finish = pred_finish.assign(prediction=prediction)
pd.options.display.max_rows
pd.set_option('display.max_rows',None)
print(pred_finish)
print(pred_finish[['SEED','prediction']])
#
# my_pred_y = pred_finish['prediction']
# actual_y = pred_finish['POSTSEASON']
# count_misclassified = (actual_y != my_pred_y).sum()
# print('Misclassified test samples: {}'.format(count_misclassified))
# accuracy = metrics.accuracy_score(actual_y, my_pred_y)
# f1_score = metrics.f1_score(actual_y, my_pred_y,average='weighted')
# print('Accuracy: {:.2f}'.format(accuracy))
# print('F1 Score: {:.2f}\n'.format(f1_score))
#
# cnf_matrix = metrics.confusion_matrix(actual_y,my_pred_y)
# class_names=['2ND','W','E8','F4','R32','R64','R68','S16']
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

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

#Importing data
data = pd.read_csv('cbb.csv', header=0,index_col='TEAM')
win_perc = pd.Series(data['W']/data['G'])
data['win%'] = win_perc.values
data = pd.DataFrame.drop(data, 'W',1)
data = pd.DataFrame.drop(data, 'G',1)

# Removing wins due to factoring in postseason wins, if needed convert to win %

#Visualization:
seed_avg = data.groupby("SEED").mean()

#Create Dummy conference variables
conf_list = []
for var in data['CONF']:
    if 'CONF'+'_'+var not in conf_list:
        conf_list.append('CONF'+'_'+var)
        conf = pd.Series(np.where(data['CONF']==var,1,0))
        data['CONF'+'_'+var] = conf.values
data = pd.DataFrame.drop(data,'CONF',1)
#Variabe Selection:
def VariableSelection():
    vs_data = pd.DataFrame.drop(data,'YEAR',1)
    vs_data = vs_data.dropna()
    array = vs_data.values
    Y = array[:,17] #17 index is postseason
    Y = Y.astype('str')
    vs_data = pd.DataFrame.drop(vs_data,'POSTSEASON',1)
    array = vs_data.values
    X = array[:,0:52]
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
# Appears BARTHAG, SEED, WAB, ADJOE, ADJDE, 3P_D, and win% are most important (next few are TOR, TORD, ORB...)

#Balancing:
data_final = data.dropna()
data_final = data_final[data_final['YEAR']<2019]
X = data_final.loc[:, data_final.columns != 'POSTSEASON']
y = data_final.loc[:, data_final.columns == 'POSTSEASON']
from imblearn.over_sampling import *
os = RandomOverSampler(random_state=0)
columns = X.columns
os_data_X,os_data_y=os.fit_sample(X, y)
os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
os_data_y= pd.DataFrame(data=os_data_y,columns=['POSTSEASON'])

#Training:
training_x = pd.DataFrame.drop(os_data_X,'YEAR',1)
#training_x = training[['BARTHAG','SEED','WAB','ADJOE','ADJDE','3P_D','TOR']]
training_y = os_data_y

log_model = LogisticRegression(solver='lbfgs',max_iter=100000)
extra_trees_model = ExtraTreesClassifier(n_estimators=1000)
NN_model = MLPClassifier((512,256,),activation='logistic',solver='adam',max_iter=1000)
random_trees_model = RandomForestClassifier(n_estimators=1000)
svc_model = SVC(probability=True)

log_model.fit(training_x,training_y)

#pred_train_y = log_model.predict(training_x)
#count_misclassified = (training_y != pred_train_y).sum()
#print('Misclassified samples: {}'.format(count_misclassified))
#accuracy = metrics.accuracy_score(training_y, pred_train_y)
#print('Accuracy: {:.2f}'.format(accuracy))

#Testing:
test = data[data['YEAR']==2019].dropna()
#test = data.dropna()
test = pd.DataFrame.drop(test,'YEAR',1)
test_x = pd.DataFrame.drop(test,'POSTSEASON',1)
#test_x = test[['BARTHAG','SEED','WAB','ADJOE','ADJDE','3P_D','TOR']]
test_y = test['POSTSEASON']

pred_y = log_model.predict(test_x)

#pred_prob_y = log_model.predict_proba(test_x).tolist()
#for i in range(len(pred_prob_y)):
#    for j in range(len(pred_prob_y[i])):
#        pred_prob_y[i][j] = round(pred_prob_y[i][j],3)
#    pred_prob_y[i].append(pred_y[i])
#print(pred_prob_y)
'POSSIBLY PREDIT 2020 TOURNAMENT WITH BOTH SET OF CODES!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'

#Results:
count_misclassified = (test_y != pred_y).sum()
print('Misclassified samples: {}'.format(count_misclassified))
accuracy = metrics.accuracy_score(test_y, pred_y)
print('Accuracy: {:.2f}'.format(accuracy))
results = test[['POSTSEASON','SEED']]
results['predicted'] = pd.Series(pred_y).values
#print(results)

performance = cross_val_score(log_model,test_x,pred_y, cv=5,scoring='accuracy')
print(performance)

cnf_matrix = metrics.confusion_matrix(test_y,pred_y)
class_names=['2ND','W','E8','F4','R32','R64','R68','S16']
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="coolwarm" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
plt.show()

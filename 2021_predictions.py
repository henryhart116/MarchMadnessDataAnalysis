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
#for var in data['CONF']:
#    if 'CONF'+'_'+var not in conf_list:
#        conf_list.append('CONF'+'_'+var)
#        conf = pd.Series(np.where(data['CONF']==var,1,0))
#        data['CONF'+'_'+var] = conf.values
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

extra_trees_model.fit(training_x,training_y)

#Testing:
data_2021 = pd.read_csv('2021_data.csv')
data_2021['TEAM'] = data_2021.TEAM.str.split('\\n',expand=True)[0]
data_2021['W'] = data_2021.Rec.str.split('\\n',expand=True)[0].str.split('-',expand=True)[0].astype('float')
data_2021['ADJDE'] = data_2021.AdjDE.str.split('\\n',expand=True)[0]
data_2021['ADJOE'] = data_2021.AdjOE.str.split('\\n',expand=True)[0]
data_2021['BARTHAG'] = data_2021.Barthag.str.split('\\n',expand=True)[0]
data_2021['EFG_O'] = data_2021['EFG%'].str.split('\\n',expand=True)[0]
data_2021['EFG_D'] = data_2021['EFGD%'].str.split('\\n',expand=True)[0]
data_2021['2P_O'] = data_2021['2P%'].str.split('\\n',expand=True)[0]
data_2021['2P_D'] = data_2021['2P%D'].str.split('\\n',expand=True)[0]
data_2021['3P_O'] = data_2021['3P%'].str.split('\\n',expand=True)[0]
data_2021['3P_D'] = data_2021['3P%D'].str.split('\\n',expand=True)[0]
data_2021['ADJ_T'] = data_2021['Adj T.'].str.split('\\n',expand=True)[0]
data_2021 = data_2021.drop(columns=['Rk','Rec','AdjOE','AdjDE','Barthag','EFG%','EFGD%','2P%','2P%D','3P%','3P%D','Adj T.'])
data_2021['TOR'] = data_2021['TOR'].str.split('\\n',expand=True)[0]
data_2021['TORD'] = data_2021['TORD'].str.split('\\n',expand=True)[0]
data_2021['ORB'] = data_2021['ORB'].str.split('\\n',expand=True)[0]
data_2021['DRB'] = data_2021['DRB'].str.split('\\n',expand=True)[0]
data_2021['FTR'] = data_2021['FTR'].str.split('\\n',expand=True)[0]
data_2021['FTRD'] = data_2021['FTRD'].str.split('\\n',expand=True)[0]
data_2021['WAB'] = data_2021['WAB'].str.split('\\n',expand=True)[0]
win_perc = pd.Series(data_2021['W']/data_2021['G'])
data_2021['win%'] = win_perc.values
data_2021 = pd.DataFrame.drop(data_2021, 'W',1)
data_2021 = pd.DataFrame.drop(data_2021, 'G',1)
test = data_2021.set_index('TEAM')
#test = data.dropna()
test_x = pd.DataFrame.drop(test,'YEAR',1)
test_x = pd.DataFrame.drop(test_x,'CONF',1)
#test_x = pd.DataFrame.drop(test,'POSTSEASON',1)
#test_x = test[['BARTHAG','SEED','WAB','ADJOE','ADJDE','3P_D','TOR']]
#test_y = test['POSTSEASON']

pred_y = extra_trees_model.predict(test_x)

pred_prob_y = extra_trees_model.predict_proba(test_x).tolist()
for i in range(len(pred_prob_y)):
    for j in range(len(pred_prob_y[i])):
        pred_prob_y[i][j] = round(pred_prob_y[i][j],3)
probs = test[['SEED']]
probs['predicted'] = pd.Series(pred_y).values
probs['predicted probabilities'] = pd.Series(pred_prob_y).values
# probs[['2ND','W','E8','F4','R32','R64','S16']] = probs['predicted probabilities'].str.split(',',expand=True)
pd.set_option("display.max_rows", 68, "display.max_columns", 5)
print(probs[['predicted probabilities']])
pd.reset_option("display.max_rows", "display.max_columns")
#Results:
#count_misclassified = (test_y != pred_y).sum()
#print('Misclassified samples: {}'.format(count_misclassified))
#accuracy = metrics.accuracy_score(test_y, pred_y)
#print('Accuracy: {:.2f}'.format(accuracy))
#results = test[['POSTSEASON','SEED']]
#results['predicted'] = pd.Series(pred_y).values
#print(results)

# performance = cross_val_score(extra_trees_model,test_x,pred_y, cv=5,scoring='accuracy')
# print(performance)

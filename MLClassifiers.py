
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import gc

from sklearn.metrics import auc, roc_curve, log_loss
from sklearn.preprocessing import LabelEncoder

get_ipython().magic(u'matplotlib inline')


# In[2]:


#load data
dataframe = pd.read_csv('C:/Users/yashas/MLPreProcessData.csv')


# In[3]:


dataframe.shape


# In[5]:


dataframe = dataframe.iloc[0:200000,:]


# In[6]:


dataframe.shape


# In[7]:


#create x and y

x = dataframe.drop(['loan_status'], axis = 1)
y = dataframe['loan_status']


# In[8]:


#split x and y into train test split
#with 80% of data as train data and 20% as test data
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)


# In[9]:


print x_train.shape
print y_train.shape
print x_test.shape
print y_test.shape


# In[10]:


#MLP Classifier
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier()
model = mlp.fit(x_train,y_train) 
pred = model.predict(x_test)

from sklearn.metrics import accuracy_score 
MLPAccuracy = accuracy_score(y_test, pred) 
print("Neural Network Accuracy:",MLPAccuracy)

from sklearn.metrics import f1_score 
mlpF1 = f1_score(y_test, pred, average= 'weighted')
print("F1Score:",mlpF1)


# In[11]:


#Perceptron
from sklearn.linear_model import Perceptron

per = Perceptron()
model = per.fit(x_train,y_train)
pred = model.predict(x_test)

from sklearn.metrics import accuracy_score
PerAccuracy = accuracy_score(y_test, pred)
print("Perceptron Accuracy:",PerAccuracy)

from sklearn.metrics import f1_score
perF1 = f1_score(y_test, pred, average= 'weighted') 
print("F1Score:",perF1)



# In[12]:


#LogisticRegression
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
model=lr.fit(x_train,y_train)
pred = model.predict(x_test)

from sklearn.metrics import accuracy_score
LRAccuracy = accuracy_score(y_test, pred)
print("LR Accuracy:",LRAccuracy)

from sklearn.metrics import f1_score
lrF1 = f1_score(y_test, pred, average= 'weighted') 
print("F1Score:",lrF1)


# In[13]:


#Gaussian NB
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
model = nb.fit(x_train,y_train)
pred = model.predict(x_test)

from sklearn.metrics import accuracy_score
GNBAccuracy = accuracy_score(y_test, pred)
print("Naive base Accuracy:",GNBAccuracy)

from sklearn.metrics import f1_score
nbF1 = f1_score(y_test, pred, average= 'weighted') 
print("F1Score:",nbF1)


# In[14]:


#Decision Tree
from sklearn import tree

dt = tree.DecisionTreeClassifier()
model=dt.fit(x_train,y_train)
pred = model.predict(x_test)

from sklearn.metrics import accuracy_score
DTAccuracy = accuracy_score(y_test, pred)
print("Decision Tree Accuracy:",DTAccuracy)

from sklearn.metrics import f1_score
dtF1 = f1_score(y_test, pred, average= 'weighted') 
print("F1Score:",dtF1)


# In[15]:


#Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
model = rf.fit(x_train,y_train)
pred = model.predict(x_test)

from sklearn.metrics import accuracy_score
RFAccuracy = accuracy_score(y_test, pred)
print("Random Forest Accuracy:",RFAccuracy)

from sklearn.metrics import f1_score
rfF1 = f1_score(y_test, pred, average= 'weighted') 
print("F1Score:",rfF1)


# In[16]:


#ExtraTrees
from sklearn.ensemble import ExtraTreesClassifier

et = ExtraTreesClassifier()
model = et.fit(x_train,y_train)
pred = model.predict(x_test)

from sklearn.metrics import accuracy_score
ETAccuracy = accuracy_score(y_test, pred)
print("Extra Trees Accuracy:",ETAccuracy)

from sklearn.metrics import f1_score
etF1 = f1_score(y_test, pred, average= 'weighted') 
print("F1Score:",etF1)


# In[17]:


#combining Bagging classifier and decision tree
from sklearn import tree
from sklearn.ensemble import BaggingClassifier

com = BaggingClassifier(tree.DecisionTreeClassifier())
model = com.fit(x_train,y_train)
pred = model.predict(x_test)

from sklearn.metrics import accuracy_score
comAccuracy = accuracy_score(y_test, pred)
print("Combined Accuracy:",comAccuracy)

from sklearn.metrics import f1_score
comF1 = f1_score(y_test, pred, average= 'weighted') 
print("F1Score:",comF1)


# In[20]:


#plot and compare scores 

x_axis = np.arange(8)
y_axis = [ mlpF1, perF1, lrF1, nbF1, dtF1, rfF1, etF1, comF1]


plt.bar(x_axis, y_axis, width=0.5)
plt.xticks(x_axis+0.5/10.,('MLP','LR',
           'GNB','DT','RFt','Per','ET','Com'))
           
                          
plt.ylabel('F1 Score')

plt.show()


# In[ ]:


#desision tree; random forest; combined


# In[24]:


#Cross_validation - 5 folds
from sklearn.cross_validation import cross_val_score

dt_score = cross_val_score(dt, x_train, y_train, cv=5).mean()
rf_score = cross_val_score(rf, x_train, y_train, cv=5).mean()
com_score = cross_val_score(com, x_train, y_train, cv=5).mean()

print('Decision Tree cross validation score', dt_score)
print('Random Forest cross validation score', rf_score)
print('Combined cross validation score', com_score)


# In[25]:


#Plot graph

x_axis = np.arange(3)
y_axis = [dt_score, rf_score, com_score]

plt.bar(x_axis, y_axis, width=0.5)
plt.xticks(x_axis+0.5/10.,('DT','RF','Comb'))
plt.ylabel('Cross-Val Accuracy-5')

plt.show()


# In[27]:


#Cross_validation - 10 folds
from sklearn.cross_validation import cross_val_score

dt_sc = cross_val_score(dt, x_train, y_train, cv=10).mean()
rf_sc = cross_val_score(rf, x_train, y_train, cv=10).mean()
com_sc = cross_val_score(com, x_train, y_train, cv=10).mean()

print('Decision Tree validation score', dt_sc)
print('Random Forest cross validation score', rf_sc)
print('Combined cross validation score', com_sc)


# In[29]:


#Plot graph

x_axis = np.arange(3)
y_axis = [dt_sc, rf_sc, com_sc]

plt.bar(x_axis, y_axis, width=0.5)
plt.xticks(x_axis+0.5/10.,('DT','RF','Comb'))
plt.ylabel('Cross-Val Accuracy-5')

plt.show()


# In[ ]:


#Tweetking the top classifier


# In[37]:


#Decision Tree
from sklearn.grid_search import GridSearchCV
from sklearn import tree

parameters ={
            'min_samples_split':[5,10],
            'min_samples_leaf':[3,5],
            'max_depth':[10,15],
            'max_leaf_nodes':[20,40]
}

dtg = tree.DecisionTreeClassifier()
grid_r = GridSearchCV(dtg,parameters,cv=10,scoring='accuracy')
grid_r.fit(x_train, y_train)
print('Decision Tree:')
print(grid_r.best_score_)


# In[38]:


#Random Forest
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

parameters ={
            'min_samples_split':[5,10],
            'min_samples_leaf':[3,5],
            'max_depth':[10,15],
            'max_leaf_nodes':[20,40]
}

rfg = RandomForestClassifier()
grid_rf = GridSearchCV(rfg,parameters,cv=10,scoring='accuracy')
grid_rf.fit(x_train, y_train)
print('Random Forest:')
print(grid_rf.best_score_)


# In[39]:


#Combined classifier
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import BaggingClassifier
from sklearn import tree

parameters ={
            'base_estimator__max_depth':[2,5],
            'max_samples':[0.1,0.5]
}

comg = BaggingClassifier(tree.DecisionTreeClassifier(),n_estimators=100,max_features=0.5)
grid_c = GridSearchCV(comg,parameters,scoring='accuracy')
grid_c.fit(x_train, y_train)
print('Combined Classifier- Bagging & Decision Tree:')
print(grid_c.best_score_)


# In[41]:


#Plot graph

x_axis = np.arange(3)
y_axis = [grid_r.best_score_, grid_rf.best_score_, grid_c.best_score_]

plt.bar(x_axis, y_axis, width=0.5)
plt.xticks(x_axis+0.5/10.,('DT','RF','Com'))
plt.ylabel('FineTunedAcuuracy')

plt.show()


# In[42]:


from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

print('Decision Tree Classifier Result:')
pred_DT = grid_r.predict(x_test)
dta = metrics.f1_score(y_test, pred_DT, average= 'weighted')
print('Prediction Accuracy')
print(dta)
print('Report')
print(classification_report(y_test,pred_DT))
print('Confusion Matrix')
print(confusion_matrix(y_test, pred_DT))

print('********************************************************************')
print('Random Forest Result:')
pred_RF = grid_rf.predict(x_test)
rfa = metrics.f1_score(y_test, pred_RF, average= 'weighted')
print('Prediction Accuracy')
print(rfa)
print('Report')
print(classification_report(y_test,pred_RF))
print('Confusion Matrix')
print(confusion_matrix(y_test, pred_RF))
print('********************************************************************')
print('Combined Classifiers Result')
pred_Com = grid_c.predict(x_test)
coma = metrics.f1_score(y_test, pred_Com, average= 'weighted')
print('Prediction Accuracy')
print(coma)
print('Report')
print(classification_report(y_test,pred_Com))
print('Confusion Matrix')
print(confusion_matrix(y_test, pred_Com))


# In[44]:


#Plot graph

x_axis = np.arange(3)
y_axis = [dta, rfa, coma]

plt.bar(x_axis, y_axis, width=0.5)
plt.xticks(x_axis+0.5/10.,('DT','RF','Com'))
plt.ylabel('FT Prediction Accuracy')

plt.show()


# In[55]:


#Deriving the most important features affecting classification

# fit Random Forest model to the cross-validation data
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier()
forest.fit(x_train, y_train)
importances = forest.feature_importances_

# make importance relative to the max importance
feature_importance = 100.0 * (importances / importances.max())
sorted_idx = np.argsort(feature_importance)
feature_names = list(x_test.columns.values)
feature_names_sort = [feature_names[indice] for indice in sorted_idx]
pos = np.arange(sorted_idx.shape[0]) + .5
print 'Top 10 features are: '
for feature in feature_names_sort[::-1][:6]:
    print feature


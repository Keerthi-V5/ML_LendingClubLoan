
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


# In[4]:


dataframe = dataframe.iloc[0:200000,:]


# In[5]:


dataframe.shape


# In[6]:


#create x and y

x = dataframe.drop(['loan_status'], axis = 1)
y = dataframe['loan_status']


# In[7]:


#split x and y into train test split
#with 80% of data as train data and 20% as test data
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)


# In[8]:


print x_train.shape
print y_train.shape
print x_test.shape
print y_test.shape


# In[9]:


#Applyign PCA/SVD to reduce the dimenisons of preprocessed dat

#from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA

#dr = TruncatedSVD()
dr=PCA()

dr.fit(x_train)
dr.fit(x_test)
x_train = dr.transform(x_train)
x_test = dr.transform(x_test)


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


# In[18]:


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


# In[19]:


#plot and compare scores 

x_axis = np.arange(8)
y_axis = [ mlpF1, perF1, lrF1, nbF1, dtF1, rfF1, etF1, comF1]


plt.bar(x_axis, y_axis, width=0.5)
plt.xticks(x_axis+0.5/10.,('MLP','LR',
           'GNB','DT','RFt','Per','ET','Com'))
           
                          
plt.ylabel('F1 Score')

plt.show()


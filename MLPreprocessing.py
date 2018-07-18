
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
dataframe = pd.read_csv('C:/Users/yashas/lending-club-loan-data/loan.csv')


# In[ ]:


#Analysis
#Exploration


# In[3]:


print(dataframe.shape)
#dataframe.head(5)


# In[4]:


#create concise table to understand the data

data_type = dataframe.dtypes.values
missing_count = dataframe.isnull().sum().values

unique_count = []
for attr in dataframe.columns:
    unique_count.append(dataframe[attr].unique().shape[0])
    
info ={'Attributes': dataframe.columns,
       'Attribute_Type': data_type,
       'MissingValue_Count':missing_count,
       'UniqueValue_Count': unique_count
       }
col_names = {'Attributes','Attribute_Type','MissingValue_Count','UniqueValue_Count'}

info_tab = pd.DataFrame(info,columns = col_names)
info_tab


# In[ ]:


#Preprocessing


# In[ ]:


#Target Variable - loan_status is worked on to have only two categories as 
#we are interested in Default classes
#With this we consider this as a binary classification problem


# In[5]:


print(dataframe['loan_status'].unique())


# In[ ]:


#dataframe['loan_status']
#dataframe = dataframe.drop(dataframe[dataframe['loan_status'] == 'Issued'].index)
#dataframe.shape
#two rows have been dropped

#887379 - 878919 = 8460
#rows are dropped
#lets not do it.. this will be fine by taking it as 1


# In[6]:


#Processing the target variable loan_status
loan_map = {'Current':0,
           'Fully Paid':0,
           'In Grace Period':0,
           'Does not meet the credit policy. Status:Fully Paid': 0,
           'Charged Off':1,
           'Default':1,
           'Does not meet the credit policy. Status:Charged Off':1,            
           'Late (31-120 days)':1,
           'Late (16-30 days)':1,
            'Issued':0
           }

dataframe['loan_status'] = dataframe['loan_status'].apply(lambda x:loan_map[x])


# In[ ]:


#indicates imbalanced data
#887379 is total 
#61176 - default/class of interest
#826203 - negative category


# In[7]:


default_ratio = np.round(len(dataframe[dataframe['loan_status'] == 1]) )
nondefault_ratio = np.round(len(dataframe[dataframe['loan_status'] == 0]))

print(default_ratio, '\t', nondefault_ratio)


# In[ ]:


#Working on other set of attributes
#This is decided by number of iterations done for preprocessing and using 
#the information drawn from concise table above


# In[8]:


#Dropping features with too many missing values

drop_features = ['id', 'member_id','url', 'desc',
                    'mths_since_last_delinq',
                    'mths_since_last_record',
                    'mths_since_last_major_derog',
                    'annual_inc_joint',
                    'dti_joint',
                    'verification_status_joint',
                    'open_acc_6m',
                    'open_il_6m',
                    'open_il_12m',
                    'open_il_24m',
                    'mths_since_rcnt_il',
                    'total_bal_il',
                    'il_util',
                    'open_rv_12m',
                    'open_rv_24m',
                    'max_bal_bc',
                    'all_util',
                    'inq_fi',
                    'total_cu_tl',
                   'inq_last_12m']

dataframe = dataframe.drop(labels = drop_features, axis = 1)


# In[9]:


#drop meaningless features

ml_features = ['emp_title',
               'issue_d',
               'last_pymnt_d',
               'next_pymnt_d',
               'zip_code',
               'title',
               'grade',
               'earliest_cr_line',
               'last_credit_pull_d',
               'policy_code']

dataframe = dataframe.drop(labels = ml_features, axis = 1)


# In[10]:


#drop redundant features

red = ['funded_amnt',
      'funded_amnt_inv',
      'installment',
      'sub_grade',
      'total_pymnt_inv',
      'total_rec_prncp',
      'out_prncp_inv',
      'collection_recovery_fee']

dataframe = dataframe.drop(red, axis = 1)


# In[ ]:


#dropping in total 42 attributes out of 74 attributes


# In[11]:


print(dataframe.shape)


# In[12]:


#create concise table to understand the data after modification

data_type = dataframe.dtypes.values
missing_count = dataframe.isnull().sum().values

unique_count = []
for attr in dataframe.columns:
    unique_count.append(dataframe[attr].unique().shape[0])
    
info ={'Attributes': dataframe.columns,
       'Attribute_Type': data_type,
       'MissingValue_Count':missing_count,
       'UniqueValue_Count': unique_count
       }
col_names = {'Attributes','Attribute_Type','MissingValue_Count','UniqueValue_Count'}

info_tab = pd.DataFrame(info,columns = col_names)
info_tab


# In[ ]:


#Feature Transformation


# In[13]:


#Tranforming numerical attributes to categorical
ntoc_features = ['total_rec_late_fee',
                'recoveries',                     
                'tot_coll_amt',
                'collections_12_mths_ex_med',
                'acc_now_delinq',
                'out_prncp',
                ]

for attr in ntoc_features:
    dataframe[attr] = (dataframe[attr]>0).astype(str)


# In[ ]:


#dataframe['recoveries']


# In[ ]:


#Working on emp_length attribute


# In[14]:


#Filling the missing va;ues of this attribute with the mode
#finding mode
mode_value = dataframe['emp_length'].mode().values[0]
print(mode_value)


# In[15]:


#filling the missing values with evaluated mode value
dataframe.loc[dataframe.emp_length == 'n/a', 'emp_length'] = '10+ years'


# In[ ]:


#transforming this attribute from categorical to numerical


# In[16]:


#Transforming by manual encoding
emplen_map = {
    '< 1 year':0,
    '1 year':1,
    '2 years':2,
    '3 years':3,
    '4 years':4,
    '5 years':5,
    '6 years':6,
    '7 years':7,
    '8 years':8,
    '9 years':9,
    '10+ years':10 }
dataframe['emp_length'] = dataframe['emp_length'].apply(lambda x:emplen_map[x])


# In[ ]:


#dataframe['emp_length']


# In[ ]:


#Working on missing values present in the entire dataset


# In[17]:


#separate num and cat features
numerical_feature = dataframe.select_dtypes(exclude = ['object']).columns.drop('loan_status')
categorical_feature = dataframe.select_dtypes(include = ['object']).columns

print('Numerical_count:',len(numerical_feature))
print('Categorical_count:',len(categorical_feature))


# In[18]:


#fill numerical with median
medians = dataframe[numerical_feature].median(axis=0, skipna = True)
dataframe[numerical_feature] = dataframe[numerical_feature].fillna(value = medians)


# In[19]:


print(dataframe.shape)


# In[ ]:


#printing the numerical and categorical attributes


# In[20]:


numerical_feature


# In[22]:


categorical_feature


# In[21]:


#one hot coding equivalent dummy for converting categorical 
#attributes to numerical


# In[23]:


for i in categorical_feature:
    dataframe = pd.get_dummies(dataframe, prefix=[i], columns = [i])


# In[24]:


dataframe.shape


# In[25]:


#create concise table to understand the data after modification

data_type = dataframe.dtypes.values
missing_count = dataframe.isnull().sum().values

unique_count = []
for attr in dataframe.columns:
    unique_count.append(dataframe[attr].unique().shape[0])
    
info ={'Attributes': dataframe.columns,
       'Attribute_Type': data_type,
       'MissingValue_Count':missing_count,
       'UniqueValue_Count': unique_count
       }
col_names = {'Attributes','Attribute_Type','MissingValue_Count','UniqueValue_Count'}

info_tab = pd.DataFrame(info,columns = col_names)
info_tab


# In[ ]:


#All the attributes have float or int as their type which means entire data 
#is in numerical form


# In[26]:


dataframe.to_csv('MLPreProcessData.csv')


# In[ ]:


#The data is netirely preprocessed
#We can now start workign on the preprocessed data


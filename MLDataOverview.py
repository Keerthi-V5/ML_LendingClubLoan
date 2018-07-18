
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as pt
get_ipython().magic(u'matplotlib inline')


# In[2]:


#load data
dataframe = pd.read_csv('C:/Users/yashas/lending-club-loan-data/loan.csv')


# In[3]:


#Number of rows and columns of dataframe
dataframe.shape


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
       'UniqueValue_Count': unique_count,
      
       }
col_names = {'Attributes','Attribute_Type','MissingValue_Count','UniqueValue_Count'}

info_tab = pd.DataFrame(info,columns = col_names)
info_tab


# In[6]:


print(dataframe['loan_status'].unique())


# In[7]:


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


# In[8]:


#Representing target variable using bar chart
dataframe.groupby('loan_status').size().plot(kind='bar')
pt.ylabel('Count')


# In[ ]:


#shows imbalance in the dataset


# In[13]:


bar_graph = pt.figure(figsize=(20,20))

axis1 = bar_graph.add_subplot(221)
axis1 = dataframe.groupby('home_ownership').size().plot(kind='bar')
pt.xlabel('home_ownership', fontsize=15)
pt.ylabel('Count', fontsize=10)


# In[12]:


bar_graph = pt.figure(figsize=(20,20))

axis1 = bar_graph.add_subplot(221)
axis1 = dataframe.groupby('sub_grade').size().plot(kind='bar')
pt.xlabel('subgrade', fontsize=15)
pt.ylabel('Count', fontsize=10)


# In[ ]:



bar_graph = pt.figure(figsize=(20,20))

axis1 = bar_graph.add_subplot(221)
axis1 = dataframe.groupby('emp_title').size().plot(kind='bar')
pt.xlabel('emp_title', fontsize=15)
pt.ylabel('Count', fontsize=10)


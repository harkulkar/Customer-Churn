#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv("Telco-Customer-Churn.csv")


# In[3]:


df.describe()


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


df.columns.values


# In[7]:


df.isnull().sum()


# In[8]:


df['TotalCharges'] = pd.to_numeric(df.TotalCharges, errors = 'coerce')


# In[9]:


df.isnull().sum()


# In[10]:


df['TotalCharges'].fillna(df['TotalCharges'].mean(), inplace=True)


# In[11]:


df.isnull().sum()


# In[12]:


numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
df[numerical_cols].describe().T


# In[13]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# In[14]:


x=df[['SeniorCitizen','tenure' ,'MonthlyCharges','TotalCharges']]
y=df['Churn']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
Model_random_foreset=RandomForestClassifier()


# In[15]:


Model_random_foreset.fit(X_train,y_train)


# In[16]:


pred_random=Model_random_foreset.predict(X_test)
print(pred_random)


# In[17]:


##using  SVM

from sklearn.svm import SVC


# In[18]:


svm_model=SVC(kernel='rbf')
svm_model.fit(X_train,y_train)


# In[19]:


pred_svm=svm_model.predict(X_test)
print(pred_svm)


# In[20]:


##using  decisiton tree
 
from sklearn.tree import DecisionTreeClassifier

dt_model=DecisionTreeClassifier(random_state=85)

dt_model.fit(X_train,y_train)

pred_dt=dt_model.predict(X_test)


# In[21]:


pred_dt


# In[24]:


#using logistic

from sklearn.preprocessing import MinMaxScaler

# Now you can use MinMaxScaler without encountering a NameError
scaler = MinMaxScaler()
X_new_scaled = scaler.fit_transform(X_new)

from sklearn.linear_model import LogisticRegression

LR_model=LogisticRegression(random_state=85)

LR_model.fit(X_train,y_train)

pred_LR=LR_model.predict(X_test)

print(pred_LR)


# In[ ]:





# In[25]:


# Example input data:
new_data = pd.DataFrame({
    'SeniorCitizen': [0, 1],
    'tenure': [1, 30],
    'MonthlyCharges': [29.85, 80.0],
    'TotalCharges': [29.85, 2400.0]
})

# Split the input data into training and testing sets
# Note: You should use the same scaler that you used during training
X_new = new_data  # No need to scale for Random Forest
X_new_scaled = MinMaxScaler().fit_transform(X_new)  # Scale for SVM

# Make predictions using the Random Forest model
pred_random = Model_random_foreset.predict(X_new)
print("Random Forest Predictions:", pred_random)

# Make predictions using the SVM model
pred_svm = svm_model.predict(X_new_scaled)
print("SVM Predictions:", pred_svm)


# In[ ]:





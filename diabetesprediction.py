#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score


# In[3]:


data = pd.read_csv("C:/Users/91720/OneDrive/Desktop/prodigy/diabetes_prediction_dataset.csv")


# In[4]:


data.head()


# In[5]:


data.shape


# In[6]:


X = data.drop('diabetes', axis=1)
y = data['diabetes']


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[16]:


from sklearn.preprocessing import LabelEncoder


# In[17]:




categorical_cols = X_train.select_dtypes(include=['object']).columns
for col in categorical_cols:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
    X_test[col] = le.transform(X_test[col])



# In[18]:


X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[19]:


models = {
    "Random Forest": RandomForestClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeClassifier()
}


# In[21]:


for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    if name != "Linear Regression":
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)
        print(f"Model: {name}")
        print(f"Accuracy: {accuracy}")
        print(f"ROC AUC Score: {roc_auc}")
        print(classification_report(y_test, y_pred))
    else:
        y_pred = model.predict(X_test)
      
        print(f"Model: {name}")


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# ### Flight Price Prediction

# In[1]:


import streamlit as st
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[307]:


df=pd.read_excel("C:/Users/sai sharath reddy/Downloads/Data_Train.xlsx")
df


# In[308]:


df.head()


# In[309]:


df.tail()


# In[310]:


df.columns


# In[311]:


df.describe()


# In[312]:


df.isnull().sum()


# In[313]:


df.dropna(inplace=True)


# In[314]:


df.isnull().sum()


# In[315]:


#convert date of jounery into datetime format
df['Date_of_Journey']=pd.to_datetime(df['Date_of_Journey'],format='%d/%m/%Y')


# In[316]:


df.dtypes


# In[317]:


#extract day,month and year from date of journey using split function
df['Day']=df['Date_of_Journey'].dt.date
df['Month']=df['Date_of_Journey'].dt.month
df['Day']=df['Date_of_Journey'].dt.year


# In[318]:


df.drop('Date_of_Journey',axis=1,inplace=True)


# In[319]:


df.head()


# In[320]:


df['Dep_hour']=df['Dep_Time'].str.split(':').str[0].astype(int)
df['Dep_min']=df['Dep_Time'].str.split(':').str[0].astype(int)


# In[321]:


df.drop('Dep_Time',axis=1,inplace=True)


# In[322]:


#first exytact the arrival time and remove the day and month from the arrival time column
df['Arrival_Time']=df['Arrival_Time'].str.split(' ').str[0]


# In[323]:


df.head()


# In[324]:


# extract the arrival hour and arrival min from the arrival time column
df['Arrival_hour'] = df['Arrival_Time'].str.split(':').str[0].astype(int)
df['Arrival_min'] = df['Arrival_Time'].str.split(':').str[1].astype(int)


# In[325]:


df.head()


# In[326]:


#drop the arrival time column
df.drop('Arrival_Time',axis=1,inplace=True)


# In[327]:


df.head()


# In[328]:


#extract the departure hour and departure min from dep column
df['Duration_hour']=df['Duration'].str.split(' ').str[0]
df['Duration_min']=df['Duration'].str.split(' ').str[0]


# In[329]:


df.head(3)


# In[330]:


df.drop('Additional_Info',axis=1,inplace=True)


# In[331]:


# extract the duration hour and duration min from the duration column
df['Duration_hour'] = df['Duration'].str.split(' ').str[0]
df['Duration_min'] = df['Duration'].str.split(' ').str[1]


# In[332]:


# remove the 'h' and 'm' from the duration hour and duration min columns
df['Duration_hour'] = df['Duration_hour'].astype(str).str.extract(r'(\d+)', expand=False).fillna(0).astype(int)
df['Duration_min'] = df['Duration_min'].astype(str).str.extract(r'(\d+)', expand=False).fillna(0).astype(int)


# In[333]:


df.head()


# In[334]:


#clean the total stops column
df['Total_Stops'].value_counts()


# In[335]:


#convert the non stop value into 0 stop value
df['Total_Stops']=df['Total_Stops'].replace("non-stop","0 stop")


# In[336]:


df['Total_Stops'].value_counts()


# In[337]:


df.head()


# In[338]:


df['Total_Stops']=df['Total_Stops'].replace("0 stop","0")
df['Total_Stops']=df['Total_Stops'].replace("1 stop","1")
df['Total_Stops']=df['Total_Stops'].replace("2 stops","2")
df['Total_Stops']=df['Total_Stops'].replace("3 stops","3")
df['Total_Stops']=df['Total_Stops'].replace("4 stops","4")


# In[339]:


df.head()


# In[340]:


df['Route'].unique()


# In[341]:


df['Route_1']=df['Route'].str.split('→').str[0].str.strip()
df['Route_2']=df['Route'].str.split('→').str[1].str.strip()
df['Route_3']=df['Route'].str.split('→').str[2].str.strip()
df['Route_4']=df['Route'].str.split('→').str[3].str.strip()
df['Route_5']=df['Route'].str.split('→').str[4].str.strip()


# In[342]:


df.head()


# In[343]:


df['Route_1']=df['Route'].fillna('None')
df['Route_2']=df['Route'].fillna('None')
df['Route_3']=df['Route'].fillna('None')
df['Route_4']=df['Route'].fillna('None')
df['Route_5']=df['Route'].fillna('None')


# In[344]:


df.drop('Route',axis=1,inplace=True)


# In[345]:


plt.figure(figsize=(5,6))
sns.histplot(df['Price'],kde=True)
plt.title('Distribution of Flight Prices')
plt.show()


# In[346]:


#plot the count of airlines
plt.figure(figsize=(5,6))
sns.countplot(x='Airline',data=df,color='red')
plt.title('count of airlines')
plt.xticks(rotation=90)
plt.show()


# In[347]:


#plot the count of airlines in horizontal
plt.figure(figsize=(5,6))
sns.countplot(y=df['Airline'],order=df['Airline'].value_counts().index)
plt.title('count of airlines')
plt.show()


# In[348]:


#all categorical columns convert into label encoding
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
categorical_cols=df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col]=le.fit_transform(df[col])


# In[349]:


df.head()


# In[350]:


#check central tendency of the price column
print('Mean:',df['Price'].mean())
print('Median:',df['Price'].median())
print('Mode:',df['Price'].mode().iloc[0])


# In[351]:


#splitting data into 2 parts- x & y
x=df.drop('Price',axis=1)
y=df['Price']


# In[352]:


#splitting data into train and test 
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[353]:


#train the model
from sklearn.linear_model import LinearRegression
#load the model
model=LinearRegression()


# In[354]:


#fit the model
train_model=model.fit(X_train,Y_train)


# In[363]:


#make predictions
y_pred_le=model.predict(X_test)
y_pred_le


# In[364]:


#elvaute the model
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
mse=mean_squared_error(Y_test,y_pred_le)
msa=mean_absolute_error(Y_test,y_pred_le)
r2=r2_score(Y_test,y_pred_le)
print("Mean Squared Error for Linear Regression:",mse)
print("Mean Absolute Error for Linear Regression:",msa)
print("R-squared Score for Linear Regression:",r2)


# In[365]:


#model 2 : Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor
#load the model
model=DecisionTreeRegressor()
train_model=model.fit(X_train,Y_train)
#make predictions
y_pred_dt=model.predict(X_test)
y_pred_dt
#elvaute the model
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
mse=mean_squared_error(Y_test,y_pred_dt)
msa=mean_absolute_error(Y_test,y_pred_dt)
r2=r2_score(Y_test,y_pred_dt)
print("Mean Squared Error for Decision Tree Regression:",mse)
print("Mean Absolute Error for Decision Tree Regression:",msa)
print("R-squared Score for Decision Tree Regression:",r2)


# In[367]:


#model 3: SVR
from sklearn.svm import SVR
#load the model
model=SVR(kernel='linear',C=1.0,epsilon=0.1)
train_model=model.fit(X_train,Y_train)
#make predictions
y_pred_svr=model.predict(X_test)
y_pred_svr
#elvaute the model
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
mse=mean_squared_error(Y_test,y_pred_svr)
msa=mean_absolute_error(Y_test,y_pred_svr)
r2=r2_score(Y_test,y_pred_svr)
print("Mean Squared Error for SVR:",mse)
print("Mean Absolute Error for SVR:",msa)
print("R-squared Score for SVR:",r2)


# In[368]:


#model 4: KNN
from sklearn.neighbors import KNeighborsRegressor
#load the model
model=KNeighborsRegressor(n_neighbors=5)
train_model=model.fit(X_train,Y_train)
#make predictions
y_pred_knn=model.predict(X_test)
y_pred_knn
#elvaute the model
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
mse=mean_squared_error(Y_test,y_pred_knn)
msa=mean_absolute_error(Y_test,y_pred_knn)
r2=r2_score(Y_test,y_pred_knn)
print("Mean Squared Error for KNN:",mse)
print("Mean Absolute Error for KNN:",msa)
print("R-squared Score for KNN:",r2)


# In[376]:


from sklearn.ensemble import RandomForestRegressor
# model
model = RandomForestRegressor(n_estimators=100,random_state=9)
# train
model.fit(X_train, Y_train)
# predict
y_pred_rf = model.predict(X_test)
#elvaute the model
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
mse=mean_squared_error(Y_test,y_pred_rf)
msa=mean_absolute_error(Y_test,y_pred_rf)
r2=r2_score(Y_test,y_pred_rf)
print("Mean Squared Error for Random Forest:",mse)
print("Mean Absolute Error for Random Forest:",msa)
print("R-squared Score for Random Forest:",r2)


# In[377]:


#make a dataframe to compare the results of all the models
model_comparison = pd.DataFrame({'Model': ['Linear Regression', 'Decision Tree Regressor', 'SVR', 'KNN Regressor', 'Random Forest Regressor'],
                                 'MAE': [mean_absolute_error(Y_test, y_pred), mean_absolute_error(Y_test, y_pred_dt), mean_absolute_error(Y_test, y_pred_svr), mean_absolute_error(Y_test, y_pred_knn), mean_absolute_error(Y_test, y_pred_rf)],
                                 'MSE': [mean_squared_error(Y_test, y_pred), mean_squared_error(Y_test, y_pred_dt), mean_squared_error(Y_test, y_pred_svr), mean_squared_error(Y_test, y_pred_knn), mean_squared_error(Y_test, y_pred_rf)],
                                 'R2 Score': [r2_score(Y_test, y_pred), r2_score(Y_test, y_pred_dt), r2_score(Y_test, y_pred_svr), r2_score(Y_test, y_pred_knn), r2_score(Y_test, y_pred_rf)]})




# In[375]:


model_comparison


# In[ ]:





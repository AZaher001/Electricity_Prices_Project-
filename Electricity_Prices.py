#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


sns.set(rc={'figure.figsize': [10, 10]}, font_scale=1.3) 


# In[2]:


#Read Data
df = pd.read_csv('electricity_prices.csv',low_memory=False)
df


# In[3]:


#dataFrame Information
df.info()


# # Data Analyst

# In[4]:


# Data Convertion to Numircal
numircal = ['ForecastWindProduction','SystemLoadEA','SMPEA','ORKTemperature','ORKWindspeed',
'CO2Intensity','ActualWindProduction','SystemLoadEP2','SMPEP2']
for col in numircal:
    df[col] = pd.to_numeric(df[col] , errors='coerce')


# In[5]:


#data convertion to datetime
df['DateTime'] = pd.to_datetime(df['DateTime'] ,format='%d/%m/%Y %H:%M' ,errors='coerce')


# In[6]:


df['Holiday'].unique()


# In[7]:


df.describe()


# In[8]:


df.groupby('Year').describe()['SMPEP2'].transpose()


# In[9]:


df.groupby('Year').describe()['ForecastWindProduction'].transpose()


# In[10]:


df.corr()[['ForecastWindProduction','SystemLoadEA','SMPEA','ORKTemperature','ORKWindspeed',
'CO2Intensity','ActualWindProduction','SystemLoadEP2','SMPEP2']]


# In[11]:


sns.heatmap(df.corr()[['ForecastWindProduction','SystemLoadEA','SMPEA','ORKTemperature','ORKWindspeed',
'CO2Intensity','ActualWindProduction','SystemLoadEP2','SMPEP2']], annot=True)


# In[12]:


sns.heatmap(df.corr()[['ForecastWindProduction','ActualWindProduction']], annot=True)


# In[13]:


sns.heatmap(df.corr()[['SystemLoadEA','SystemLoadEP2']], annot=True)


# In[14]:


df.groupby('HolidayFlag').describe()['SMPEP2'].transpose()


# In[15]:


df[df['ORKTemperature'] == df['ORKTemperature'].max()]


# In[16]:


df.sort_values(by='ForecastWindProduction',ascending=False,).head()


# In[17]:


df.groupby('Month').describe()['SMPEP2'].transpose()


# In[18]:


df.groupby('Year').describe()['ActualWindProduction'].transpose()


# In[19]:


df.groupby('Year').describe()['ForecastWindProduction'].transpose()


# In[20]:


df[df['ForecastWindProduction'] == df['ActualWindProduction']].head()


# # Data Visualization

# In[21]:


data_Numeric = ['HolidayFlag','DayOfWeek','WeekOfYear','Day','Month','Year','PeriodOfDay','ForecastWindProduction',
'ActualWindProduction','SystemLoadEA','SMPEA','ORKTemperature','ORKWindspeed','CO2Intensity','SystemLoadEP2','SystemLoadEA','SMPEP2']
for col in data_Numeric:
    sns.displot(df[col],height=8,bins=30)


# In[22]:


sns.jointplot(data = df,x = 'SMPEP2' ,y ='Day',color='r')


# In[23]:


sns.jointplot(data = df,x = 'SMPEP2' ,y ='SystemLoadEA',color='m')


# In[24]:


sns.jointplot(data = df,x = 'SMPEP2' ,y ='SMPEA',color='y')


# In[25]:


sns.jointplot(data = df,x = 'SMPEP2' ,y ='ForecastWindProduction',color='g')


# In[26]:


sns.jointplot(data = df,x = 'ActualWindProduction' ,y ='ForecastWindProduction',color='r')


# In[27]:


sns.jointplot(data = df,x = 'SystemLoadEP2' ,y ='SystemLoadEA',color='r')


# In[28]:


sns.jointplot(data = df,x = 'SMPEP2' ,y ='ORKWindspeed',color='m')


# In[29]:


sns.jointplot(data = df,x = 'SMPEP2' ,y ='ORKTemperature',color='m')


# In[30]:


sns.countplot(y='Holiday', data=df, palette='flare')


# In[31]:


sns.countplot(x='Day', data=df, palette='flare')


# In[32]:


sns.countplot(x='DayOfWeek', data=df, palette='flare')


# In[33]:


sns.countplot(x='Month', data=df, palette='flare')


# In[34]:


sns.countplot(x='Year', data=df, palette='flare')


# In[35]:


sns.countplot(x='HolidayFlag', data=df, palette='flare')


# In[36]:


sns.countplot(x='PeriodOfDay', data=df, palette='flare')


# In[37]:


sns.boxplot(x='DayOfWeek', y='ForecastWindProduction', data=df, palette='flare',hue='Year')


# In[38]:


sns.boxplot(x='DayOfWeek', y='ActualWindProduction', data=df, palette='flare',hue='Year')


# In[39]:


sns.boxplot(x='DayOfWeek', y='SystemLoadEA', data=df, palette='flare',hue='Year')


# In[40]:


sns.boxplot(x='DayOfWeek', y='SMPEP2', data=df, palette='flare',hue='Year')


# In[41]:


sns.boxplot(x='DayOfWeek', y='ORKWindspeed', data=df, palette='flare',hue='Year')


# In[42]:


sns.boxplot(x='DayOfWeek', y='ORKTemperature', data=df, palette='flare',hue='Year')


# In[43]:


sns.boxplot(x='Year', y='ORKTemperature', data=df)


# In[44]:


sns.boxplot(x='Year', y='SMPEP2', data=df, palette='flare')


# In[45]:


sns.boxplot(x='Year', y='SMPEA', data=df, palette='flare')


# In[46]:


sns.boxplot(x='Year', y='SystemLoadEA', data=df, palette='flare')


# In[47]:


sns.boxplot(x='Year', y='ORKWindspeed', data=df, palette='flare')


# In[48]:


sns.boxplot(x='Year', y='ActualWindProduction', data=df, palette='flare')


# In[49]:


sns.boxplot(x='Year', y='ForecastWindProduction', data=df, palette='flare')


# In[50]:


sns.boxplot(x='PeriodOfDay', y='SMPEA', data=df, palette='flare')


# In[51]:


sns.boxplot(x='PeriodOfDay', y='SystemLoadEP2', data=df, palette='flare')


# In[52]:


sns.boxplot(x='PeriodOfDay', y='CO2Intensity', data=df, palette='flare')


# In[53]:


sns.pairplot(df,diag_kind='kde')


# In[54]:


plt.scatter(x= df['SMPEP2'],y=df['ORKWindspeed'])
plt.xlabel('SMPEP2')
plt.ylabel('ORKWindspeed')
plt.show()


# In[55]:


plt.scatter(x= df['SMPEP2'],y=df['CO2Intensity'])
plt.xlabel('SMPEP2')
plt.ylabel('CO2Intensity')
plt.show()


# In[56]:


plt.scatter(x= df['SMPEP2'],y=df['ActualWindProduction'])
plt.xlabel('SMPEP2')
plt.ylabel('ActualWindProduction')
plt.show()


# In[57]:


plt.scatter(x= df['SMPEP2'],y=df['SystemLoadEP2'],c='y')
plt.xlabel('SMPEP2')
plt.ylabel('SystemLoadEP2')
plt.show()


# In[58]:


plt.scatter(x= df['ORKWindspeed'],y=df['ForecastWindProduction'],c='g')
plt.xlabel('ORKWindspeed')
plt.ylabel('ForecastWindProduction')
plt.show()


# # Data preprocessing

# In[59]:


df.info()


# In[60]:


df[df['Holiday'] =='None']


# In[61]:


df.drop('Holiday',axis = 1 ,inplace=True)


# In[62]:


# outfiter & Missing Data
df.isna().sum()


# In[63]:


from sklearn.impute import KNNImputer
imputer = KNNImputer()
for col in ['ForecastWindProduction','SystemLoadEA','SMPEA','ORKTemperature','ORKWindspeed',
'CO2Intensity','ActualWindProduction','SystemLoadEP2','SMPEP2']:
    df[col]=imputer.fit_transform(df[[col]]) 


# In[64]:


df.columns


# In[65]:


df = df[['HolidayFlag', 'DayOfWeek', 'WeekOfYear', 'Day', 'Month',
       'Year', 'PeriodOfDay', 'ForecastWindProduction', 'SystemLoadEA',
       'SMPEA', 'ORKTemperature', 'ORKWindspeed', 'CO2Intensity',
       'ActualWindProduction', 'SystemLoadEP2', 'SMPEP2']]


# In[66]:


df


# In[67]:


# Split data to Train and Test Sets
x = df.drop('SMPEP2',axis=1)
y = df['SMPEP2']
print(x.shape)
print(y.shape)


# In[68]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.25)


# In[69]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[70]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# # Machine Learning

# In[71]:


from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error


# In[72]:


models = {
    "LR": LinearRegression(),
    "KNNR" : KNeighborsRegressor(), 
    "SVR": SVR(),
    "DT": DecisionTreeRegressor(),
    "RF": RandomForestRegressor(),
    "XGBR": XGBRegressor()
}


# In[73]:


for name, model in models.items():
    print(f'Using model: {name}')
    model.fit(x_train, y_train)
    print(f'Training Score: {model.score(x_train, y_train)}')
    print(f'Test Score: {model.score(x_test, y_test)}') 
    y_pred = model.predict(x_test)
    print(f'RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}')
    print('-----------------------------------------------------------')


# In[83]:


model = RandomForestRegressor()
model.fit(x_train, y_train)


# In[84]:


y_pred = model.predict(x_test)
y_pred


# In[85]:


y_test


# In[86]:


from sklearn.metrics import r2_score
r2_score(y_test, y_pred)


# In[87]:


plt.plot(y_test,'o')


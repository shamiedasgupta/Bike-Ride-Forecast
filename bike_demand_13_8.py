# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 22:02:19 2023

@author: manas
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("D:/Documents/placements/bike-sharing-demand/train.csv")

sns.histplot(df['count'],kde=True,bins=5,stat="density")

df['count'].describe().to_clipboard(excel=True,sep='\t')


df['datetime']=pd.to_datetime(df['datetime'],format='%d-%m-%Y %H:%M')
df['Year']=df['datetime'].dt.year
df['Month_name']=df['datetime'].dt.month_name()
df['Day_of_week']=df['datetime'].dt.day_name()
df['Hour']=df['datetime'].dt.hour


des_w=df.groupby('Day_of_week').mean()['count']

des_w.plot()
plt.ylabel('Average number of bike rides')
plt.grid()

des_m=df.groupby('Month_name').mean()['count']

des_m.plot()
plt.ylabel('Average number of bike rides')
plt.grid()


df['season']=df['season'].apply(lambda x: str(x).replace('1','spring') if '1' in str(x) else x)
df['season'] = df['season'].apply(lambda x: str(x).replace('2', 'summer') if '2' in str(x) else x)
df['season'] = df['season'].apply(lambda x: str(x).replace('3', 'fall') if '3' in str(x) else x)
df['season'] = df['season'].apply(lambda x: str(x).replace('4', 'winter') if '4' in str(x) else x)


des_s=df.groupby('season').mean()['count']

sns.barplot(x=df['season'],y=df['count'])


# counvet the weather from numerical(1,2,3,4) to object Clear , Mist , Rainy ,snowy
df['weather'] = df['weather'].apply(lambda x: str(x).replace('1', 'Clear') if '1' in str(x) else x)
df['weather'] = df['weather'].apply(lambda x: str(x).replace('2', 'Mist') if '2' in str(x) else x)
df['weather'] = df['weather'].apply(lambda x: str(x).replace('3', 'Rainy') if '3' in str(x) else x)
df['weather'] = df['weather'].apply(lambda x: str(x).replace('4', 'snowy') if '4' in str(x) else x)


df_wt=df.groupby('weather').describe()['count']
sns.barplot(x= df['weather'] ,y = df['count'])
plt.xlabel("weather")
plt.ylabel("count")
plt.title("The effect of weather on count")


sns.barplot(x= df['holiday'] ,y = df['count'])
plt.xlabel("holiday")
plt.ylabel("count")
plt.title("The effect of holiday on count")



plt.figure(figsize= (10,6))
sns.heatmap(df.corr(),annot=True)



des_t=df.groupby('Hour').mean()['count']

des_t.plot()
plt.ylabel('Average number of bike rides')
plt.xticks(df['Hour'].drop_duplicates())
plt.grid()

des_wd=df.groupby('workingday').mean()['count']


plt.figure(figsize=(10,6))
sns.barplot(x= df['workingday'] ,y = df['count'])
plt.xlabel("workingday")
plt.ylabel("count")


df1=df.copy()

df1= pd.get_dummies(df1,columns= ['season','weather','Day_of_week'],drop_first= True)
df1['Month']= df['datetime'].dt.month
df1=df1.drop(['Month_name'],axis=1)
df1.drop(['datetime','temp','registered'],axis=1,inplace=True)

X= df1.drop(['count'],axis=1)
y= df1['count']


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1,test_size=0.25)



# =============================================================================
# from sklearn.preprocessing import StandardScaler
# 
# sc = StandardScaler()
# 
# X_train=sc.fit_transform(X_train)
# 
# X_test=sc.transform(X_test)
# 
# 
# =============================================================================

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error



models = {
    
    "LR":LinearRegression(),
    "DT": DecisionTreeRegressor(),
    "RF":RandomForestRegressor(),
    "XGB":XGBRegressor()
}

for name, model in models.items():
    print(f'Using model: {name}')
    model.fit(X_train,y_train)
    print(f'Training Score: {model.score(X_train, y_train)}')
    print(f'Test score: {model.score(X_test,y_test)}')
    y_pred = model.predict(X_test)
    print(f'RMSE : {np.sqrt(mean_squared_error(y_test,y_pred))}')
    print("-------------------------------------------")
































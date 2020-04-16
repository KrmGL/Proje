
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


ds = pd.read_csv("SEER_Breast_Cancer_Dataset.csv")

#Dataset'i Düzenleme
ds.drop(["Race ","Marital Status","Unnamed: 3","Grade",
         "A Stage","6th Stage","Status"],axis=1,inplace = True)  #işimize yaramayan sütunları kaldırdık.

#sütün isimlerinde boşluk olanları "_" ile birleştirdik
ds.columns = [each.split()[0]+"_"+each.split()[1] if (len(each.split())>1) else each for each in ds.columns]

ds.columns=[i.lower() for i in ds.columns]#sütun isimlerini küçük harf yaptık.
ds.rename(columns={'status':'live_dead'}, inplace=True)

#t_stage   T1:Tümör ≤ 20 mm==0              n_stage    N1:Hareketli ipsilateral lenf nodu==0
#           T2:Tümör>20 mm, ≤ 50 mm==1                 N2:Fikse ya da konglomere lenf nodu==1
#           T3:Tümör>50 mm  ==2                           N3:İpsilateral infraklavikular lenf nodu ==2
#           T4:inflamatuvar göğüs kanseri ==3

ds["t_stage"]=[ 0 if each=="T1" else each for each in ds.t_stage]
ds["t_stage"]=[ 1 if each=="T2" else each for each in ds.t_stage]
ds["t_stage"]=[ 2 if each=="T3" else each for each in ds.t_stage]
ds["t_stage"]=[ 3 if each=="T4" else each for each in ds.t_stage]

ds["n_stage"]=[ 0 if each=="N1" else each for each in ds.n_stage]
ds["n_stage"]=[ 1 if each=="N2" else each for each in ds.n_stage]
ds["n_stage"]=[ 2 if each=="N3" else each for each in ds.n_stage]

ds["estrogen_status"]=[1 if each=="Positive" else 0 for each in ds.estrogen_status]
ds["progesterone_status"]=[1 if each=="Positive" else 0 for each in ds.progesterone_status]


y = ds.survival_months.values.reshape(-1,1)
X = ds.drop(["survival_months"],axis=1)

y = (y - np.min(y))/(np.max(y)-np.min(y))#y yi normalize ettik

#%%---------------------------------------------------------------------------
#--------------------------------------------------------------------------
from sklearn import metrics
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#%%linearregression
from sklearn.linear_model import LinearRegression

_linear_regressor = LinearRegression()  
_linear_regressor.fit(x_train, y_train)
   
y_pred = _linear_regressor.predict(x_test) 

print("_linear_regressor score:", _linear_regressor.score(x_test,y_test))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
#%%DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor

_dt_regressor = DecisionTreeRegressor()
_dt_regressor.fit(x_train, y_train)

dt_y_pred = _dt_regressor.predict(x_test)
dt_y_pred = dt_y_pred.reshape(-1,1)

print("_dt_regressor score:", _dt_regressor.score(x_test,y_test))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
#%%
from sklearn.ensemble import RandomForestRegressor

_rf_regressor = RandomForestRegressor(n_estimators = 100, random_state =1)
_rf_regressor.fit(x_train, y_train)

rf_y_pred = _dt_regressor.predict(x_test)
rf_y_pred = dt_y_pred.reshape(-1,1)


print("_rf_regressor score:", _rf_regressor.score(x_test,y_test))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


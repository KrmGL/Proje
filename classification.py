
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


data = pd.read_csv("HCV_Egypt_Data.csv")

data.columns = [each.split()[0]+"_"+each.split()[1] if (len(each.split())>1) else each for each in data.columns]
data.columns=[i.lower() for i in data.columns]

norm = data.iloc[:,10:28]
data = data.drop(norm,axis=1)
normalize = (norm - np.min(norm))/(np.max(norm)-np.min(norm)).values

dataset = pd.concat([data,normalize],axis=1)#datamızı kullanışlı hale getirdik.

y = dataset.bhs.values
X = dataset.drop(["bhs"],axis=1)
#%%
from sklearn import metrics
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state=42)

#%%knn
#from sklearn import KNeighborsClassifier
from sklearn import neighbors

knn = neighbors.KNeighborsClassifier(n_neighbors = 100)
knn.fit(x_train,y_train)
prediction = knn.predict(x_test)
print("knn score:", knn.score(x_test,y_test))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, prediction))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, prediction))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))

#%% svm

from sklearn import svm
svm = svm.SVC(random_state=1)
svm.fit(x_train,y_train)
prediction_svm = svm.predict(x_test)
print("svm accuary: ",svm.score(x_test,y_test))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, prediction_svm))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, prediction_svm))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, prediction_svm)))
#%% rf classification

from sklearn import ensemble
rf= ensemble.RandomForestClassifier(n_estimators=10,random_state=1)
rf.fit(x_train,y_train)
prediction_rf = rf.predict(x_test)
print("rf accuracy: ",rf.score(x_test,y_test))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, prediction_rf))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, prediction_rf))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, prediction_rf)))






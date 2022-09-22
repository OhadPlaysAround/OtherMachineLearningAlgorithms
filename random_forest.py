from random import random

import pandas as pd
import numpy as np
import time
# dataset = pd.read_csv("C:\\Users\\ohad.benhaim\\OneDrive - Qualitest Group\\Documents\\AlgoTrace Datasets\\Classification Examples\\breast_cancer.csv")
folder = "C:\\Users\\ohad.benhaim\\OneDrive - Qualitest Group\\Documents\\AlgoTrace Datasets\\Classification and Estimation\\"
dataset = pd.read_csv(folder + "weatherAUS_no_null_everywhere.csv")
ncol = dataset.shape[1]-1
X = dataset.iloc[:, 1:ncol-1].values
y = dataset.iloc[:, ncol].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
# from sklearn.preprocessing import StandardScaler

# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)


from sklearn import preprocessing

le = preprocessing.LabelEncoder()
for i in range(ncol-2):
    X_train[:, i] = le.fit_transform(X_train[:, i])
    X_test[:, i] = le.fit_transform(X_test[:, i])

from sklearn.ensemble import RandomForestRegressor
Y = np.zeros((len(y_test), 210))
Y[:, 0] = y_test
for i in range (1,201):
    # random.seed(i)
    regressor = RandomForestRegressor(n_estimators=20, random_state=i)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    Y[:, i] = y_pred
    print(i/200)
Y[:, 201] = Y[:, 1:11].mean(1)
Y[:, 202] = Y[:, 1:51].mean(1)
Y[:, 203] = Y[:, 1:101].mean(1)
Y[:, 204] = Y[:, 1:201].mean(1)

Y[:, 205] = 1*(Y[:, 1]>=0.5)
Y[:, 206] = 1*(Y[:, 201]>=0.5)
Y[:, 207] = 1*(Y[:, 202]>=0.5)
Y[:, 208] = 1*(Y[:, 203]>=0.5)
Y[:, 209] = 1*(Y[:, 204]>=0.5)

np.savetxt("foo.csv", Y, delimiter=',')
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import confusion_matrix

# print(confusion_matrix(y_test,y_pred_clean))
# print(classification_report(y_test,y_pred_clean))
# print(accuracy_score(y_test, y_pred_clean))

print(confusion_matrix(Y[:, 0], Y[:, 205]))
print(confusion_matrix(Y[:, 0], Y[:, 206]))
print(confusion_matrix(Y[:, 0], Y[:, 207]))
print(confusion_matrix(Y[:, 0], Y[:, 208]))
print(confusion_matrix(Y[:, 0], Y[:, 209]))


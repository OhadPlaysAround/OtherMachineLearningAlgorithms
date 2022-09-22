import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier

# Now let's load in our training data:

folder = "C:\\Users\\ohad.benhaim\\OneDrive - Qualitest Group\\Documents\\AlgoTrace Datasets\\Classification and Estimation\\"
dataset = pd.read_csv(folder + "weatherAUS_no_null_everywhere.csv")
n_col = dataset.shape[1]
X = dataset.iloc[:, 1:n_col-2].values
y = dataset.iloc[:, n_col-1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

le = preprocessing.LabelEncoder()
for i in range(n_col-3):
    X_train[:, i] = le.fit_transform(X_train[:, i])
    X_test[:, i] = le.fit_transform(X_test[:, i])

# We'll now scale our data by creating an instance of the Scaler and scaling it:

Scaler = MinMaxScaler()
X_train = Scaler.fit_transform(X_train)
X_test = Scaler.transform(X_test)

# # Now we can split the data into training and testing sets. Let's also set a seed (so you can replicate the results) and
# # select the percentage of the data for testing on:
#
# # state = 12
# # test_size = 0.20
#
# # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_size, random_state=state)
#
# # Now we can try setting different learning rates, so that we can compare the performance of the classifier's
# # performance at different learning rates.
#
# Y = np.zeros((len(y_test), 210))
# Y[:, 0] = y_test
# # these are the parameters tuned in the advanced hyper-parameters in Qualisense
# eta = 0.3
# sub = 0.75
# depth = 10
# n_round = 5
# for i in range(1, 201):
#     seed = i
#     gb_clf = GradientBoostingClassifier(n_estimators=n_round, learning_rate=eta, max_depth=depth,
#                                         subsample=sub, random_state=seed)
#     gb_clf.fit(X_train, y_train)
#     predictions = gb_clf.predict_proba(X_test)
#     Y[:, i] = predictions[:, 1]
#     print(i/200)
#
# Y[:, 201] = Y[:, 1:11].mean(1)
# Y[:, 202] = Y[:, 1:51].mean(1)
# Y[:, 203] = Y[:, 1:101].mean(1)
# Y[:, 204] = Y[:, 1:201].mean(1)
#
# Y[:, 205] = 1*(Y[:, 1] >= 0.5)
# Y[:, 206] = 1*(Y[:, 201] >= 0.5)
# Y[:, 207] = 1*(Y[:, 202] >= 0.5)
# Y[:, 208] = 1*(Y[:, 203] >= 0.5)
# Y[:, 209] = 1*(Y[:, 204] >= 0.5)
#
# np.savetxt("xgb_report.csv", Y, delimiter=',')
#
# print("Confusion Matrix:")
# print(confusion_matrix(Y[:, 0], Y[:, 205]))
# print(confusion_matrix(Y[:, 0], Y[:, 206]))
# print(confusion_matrix(Y[:, 0], Y[:, 207]))
# print(confusion_matrix(Y[:, 0], Y[:, 208]))
# print(confusion_matrix(Y[:, 0], Y[:, 209]))


eta = 0.3
sub = 0.75
depth = 10
n_round = 5
seed = 1
gb_clf = GradientBoostingClassifier(n_estimators=n_round, learning_rate=eta, max_depth=depth,
                                    subsample=sub, random_state=seed)
gb_clf.fit(X_train, y_train)
predictions = gb_clf.predict_proba(X_test)
print(confusion_matrix(y_test, 1*(predictions[:, 1] >= 0.5)))

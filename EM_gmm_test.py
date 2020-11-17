
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn import datasets, metrics
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold

df = pd.read_csv("letter-recognition.csv", header=None) # !!!type is dataframe, not ndarray!!
df = np.array(df)  # type conversion needed to use slicing
# df[df[:, 0] == 'A'] = 0
# df[df[:, 0] == 'B'] = 1
# df[df[:, 0] == 'C'] = 2
# df[df[:, 0] == 'D'] = 3
# df[df[:, 0] == 'E'] = 4
# df[df[:, 0] == 'F'] = 5
# df[df[:, 0] == 'G'] = 6
# df[df[:, 0] == 'H'] = 7
# df[df[:, 0] == 'I'] = 8
# df[df[:, 0] == 'J'] = 9
# df[df[:, 0] == 'K'] = 10
# df[df[:, 0] == 'L'] = 11
# df[df[:, 0] == 'M'] = 12
# df[df[:, 0] == 'N'] = 13
# df[df[:, 0] == 'O'] = 14
# df[df[:, 0] == 'P'] = 15
# df[df[:, 0] == 'Q'] = 16
# df[df[:, 0] == 'R'] = 17
# df[df[:, 0] == 'S'] = 18
# df[df[:, 0] == 'T'] = 19
# df[df[:, 0] == 'U'] = 20
# df[df[:, 0] == 'V'] = 21
# df[df[:, 0] == 'W'] = 22
# df[df[:, 0] == 'X'] = 23
# df[df[:, 0] == 'Y'] = 24
# df[df[:, 0] == 'Z'] = 25
# df_target_names = ['A', 'B', 'C']
df = df[:200, :]
df_data = df[:, 1:]
df_target = df[:, 0]


gmm = GaussianMixture(n_components=16, n_init=10)
gmm.fit(df_data)
prob = gmm.predict_proba(df_data)
print (prob.shape)
para = gmm._get_parameters()
print (len(para) ) # tuple length
for i in range(len(para)):
    print (para[i].shape)



# X_train = df_data[:15000, 1:]
# y_train = df_target[:15000]
# X_test = df_data[15000:, 1:]
# y_test = df_target[15000:]
#
# n_classes = len(np.unique(y_train))
#
# # Try GMMs using different types of covariances.
# estimators = dict((cov_type, GaussianMixture(n_components=n_classes,
#                    covariance_type=cov_type, max_iter=200, random_state=0))
#                   for cov_type in ['spherical', 'diag', 'tied', 'full'])
#
# n_estimators = len(estimators)
#
#
# for index, (name, estimator) in enumerate(estimators.items()):
#     # Since we have class labels for the training data, we can
#     # initialize the GMM parameters in a supervised manner.
#     estimator.means_init = np.array([X_train[y_train == i].mean(axis=0)
#                                     for i in range(n_classes)])
#
#     # Train the other parameters using the EM algorithm.
#     estimator.fit(X_train)
#
#     y_train_pred = estimator.predict(X_train)
#     #print(y_train_pred[0:50])
#     train_accuracy = np.mean(y_train_pred.ravel() == y_train.ravel()) * 100
#     print train_accuracy, "train_accuracy for", index
#
#     y_test_pred = estimator.predict(X_test)
#     test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
#     print test_accuracy, "test_accuracy for ", index


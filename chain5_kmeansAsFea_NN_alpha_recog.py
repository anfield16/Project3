

from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import scale


def neural_network_predict(data_transformed, label):
    X = data_transformed[0:16000, :]
    y = label[0:16000]

    # neural_clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    neural_clf = MLPClassifier()

    neural_clf.fit(X, y)
    predicted = neural_clf.predict(data_transformed[16000:, :])
    expected = label[16000:]

    return metrics.accuracy_score(expected, predicted)

np.random.seed(42)

df = pd.read_csv("letter-recognition.csv", header=None) # !!!type is dataframe, not ndarray!!
df = np.array(df)  # type conversion needed to use slicing
data = df[:, 1:]
data = scale(data)
labels = df[:, 0]

dimensions = np.array([2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17])
predict_accu = np.zeros(len(dimensions))

for i in range(len(dimensions)):
    if dimensions[i] < 17:
        kmeans = KMeans(n_clusters=dimensions[i], max_iter=300).fit_transform(data)
        cluster_as_feature = kmeans
    else:
        cluster_as_feature = data
    predict_accu[i] = neural_network_predict(data_transformed=cluster_as_feature, label=labels)

plt.plot(dimensions[:-1], predict_accu[:-1], marker='o', linestyle='--', color='g')
plt.plot(dimensions[-1], predict_accu[-1], marker='D', color='b')
plt.ylabel('Prediction Accuracy')
plt.xlabel('New features from Kmeans used for NN')
plt.title("Accuracy of NN with only new features from Kmeans")
plt.show()




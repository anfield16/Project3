
from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale


df = pd.read_csv("letter-recognition.csv", header=None) # !!!type is dataframe, not ndarray!!
df = np.array(df)  # type conversion needed to use slicing
data = df[:, 1:]
labels = df[:,0]
print(data.shape)
dimensions = range(2,17)
for dim in dimensions:
    pca = PCA(n_components=2, whiten=True)
    pca.fit_transform(data)
    print(pca.explained_variance_)

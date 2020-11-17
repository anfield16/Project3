#############################################################################
## code for dataset: letter-recognition
## precision is

from sklearn import metrics
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np


df = pd.read_csv("letter-recognition.csv", header = None) # !!!type is dataframe, not ndarray!!
# print 'shape of data: ', df.shape
df = np.array(df)  # type conversion needed to use slicing
# print type(df2)
# print df2[:1,:]
data = df[:, 1:]
X = df[0:16000, 1:]
y = df[0:16000, 0]
test_data = df[16000:, 1:]
test_label = df[16000:, 0]


# Create a classifier: kmeans
kmeans = KMeans(n_clusters=26, max_iter = 300).fit(data)
trasnformed_data = kmeans.transform(data)
print trasnformed_data.shape
print type(kmeans)
attrs = vars(kmeans)
#print ', '.join("%s: %s" % item for item in attrs.items())

# for i in range(0, 100):
#     print y[i], kmeans.labels_[i]


#kmeans.predict(test_data)
# print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, kmeans.labels_))
# print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
# print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))



# # We learn the digits on the first half of the digits
# adabst_classifier.fit(X, y)
#
# # Now predict the value of the digit on the second half:
# expected = df[16000:, 0]
# predicted = adabst_classifier.predict(df[16000:, 1:])
#
# print("Classification report for classifier %s:\n%s\n"
#       % (adabst_classifier, metrics.classification_report(expected, predicted)))
# print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
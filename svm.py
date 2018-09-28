#from sklearn.svm import LinearSVC
import numpy as np
from sklearn import svm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


data = np.loadtxt('svm.csv', delimiter='	', dtype = float)
labels = data[:, 0:1]
features = preprocessing.minmax_scale(data[:, 1:])

x_train, x_test, y_train, y_test = train_test_split(features, labels.ravel(), test_size=0.5)


clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovo', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
clf.fit(x_train, y_train)
clf.predict(x_test)





from sklearn.metrics import accuracy_score, precision_score, recall_score

predict = clf.predict(x_test)
print(accuracy_score(y_test, predict), precision_score(y_test, predict, average='micro'), recall_score(y_test, predict, average='micro'))

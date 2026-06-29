import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import pickle

iris = load_iris()
iris_x = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_y = pd.Series(iris.target)

X_train, X_test, y_train, y_test = train_test_split(iris_x, iris_y, test_size=.2, random_state=1)

log = LogisticRegression(class_weight='balanced')
log.fit(X_train, y_train)

pickle.dump(log, open('iris_prediction.pkl', 'wb'))
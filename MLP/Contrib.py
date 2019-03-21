#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 23:46:58 2019

@author: sameepshah
"""

from sklearn.datasets import load_iris
iris = load_iris()
X = iris['data']
y = iris['target']

y.dtype

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

import tensorflow.contrib.learn.python.learn as learn

classifier = learn.DNNClassifier(hidden_units=[10, 20, 10], n_classes=3)
classifier.fit(X_train, y_train, steps=200, batch_size=32)

iris_predictions = classifier.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(y_test,iris_predictions))
print(confusion_matrix(y_test, iris_predictions))

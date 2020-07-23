#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Importing the libraries
import pandas as pd
import numpy as np
import tensorflow as tf


#Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

#Encoding categorical data

### Gender column ###
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

### Geography Column ###
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [1])],
                       remainder = 'passthrough')
X = np.array(ct.fit_transform(X))

#Splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Initializing the ANN
ann = tf.keras.models.Sequential()

#Adding the input layer an the first hidden layer
ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu'))

#Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu'))

#Adding the output layer
ann.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))

#Compiling the ANN
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Training the ANN on the training set
ann.fit(X_train, y_train, batch_size = 32, epochs = 100)

#Predicting the result of a single observation
print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5)

#Predicting the test set results
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
conc = np.concatenate((y_test.reshape(len(y_test), 1), y_pred.reshape(len(y_pred), 1)),
                      axis = 1)
print(conc)

#Making the confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
print(cm)
print(acc)

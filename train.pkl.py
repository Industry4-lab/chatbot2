# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 14:36:02 2025

@author: Industry4.0
"""

# train_knn_model.py
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Load Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split into training/testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train KNN model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "train.pkl")
print("Model saved as train.pkl")

# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 21:51:58 2021

@author: david
""" 

# Adding to path the project root
import sys
sys.path.append("..")

# Preprocessing class
from DataProcessing.Preprocessing import Preprocessing

# Import LinearRegression class
from sklearn.linear_model import LinearRegression

# Plotting library
import matplotlib.pyplot as plt


# ---------------------------- Data processing -----------------------------
pp = Preprocessing('Salary_Data.csv', 'Salary') # Instance of preprocessor

# Treat missing and categorical data
dataset = pp.treatData()

# Split and scale dataset
x_train, x_test, y_train, y_test = pp.splitAndScale(test_size=1/3)
# ---------------------------- Data processing -----------------------------

# --------------------------- Linear regression ----------------------------
regression = LinearRegression()     # Instance of linear regression
regression.fit(x_train, y_train)    # Train model

y_pred = regression.predict(x_test) # Make predictions

# Training data visualization
plt.scatter(x_train, y_train, color="red")                   # Training data
plt.plot(x_train, regression.predict(x_train), color="blue") # Prediction data
plt.title("Sueldo vs Años de experiencia (conjunto de entrenamiento)")
plt.xlabel("Años de experiencia")
plt.ylabel("Sueldo")
plt.grid()
plt.show()

# Training data visualization
plt.scatter(x_test, y_test, color="red")                     # Test data
plt.plot(x_train, regression.predict(x_train), color="blue") # Prediction data
plt.title("Sueldo vs Años de experiencia (conjunto de testing)")
plt.xlabel("Años de experiencia")
plt.ylabel("Sueldo")
plt.grid()
plt.show()
# --------------------------- Linear regression ----------------------------

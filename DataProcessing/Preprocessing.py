# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 03:02:44 2021

@author: david

Data preprocessing template
"""

import numpy as np
import pandas as pd

# Data NA treatment
from sklearn.impute import SimpleImputer

# Data encoding for categorical data
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Data splitting in training and testing
from sklearn.model_selection import train_test_split

# Data scaling
from sklearn.preprocessing import StandardScaler


class Preprocessing:
    def __init__(self, pathToCsv, colToPredict):

        self.csv_path = pathToCsv
        self.dataset = None
        self.x = None
        self.y = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

        # Import dataset
        self.importDataset(colToPredict)

    # Dataset import
    def importDataset(self, column):
        # Dataset import
        self.dataset = pd.read_csv(self.csv_path)
        # Set dependent variables
        self.x = self.dataset.iloc[:, self.dataset.columns != column].values
        # Set independent variable
        self.y = self.dataset.iloc[:, self.dataset.columns == column].values

    # Missing data
    def naDataTreatment(self, missing):
        # Instantiate SimpleImputer with mean strategy
        # Missing values are the NaN
        si = SimpleImputer(missing_values=np.nan, strategy="mean")
        # Treat all rows and replace only columns number 1 and 2 in our x
        self.x[:, missing] = si.fit_transform(self.x[:, missing])

    # Categorical data treatment
    def categoricalDataTreatment(self, categorical_x, categorical_y):
        # Encode categorical data for x
        if len(categorical_x) != 0:
            ct = ColumnTransformer(
                [('one_hot_encoder',
                  OneHotEncoder(categories='auto', drop='if_binary'), categorical_x)],
                remainder='passthrough'
            )
            self.x = np.array(ct.fit_transform(self.x), dtype=float)


        # Encode categorical data for y
        if categorical_y:
            ct = ColumnTransformer(
                [('one_hot_encoder', OneHotEncoder(categories='auto', drop='if_binary'), [0])],
                remainder='passthrough'
            )
            self.y = np.array(ct.fit_transform(self.y), dtype=float)

    # Split dataset in training and testing sets
    def splitDataset(self, test_size):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x,
            self.y,
            test_size=test_size,
            random_state=0
        )
        
        return self.x_train, self.x_test, self.y_train, self.y_test
        
    def scaleData(self, scale_x, scale_y):
        # Standardization
        if scale_x:
            # Scaling x (train and testing)
            sc_x = StandardScaler()
            self.x_train = sc_x.fit_transform(self.x_train)
            # Here we only do a transformation and not a fitting as we want to keep
            # the same scaling on both matrixes
            self.x_test = sc_x.transform(self.x_test)
            
        if scale_y:
            # Scaling y (train and testing)
            sc_y = StandardScaler()
            self.y_train = sc_y.fit_transform(self.y_train)
            # Here we only do a transformation and not a fitting as we want to keep
            # the same scaling on both matrixes
            self.y_test = sc_y.transform(self.y_test)
            
        return self.x_train, self.x_test, self.y_train, self.y_test
            
        
    def treatData(self, missing=[], cat_x=[], cat_y=False):
        # Missing data treatment
        if len(missing) != 0:
            self.naDataTreatment(missing)
            
        # Categorical data treatment
        if len(cat_x) != 0 or cat_y:
            self.categoricalDataTreatment(cat_x, cat_y)
            
        return self.dataset, self.x, self.y
    
    def splitAndScale(self, test_size=0.2, sc_x=False, sc_y=False):
        self.splitDataset(test_size)
        return self.scaleData(sc_x, sc_y)
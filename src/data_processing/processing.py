import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


# Imputation of missing values for numerical variables
# by the mode (the value that appears most often in a set of data values).
class ImputerNumericalVariable(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        self.variables = variables

    def fit(self, X, y=None):
        self.variable_mode_dict = {}
        for variable in self.variables:
            self.variable_mode_dict[variable] = X[variable].mode()[0]
        return self

    def transform(self, X):
        X = X.copy()
        for variable in self.variables:
            X[variable].fillna(self.variable_mode_dict[variable], inplace=True)
        return X


# Imputation of missing values for categorical variables.
# Replace missing values with new label: "Missing".
class ImputerCategoricalVariable(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for variable in self.variables:
            X[variable] = X[variable].fillna("Missing")
        return X


# Get the time elapsed between variable and the year in which the house was sold
class ProcessorTemporalVariable(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None, reference_variable=None):
        self.variables = variables
        self.reference_variables = reference_variable

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for variable in self.variables:
            X[variable] = X[self.reference_variables] - X[variable]
        return X


# Logarithm transformation of non-normal distributed variables.
class TransformerLogarithm(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for variable in self.variables:
            X[variable] = np.log(X[variable])
        return X


# Replace rare labels (which appear only in a small proportion of the observations) by the string "Rare".
class EncoderRareLabel(BaseEstimator, TransformerMixin):

    def __init__(self, tolerance, variables=None):
        self.variables = variables
        self.tolerance = tolerance

    def fit(self, X, y=None):
        self.rare_label_dict = {}
        for variable in self.variables:
            frequent_var = pd.Series(X[variable].value_counts() / np.float(len(X)))
            self.rare_label_dict[variable] = list(frequent_var[frequent_var >= self.tolerance].index)
        return self

    def tranform(self, X):
        X = X.copy()
        for variable in self.variables:
            X[variable] = np.where(X[variable].isin(self.rare_label_dict[variable]), X[variable], "Rare")
        return X


# Transform the strings of the categorical variables into numbers.
class EncoderCategoricalVariable(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        self.variables = variables

    def fit(self, X, y):
        temp = pd.concat([X, y], axis=1)
        temp.columns = list(X.columns) + ['target']
        self.ordered_labels_dict = {}
        for variable in self.variables:
            ordered_labels = temp.groupby([variable])['target'].mean().sort_values(ascending=True).index
            self.ordered_labels_dict[variable] = {k: i for i, k in enumerate(ordered_labels, 0)}
        return self

    def transform(self, X):
        X = X.copy()
        for variable in self.variables:
            X[variable] = X[variable].map(self.ordered_labels_dict[variable])
        return X


# Drop unnecessary variables.
class DropSelectedVariable(BaseEstimator, TransformerMixin):

    def __init__(self, drop_variables=None):
        self.variables = drop_variables

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X = X.drop(self.variables, axis=1)
        return X



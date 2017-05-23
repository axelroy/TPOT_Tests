from tpot import TPOTRegressor
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from deap import creator
from sklearn.model_selection import cross_val_score

# CHUV datas
dataset = pd.read_csv('data.csv', header=1, sep=',')
# Slices the columns
X, y = dataset.iloc[:,:-1], dataset.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tpot = TPOTRegressor(generations=1, population_size=50, verbosity=2)
tpot.fit(X_train, y_train)
print("Score of the best model found with the test data", tpot.score(X_test, y_test), "\n")

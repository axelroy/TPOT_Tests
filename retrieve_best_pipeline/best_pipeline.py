from tpot import TPOTClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
from deap import creator
from sklearn.model_selection import cross_val_score

# Iris flower classification
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data.astype(np.float64),
    iris.target.astype(np.float64), train_size=0.75, test_size=0.25)

tpot = TPOTClassifier(generations=3, population_size=50, verbosity=2)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))

# print part of pipeline dictionary
print("Part of the dict", dict(list(tpot._evaluated_individuals.items())[0:2]))

# print a pipeline and its values
pipeline_str = list(tpot._evaluated_individuals.keys())[0]
print("First dict entry config \n", pipeline_str)
print("First dict entry value \n", tpot._evaluated_individuals[pipeline_str])

# convert pipeline string to scikit-learn pipeline object
optimized_pipeline = creator.Individual.from_string(pipeline_str, tpot._pset) # deap object
fitted_pipeline = tpot._toolbox.compile(expr=optimized_pipeline ) # scikit-learn pipeline object

# print scikit-learn pipeline objectP
print("Best Pipeline \n", fitted_pipeline)

# Fix random state when the operator allows  (optional) just for get consistent CV score
tpot._set_param_recursive(fitted_pipeline.steps, 'random_state', 42)

# CV scores from scikit-learn
scores = cross_val_score(fitted_pipeline, X_train, y_train, cv=5, scoring='accuracy', verbose=0)
print("Scores  \n", np.mean(scores))
print("Scores  \n", tpot._evaluated_individuals[pipeline_str][1])

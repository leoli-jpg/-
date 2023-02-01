import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import tree

data = np.array([[0, 2, 1, 1], 
[0, 2, 1, 0], 
[1, 2, 1, 1], 
[2, 1, 1, 1], 
[2, 0, 0, 1],
[2, 0, 0, 0],
[1, 0, 0, 0],
[0, 1, 1, 1],
[0, 0, 0, 1],
[2, 1, 0, 1],
[0, 1, 0, 0],
[1, 1, 1, 0],
[1, 2, 0, 1]
])

clf = DecisionTreeClassifier(criterion="entropy")
y = [1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0]
clf.fit(data, y)
fn = ['Age', 'Incoming', 'Student', 'Credit Rating']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(clf, feature_names=fn)
plt.show()

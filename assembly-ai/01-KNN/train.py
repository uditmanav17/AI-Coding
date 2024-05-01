# %% library imports
import matplotlib.pyplot as plt
import numpy as np
from knn import KNN
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.model_selection import train_test_split

# %% load, split and plot data
cmap = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])

iris = datasets.load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

plt.figure()
plt.scatter(X[:, 2], X[:, 3], c=y, cmap=cmap, edgecolor="k", s=20)
plt.show()

# %% train KNN
clf = KNN(k=5)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
# %%
acc = np.sum(predictions == y_test) / len(y_test)
print(acc)
# %%

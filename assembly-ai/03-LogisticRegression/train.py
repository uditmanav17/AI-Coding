from logistic_reg import LogisticRegression
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=43,
)

clf = LogisticRegression(lr=0.01)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test, thresh=0.4)


print(f"Test accuracy : {accuracy_score(y_test, y_pred)}")

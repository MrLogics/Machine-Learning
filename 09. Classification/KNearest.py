from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier


iris = datasets.load_iris()
# print(iris.DESCR)

features = iris.data
labels = iris.target

clf = KNeighborsClassifier()
clf.fit(features, labels)

pred = clf.predict([[5.1, 3.2, 0.6, 1]])

if pred == 0:
    print("Setosa")
elif pred == 1:
    print("Versicolor")
elif pred == 2:
    print("verginica")



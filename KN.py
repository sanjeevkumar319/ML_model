from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

#loading datasets
iris= datasets.load_iris()

#printing description and features
print(iris.DESCR)
features=iris.data
labels = iris.target
print(features[0] , labels[0])

#  training the classifier
clf= KNeighborsClassifier()
clf.fit(features,labels)

predi=clf.predict([[1,1,1,1]])  #means saple length saple width , petal length and petal width
print(predi)
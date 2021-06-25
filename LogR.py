from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
iris=datasets.load_iris()
# print(iris.keys())
# print(iris['data'])
# print(iris['target'])
# print(iris['DESCR'])

#using now one feature and trained our logistic classifier to predict whether a
# flower is iris virginica or not

x=iris["data"][:, 3:]   #by slicing we take only 3rd column
y=(iris["target"] == 2).astype(np.int)   #2 is our iris virginca flower and convert it to 0 or 1 by numpy
# print(y)

# training a logistic regression classifier
clf=LogisticRegression()
clf.fit(x,y)

examle=clf.predict(([[2.6]]))
print(examle)

#now plot the visulization using matplotlib
x_new=np.linspace(0,3,1000).reshape(-1,1)   #using linspace we take 1000 points btw o and 3
                                             # and reshape it between -1 to 1
y_prob = clf.predict_proba(x_new)    #pridict the probaility

plt.plot(x_new,y_prob[:,1] , "g-" , label="virginica")
plt.show()

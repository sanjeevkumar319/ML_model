import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets ,linear_model
from sklearn.metrics import mean_squared_error

diabetes=datasets.load_diabetes()
print(diabetes.keys())   #datasets ki key hh (['data', 'target', 'frame', 'DESCR',
                         # 'feature_names', 'data_filename', 'target_filename'])
diabetes_X = diabetes.data[:,np.newaxis,2]  #slicing for two features
#print(diabetes_X)

diabetes_X_train= diabetes_X[:-30] #slicing , training of last 30 features
diabetes_X_test = diabetes_X[-30:] #testing of first 30 features

diabetes_Y_train =diabetes.target[:-30]  #this is y labeling
diabetes_Y_test = diabetes.target[-30:]

model = linear_model.LinearRegression()
model.fit(diabetes_X_train,diabetes_Y_train)
diabetes_Y_predicted=model.predict(diabetes_X_test)  #fun bnaya testing k liye

print("mean sqaured error is " ,mean_squared_error(diabetes_Y_test ,diabetes_Y_predicted))

print("weights:",model.coef_)
print("interscept:" , model.intercept_)

plt.scatter(diabetes_X_test ,diabetes_Y_test)
plt.plot(diabetes_X_test,diabetes_Y_predicted)
plt.show()
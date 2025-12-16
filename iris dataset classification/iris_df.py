# -*- coding: utf-8 -*-
#import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report

#initialise df
iris_df = pd.read_csv("/content/iris_dataset.csv")
iris_df.head()

#check if the data has null values
print(iris_df.shape)
print(iris_df.isnull().sum())

#check if the data is balanced to reduce bias
print(iris_df["target"].value_counts())

#map string target values to numerical values
target_map_dict = {"Iris-setosa":0,
                   "Iris-versicolor":1,
                   "Iris-virginica":2}

iris_df["target"] = iris_df["target"].map(target_map_dict)
iris_df.head()

#plotting correlation heatmap
sns.heatmap(iris_df.corr(),annot=True,fmt="g")

"""corrlation strength ranking for features with the target:

1. Petal width
2. petal length
3. sepal length
4. sepal width
"""

#plotting features with scatterplot
fig,axs = plt.subplots(2,3,sharex=True,sharey=True,figsize=(20,15))

axs[0][0].scatter(iris_df["petal length (cm)"],iris_df["petal width (cm)"],c=iris_df["target"])
axs[1][0].scatter(iris_df["sepal length (cm)"],iris_df["sepal width (cm)"],c=iris_df["target"])

axs[0][1].scatter(iris_df["petal length (cm)"],iris_df["sepal width (cm)"],c=iris_df["target"])
axs[1][1].scatter(iris_df["sepal length (cm)"],iris_df["petal width (cm)"],c=iris_df["target"])

axs[0][2].scatter(iris_df["sepal length (cm)"],iris_df["petal length (cm)"],c=iris_df["target"])
axs[1][2].scatter(iris_df["sepal width (cm)"],iris_df["petal width (cm)"],c=iris_df["target"])

#plotting feature boxplots
fig,axs = plt.subplots(2,2,sharex=True,sharey=True,figsize=(14,24))

sns.boxplot(y=iris_df["petal length (cm)"],x=iris_df["target"],ax = axs[0][0],hue=iris_df["target"])
sns.boxplot(y=iris_df["sepal length (cm)"],x=iris_df["target"],ax = axs[1][0],hue=iris_df["target"])

sns.boxplot(y=iris_df["petal length (cm)"],x=iris_df["target"],ax=axs[0][1],hue=iris_df["target"])
sns.boxplot(y=iris_df["sepal length (cm)"],x=iris_df["target"],ax=axs[1][1],hue=iris_df["target"])

#create training and testing datasets
x_cols = iris_df.columns[:-1]
y = iris_df["target"]
x = iris_df[x_cols]

X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)

#classification with K-Nearest Neighbours
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train,y_train)
knn.score(X_test,y_test)

#feature selection using backwards sequential selection for Knn
sfs = SequentialFeatureSelector(estimator=knn,n_features_to_select=3, direction="backward")
sfs.fit(X_train,y_train)
sfs.get_support() # returns array([True,  False,  True,  True])

#refitting the knn model
knn_x_train = X_train.iloc[:,1:]
knn_x_test = X_test.iloc[:,1:]

knn_sfs = KNeighborsClassifier(n_neighbors = 3)
knn_sfs.fit(knn_x_train,y_train)
knn_sfs.score(knn_x_test,y_test)

#classification report & confusion matrix for knn with/without feature selection
knn_y_pred = knn.predict(X_test)
knn_sfs_y_pred = knn_sfs.predict(knn_x_test)

print("K Nearest Neighbors without feature selection")
print(confusion_matrix(y_test,knn_y_pred))
print(classification_report(y_test,knn_y_pred))

print("K Nearest Neighbors with feature selection")
print(confusion_matrix(y_test,knn_sfs_y_pred))
print(classification_report(y_test,knn_sfs_y_pred))

#classification with SVC (Support Vector Classification)
#SVC with RBF Kernel
svc_rbf = SVC()
svc_rbf.fit(X_train,y_train)
svc_rbf.score(X_test,y_test)

#feature selection for svc_rbf
sfs = SequentialFeatureSelector(estimator=svc_rbf,n_features_to_select=3, direction="backward")
sfs.fit(X_train,y_train)
sfs.get_support() # returns array([False,  True,  True,  True])

#refitting the svc_rbf model
rbf_x_train = X_train.loc[:,["sepal length (cm)","petal length (cm)","petal width (cm)"]]
rbf_x_test = X_test.loc[:,["sepal length (cm)","petal length (cm)","petal width (cm)"]]

rbf_sfs = SVC()
rbf_sfs.fit(rbf_x_train,y_train)
print(rbf_sfs.score(rbf_x_test,y_test))

#classification report & confusion matrix for svc_rbf with/without feature selection
rbf_y_pred = svc_rbf.predict(X_test)
rbf_sfs_y_pred = rbf_sfs.predict(rbf_x_test)

print("SVC RBF without feature selection")
print(confusion_matrix(y_test,rbf_y_pred))
print(classification_report(y_test,rbf_y_pred))

print("SVC RBF with feature selection")
print(confusion_matrix(y_test,rbf_sfs_y_pred))
print(classification_report(y_test,rbf_sfs_y_pred))

#SVC with polynomial Kernel
svc_poly = SVC(kernel="poly")
svc_poly.fit(X_train,y_train)
print(svc_poly.score(X_test,y_test))

#feature selection for svc_rbf
sfs = SequentialFeatureSelector(estimator=svc_poly,n_features_to_select=3, direction="backward")
sfs.fit(X_train,y_train)
sfs.get_support() # returns array([False,  True,  True,  True])

#refitting the svc_rbf model
poly_x_train = X_train.iloc[:,1:]
poly_x_test = X_test.iloc[:,1:]

poly_sfs = SVC()
poly_sfs.fit(poly_x_train,y_train)
print(poly_sfs.score(poly_x_test,y_test))

#classification report & confusion matrix for svc_poly with/without feature selection
poly_y_pred = svc_poly.predict(X_test)
poly_sfs_y_pred = poly_sfs.predict(knn_x_test)

print("SVC poly without feature selection")
print(confusion_matrix(y_test,poly_y_pred))
print(classification_report(y_test,poly_y_pred))

print("SVC poly with feature selection")
print(confusion_matrix(y_test,poly_sfs_y_pred))
print(classification_report(y_test,poly_sfs_y_pred))

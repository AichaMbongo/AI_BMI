from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
import pickle
import numpy as np #working with arrays. 
import pandas as pd 
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
#data manipulation library that is necessary for every aspect of data analysis or machine learning.

bmi=pd.read_csv("details.csv")

feature_names=["Height","Weight"]
X=bmi[feature_names].values
y=bmi["Index"].values

#Spliting dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y,  random_state = 0)

# Instantiate learning model (k = 3)
classifier = KNeighborsClassifier(n_neighbors=3)

# Fitting the model
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

#Model Evaluation
#Building a Confusion Matrix
# cm = confusion_matrix(y_test, y_pred)
# cm
# scaler=MinMaxScaler()
# X_train=scaler.fit_transform(X_train)
# X_test=scaler.transform(X_test)

# knn=KNeighborsClassifier()
# knn.fit(X_train,y_train)

pickle.dump(classifier, open('model.pkl','wb'))
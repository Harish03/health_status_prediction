import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import svm

def readData():
    dataset = pd.read_csv('C:\\Users\\HARISH\\Desktop\\abc1.txt',sep = ',', header = None)
    #dataset.isnull.sum()
    print("Dataset length" , len(dataset))
    print("Dataset shape" ,dataset.shape)
    print("Dataset : ")
    #print(dataset.head())
    return dataset

def splitData(dataset):
    X = dataset.values[:,0:13]   
    Y = dataset.values[:,13:14]
    for i in range(0,len(Y)):
        if Y[i]>0:
            Y[i] = 1
        else:
            Y[i] = 0
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.3,random_state = 84)
    return X,Y,X_train,X_test,Y_train,Y_test

def trainModel(x_train,y_train):
    model_svm = svm.SVC(C = 1.0,gamma = 'auto',kernel = 'linear',random_state = 84)
    model_svm.fit(x_train,y_train)
    return model_svm

def prediction(X_test,model):
    y_predicted = model.predict(X_test)
    print('predicted values : ')
    print(y_predicted)
    return y_predicted

def cal_accuracy(y_test,y_pred):
    print("Confusion Matrix: ")
    print(confusion_matrix(y_test,y_pred))
    print("Accuracy : ", accuracy_score(y_test,y_pred)*100)
    
def main():
    dataset = readData()
    X,Y,X_train,X_test,Y_train,Y_test = splitData(dataset)
    model = trainModel(X_train,Y_train)
    
    print('using svm classifier')
    y_predicted = prediction(X_test,model)
    cal_accuracy(Y_test,y_predicted)
    
main()
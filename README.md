# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas.
2. Calculate the null values present in the dataset and apply label encoder.
3. Determine test and training data set and apply decison tree regression in dataset.
4. Calculate Mean square error,data prediction and r2.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Abisheik Raj.J
RegisterNumber: 212224230006 
*/
```
```
import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()

data.info

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
y=data[["Salary"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])
```

## Output:
### DATA HEAD:
![image](https://github.com/user-attachments/assets/0bb68497-49af-4a4f-b3a0-34d0cd93c25c)


### DATA INFO:
![image](https://github.com/user-attachments/assets/afc8bd44-0372-417f-83e0-b2d32553957c)



### ISNULL() AND SUM():

![image](https://github.com/user-attachments/assets/6bce5a9c-c149-4d56-b17f-bb688367f7ca)

### DATA HEAD FOR SALARY:

![image](https://github.com/user-attachments/assets/a82018c5-c1ad-4eb6-9874-5b388a05f96e)

### MEAN SQUARED ERROR:

![image](https://github.com/user-attachments/assets/7946a4c6-e02b-4977-92b9-6f1d810367f4)

### R2 VALUE:

![image](https://github.com/user-attachments/assets/b7d0d69d-71dc-46ed-80dd-1b663cf658bd)

### DATA PREDICTION:
![image](https://github.com/user-attachments/assets/81d133d7-7d08-4110-923d-f6f89b05365c)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.

# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Use the standard libraries in python.
2. Set variables for assigning dataset values.
3. Import LinearRegression from the sklearn.
4. Assign the points for representing the graph.
5. Predict the regression for marks by using the representation of graph.
6. Compare the graphs and hence we obtain the LinearRegression for the given datas.

## Program:
```python
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Chandhuru S
RegisterNumber: 212220220007
*/



import pandas as pd
import numpy as np
df=pd.read_csv('student_scores.csv')
print(df)

X=df.iloc[:,:-1].values
Y=df.iloc[:,1].values
print(X,Y)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)

Y_pred=reg.predict(X_test)
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error , mean_squared_error

plt.scatter(X_train,Y_train,color='green')
plt.plot(X_train,reg.predict(X_train),color='purple')
plt.title(' Training set (Hours Vs Scores)')
plt.xlabel('Hours')
plt.ylabel('Scores')

plt.scatter(X_test,Y_test,color='green')
plt.plot(X_test,reg.predict(X_test),color='purple')
plt.title(' Training set (Hours Vs Scores)')
plt.xlabel('Hours')
plt.ylabel('Scores')

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print('RMSE = ',rmse)
```

## Output:

![c1](https://github.com/ChandhuruS/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/123381860/6eb28f78-9226-48bd-9521-9cbde08ff879)
![c2](https://github.com/ChandhuruS/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/123381860/112a0ace-d842-42f7-a1fc-ddc236190f48)
![c3](https://github.com/ChandhuruS/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/123381860/12a812a2-1345-47c0-a6a8-356edf59ecb1)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.

# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: thamizarasan.s
RegisterNumber:  212223220116
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:
![image](https://github.com/thamizh610/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150418511/40534d9a-2d78-41aa-b407-200418efd42f)
![image](https://github.com/thamizh610/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150418511/a8c8fecd-598e-4f66-b79e-e80e831fca89)

![image](https://github.com/thamizh610/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150418511/4356220d-313d-433b-af90-977864ed75a7)
![image](https://github.com/thamizh610/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150418511/b28a2224-722e-4ed7-8600-758c7b78058f)
![image](https://github.com/thamizh610/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150418511/4184b84f-286e-4c98-b13a-5636246770fc)
![image](https://github.com/thamizh610/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150418511/fee43500-b8de-4378-adb1-4bdda8810161)
![image](https://github.com/thamizh610/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150418511/08beac79-fb97-4cb3-bb07-cdbdfc9555b6)
![image](https://github.com/thamizh610/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150418511/f485ff40-2ae5-48ea-b723-c6671e596759)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.

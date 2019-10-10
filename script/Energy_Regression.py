#Anthony Windmon - Regression for Energy Efficiency Dataset

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import cross_validate

df = pd.read_csv('C:\\Users\\awindmon\\Desktop\\Projects\\ENB2012_data_two.csv')
print(df)

#visualizing data
plt.scatter(df.X1,df.X2, color = 'red', marker = '+') #we plot
plt.show()

#building linear model with Y1 as target
model = linear_model.LinearRegression()
print('-----------------TARGET = Y1--------------------')
model.fit(df[['X1','X2','X3','X4','X5','X6','X7','X8']], df.Y1)
results = model.predict([[0.98, 514.0, 294.0, 110.00, 3.5, 2, 0.0, 4]])
print('Heating Load =', results)

#building linear model with Y2 as target
print('-----------------TARGET = Y2--------------------')
model.fit(df[['X1','X2','X3','X4','X5','X6','X7','X8']], df.Y2)
results_two = model.predict([[0.98, 514.0, 294.0, 110.00, 3.5, 2, 0.0, 4]])
print('Cooling Load =', results_two)

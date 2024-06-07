import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LinearRegression # using a multiple linear regression model
from scipy.optimize import curve_fit

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

cols = ['Temperature','CO2 Emissions','Sea Level Rise', 'Precipitation', 'Humidity', 'Wind Speed']
df = pd.read_csv("/kaggle/input/co-and-greenhouse-gas-emissions/1- temperature-anomaly.csv")
df.head()
print(df[['Year']])


# define model training data 
X = df[['Year']]
y = df[['Global average temperature anomaly relative to 1961-1990']]

#print(X)

# set up the model
model = LinearRegression(fit_intercept = True)

# train the model
model.fit(X, y)

#X = 'Global average temperature anomaly relative to 1961-1990'
y_pred = model.predict(X)
plt.plot(X, y_pred, color='red') # line of best fit

# Graph our data and the line of best fit.
# Format: y = mx + b
plt.scatter(X, y)
plt.xlabel('Year') # set the labels of the x and y axes
plt.ylabel('Relative global average temperature to 1961-199')
plt.show()

# Intrerpet the model by identifiying the b (intercept) and m (slope) 
print("intercept (b) = ", model.intercept_)
print("slope (m) = ", model.coef_)

# Test our model by asking the user for an input year, and output the relative global temperature.
input_year = int(input("Enter a year for which you want the global temperature relative to 1961-1990: "))
output = (input_year * model.coef_) + model.intercept_
print("The relative global temperature for the year you entered is: ", output)
'''
# Sample data
x = np.array(df['Year'])
y = np.array(df['Global average temperature anomaly relative to 1961-1990'])  # Exponential growth data

# Define the exponential function
def exponential_function(x, a, b):
    return a * np.exp(b * x)

# Fit the exponential curve to the data
params, covariance = curve_fit(exponential_function, x, y)

# Extract the fitted parameters
a, b = params

# Plot the original data and the fitted curve
plt.scatter(x, y, label='Data')
plt.plot(x, exponential_function(x, a, b), color='red', label='Prediction Curve')
plt.xlabel('Year')
plt.ylabel('Global average temperature')
plt.legend()
plt.show()

# Print the fitted parameters
print("Fitted Parameters:")
print("a =", a)
print("b =", b)
'''
# check r^2
print('The multiple linear model had an R^2 of: %0.3f'%model.score(X, y))

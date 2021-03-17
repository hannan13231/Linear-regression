# Simple Linear Regression
# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd

# Importing the datasets

data_source = "http://bit.ly/w-data"
data = pd.read_csv(data_source)

print("Data is sucessfully implemented")
data.head(24)

data.isna()
data.info()

# Plotting the distribution of scores
data.plot(x = 'Hours', y = 'Scores', style = 'o', markerfacecolor = 'green')    
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score') 
plt.title('Graph for Hours studied VS Precentage Scored\n')
plt.legend()
plt.show()

X = data.iloc[:, :-1].values  
y = data.iloc[:, 1].values  

from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size = 0.4, random_state = 0)

from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 

print("Training complete.")

# Plotting the regression line
line = regressor.coef_*X+regressor.intercept_

# Plotting for the test data
plt.scatter(X, y, color = 'green')
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score') 
plt.title('Graph for Hours studied VS Precentage Scored\n')
plt.plot(X, line);
plt.show()

print(X_test) # Testing data - In Hours
y_pred = regressor.predict(X_test) # Predicting the scores

# Comparing Actual vs Predicted
y_diff = y_test - y_pred
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred, 'Difference': y_diff })  
df

# You can also test with your own data
x = float(input('Enter number of Hours a student studies:'))
hours = [[x]]
own_pred = regressor.predict(hours)
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))

from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred)) 
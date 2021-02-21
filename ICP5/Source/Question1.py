# Delete all the outlierdata for the GarageArea field

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from numpy.core import mean, std
from sklearn.model_selection import train_test_split
from sklearn import linear_model

# Read the input data
train = pd.read_csv('data.csv')

# handling missing value
data = train.select_dtypes(include=[np.number]).interpolate().dropna()
y = np.log(train.SalePrice)  # Build a linear model
X = data.drop(['SalePrice', 'Id'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=.33)                               # Split the data into test and training sets

lr = linear_model.LinearRegression()                                    # Training the multiple regression model based on training dataset
model = lr.fit(X_train, y_train)
print("R^2 is: \n", model.score(X_test, y_test))                        # Evaluate the performance using test dataset and visualize results
predictions = model.predict(X_test)
print('RMSE is: \n', mean_squared_error(y_test, predictions))
plt.style.use(style='ggplot')                                           # Set up the output screen


# Display the scatter plot of GarageArea and SalePrice
plt.scatter(train.GarageArea, train.SalePrice, color='red')
plt.xlabel('Garage Area')
plt.ylabel('SalePrice')
plt.show()

# Solution1 : Calculate summary statistics of mean and standard deviation
data_mean, data_std = mean(train.GarageArea), std(
    train.GarageArea)  # Identify outliers, calculate the lower and upper limit after which the other values become outliers
cut_off = data_std * 3
lower, upper = data_mean - cut_off, data_mean + cut_off
outliers = [x for x in train.GarageArea if
            x < lower or x > upper]  # Identify outliers outside the range calculated using mean
outlier_drop = train[(train.GarageArea < upper) & (train.GarageArea > lower)]  # Drop the identified outliers
print('Identified outliers: %d' % len(outliers))
print('Outliers using range method ', outliers)

# The scatter plot of GarageArea and SalePrice after deleting outliers
plt.scatter(outlier_drop.GarageArea, outlier_drop.SalePrice, color='green')
plt.xlabel('Garage Area')
plt.ylabel('SalePrice')
plt.show()

# Solution2 : Z-score method
threshold = 3
outlier = []
# Iterate through all the records and identify the outliers based on mean,std and threshold
for i in train.GarageArea:
    z = (i - data_mean) / data_std
    if z > threshold:
        outlier.append(i)
print('Outlier in dataset using Z- score method are : ', outlier)

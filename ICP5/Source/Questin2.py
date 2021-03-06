import pandas
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Reading the data from the CSV file
restaurant_data = pandas.read_csv("Restaurant_Revenue_Predictiondataset.csv")
y = np.log(restaurant_data.revenue)                                             # Build a linear model, drop the columns which are not numeric
X = restaurant_data.drop(['revenue', 'City Group', 'Type'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=50, test_size=0.13)                                      # Split the data into test and training sets

# Training the multiple regression model based on training dataset
lr = linear_model.LinearRegression()
model = lr.fit(X_train, y_train)

# Evaluate the performance using test dataset and visualize results
print("R^2 is: \n", model.score(X_test, y_test))
predictions = model.predict(X_test)
print('RMSE is: \n', mean_squared_error(y_test, predictions))

# show the plot of data and the linear regression model fit
sns.regplot(x=y_test, y=predictions, ci=None, color="b")
plt.show()

# Top 5 most correlated features to the target label(revenue)
corr_features = restaurant_data.select_dtypes(include=[np.number])
corr = corr_features.corr().abs()                                   # map features to their correlation values
corr[corr == 1] = 0                                                 # set equality (self correlation) as zero
corr_cols = corr['revenue'].sort_values(ascending=False)            # Find the max correlation for revenue column in ascending order
print(corr_cols.head(5))                                            # display the top 5 highly correlated features


# Heat map of the correlated features
corr = restaurant_data.corr()
kot = corr[corr >= .9]
plt.figure(figsize=(12, 8))
sns.heatmap(kot, cmap="Greens")
plt.show()


# Question 3 : Building the model only with the correlated features
# Reading the data from the CSV file
restaurant_data1 = pandas.read_csv('Restaurant_Revenue_Predictiondataset.csv')
y = np.log(restaurant_data1.revenue)  # Build a linear model, drop the columns which are not numeric
X = restaurant_data1.drop(
    ['revenue', 'City Group', 'Type', 'P1', 'P3', 'P4', 'P5', 'P7', 'P8', 'P9', 'P10', 'P11', 'P12', 'P14', 'P15',
     'P16', 'P17', 'P18', 'P19', 'P20', 'P21', 'P22', 'P23', 'P24', 'P25', 'P26', 'P27', 'P30', 'P31', 'P32', 'P33',
     'P34', 'P35', 'P36', 'P37'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=50, test_size=0.13)  # Split the data into test and training sets

# Training the multiple regression model based on training dataset
lr1 = linear_model.LinearRegression()
model1 = lr1.fit(X_train, y_train)

# Evaluate the performance using test dataset and visualize results
print("R^2 using only top 5 correlated features is: \n", model1.score(X_test, y_test))
predictions = model1.predict(X_test)
print('RMSE using only top 5 correlated features is: \n', mean_squared_error(y_test, predictions))

# show the plot of data and the linear regression model fit
sns.regplot(x=y_test, y=predictions, ci=None, color="b")
plt.show()


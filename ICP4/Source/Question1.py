# 1. find the correlation between ‘survived’(target column) and ‘sex’ column for the Titanic use case in class.
# Do you think we should keep this feature?

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the training data from the csv file
train_df = pd.read_csv('train.csv', usecols=['Survived', 'Sex'])
# Replace the Sex column with 1 or 0 for female or male, converting data to numerical type
train_df['Sex'] = train_df['Sex'].replace({'female': 1, 'male': 0})
print(train_df)

# Calculate the correlation between two columns
corr = train_df['Survived'].corr(train_df['Sex'])
print(corr)


sns.pairplot(train_df)
plt.show()

# Correlation coefficient shows that the 2 variables survived and sex are positively and highly correlated.
# The results show that the male sex survived more than the female.


g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Sex',  bins=20)
plt.show()





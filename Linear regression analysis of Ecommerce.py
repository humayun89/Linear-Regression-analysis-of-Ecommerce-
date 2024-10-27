import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import pylab
import scipy.stats as stats

# Load and explore the data
customers = pd.read_csv('C:/Users/Humayun/Downloads/Ecommerce Customers.csv')
print(customers.head())
customers.info()
print(customers.describe())

# Data visualizations
sns.jointplot(x='Time on Website', y='Yearly Amount Spent', data=customers, alpha=0.5)
plt.show()
sns.jointplot(x='Time on App', y='Yearly Amount Spent', data=customers, alpha=0.5)
plt.show()
sns.pairplot(customers, kind='scatter', plot_kws={'alpha': 0.4}, diag_kws={'alpha': 0.55, 'bins': 40})
plt.show()
sns.lmplot(x='Length of Membership', y='Yearly Amount Spent', data=customers, scatter_kws={'alpha': 0.3})
plt.show()

# Prepare the data for regression
X = customers[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
y = customers['Yearly Amount Spent']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit the model
lm = LinearRegression()
lm.fit(X_train, y_train)

# Display coefficients
cdf = pd.DataFrame(lm.coef_, X.columns, columns=['Coef'])
print(cdf)

# Display OLS summary
X_train_sm = sm.add_constant(X_train)  # Add constant for OLS
model = sm.OLS(y_train, X_train_sm)
model_fit = model.fit()
print(model_fit.summary())

# Predictions and evaluation
predictions = lm.predict(X_test)
sns.scatterplot(x=y_test, y=predictions)
plt.ylabel('Predictions')
plt.xlabel('Actual Yearly Amount Spent')
plt.title('Yearly Amount Spent vs. Model Predictions')
plt.show()

# Error metrics
print('Mean Absolute Error:', mean_absolute_error(y_test, predictions))
print('Mean Squared Error:', mean_squared_error(y_test, predictions))
print('Root Mean Squared Error:', math.sqrt(mean_squared_error(y_test, predictions)))

# Residual plot
residuals = y_test - predictions

# Check Seaborn version to use appropriate plotting function
sns_version = tuple(map(int, sns.__version__.split('.')))
if sns_version >= (0, 11, 0):
    sns.histplot(residuals, bins=30, kde=True)
else:
    sns.distplot(residuals, bins=30, kde=True)  # For older versions

plt.title('Distribution of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()

# Q-Q plot
stats.probplot(residuals, dist="norm", plot=pylab)
pylab.title('Q-Q Plot of Residuals')
pylab.show()

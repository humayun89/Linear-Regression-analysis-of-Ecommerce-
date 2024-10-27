import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
customers = pd.read_csv('C:/Users/Humayun/Downloads/Ecommerce Customers.csv')

# Display the first few rows
print(customers.head())

# Display the DataFrame info
customers.info()

# Display summary statistics
print(customers.describe())

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler


# In[2]:


df=pd.read_csv('AAPL.csv')
#Basic Data information
df.info()


# In[3]:


df


# In[4]:


# Display summary statistics
print(df.describe())


# In[5]:


# Check for missing values
print(df.isnull().sum())


# In[6]:


# Time Series analysis of the data and the target variable
# Convert 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Set 'Date' as the index
df.set_index('Date', inplace=True)

# Plot time series data
plt.figure(figsize=(12, 6))
plt.plot(df['Close'], label='Apple Stock Close Price')
plt.title('Apple Stock Close Price Over Time')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()


# In[7]:


df.corr()['Close']


# In[8]:


# Calculate correlation matrix

correlation_matrix = df.corr()

# Plot a heatmap of the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()


# In[9]:


# Scatter plots wit all h variables

plt.scatter(df['Volume'], df['Close'])
plt.title('Scatter Plot of Volume vs. Close Price')
plt.xlabel('Volume')
plt.ylabel('Close Price')
plt.show()

plt.scatter(df['Open'], df['Close'])
plt.title('Scatter Plot of Open vs. Close Price')
plt.xlabel('Open')
plt.ylabel('Close Price')
plt.show()

plt.scatter(df['High'], df['Close'])
plt.title('Scatter Plot of High vs. Close Price')
plt.xlabel('High')
plt.ylabel('Close Price')
plt.show()

plt.scatter(df['Low'], df['Close'])
plt.title('Scatter Plot of Low vs. Close Price')
plt.xlabel('Low')
plt.ylabel('Close Price')
plt.show()

plt.scatter(df['Adj Close'], df['Close'])
plt.title('Scatter Plot of Adjacent Close vs. Close Price')
plt.xlabel('Adjacent Close')
plt.ylabel('Close Price')
plt.show()


# In[10]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset (assuming 'AAPL.csv' contains the necessary columns)
df = pd.read_csv('AAPL.csv')

# Assume 'Close' is the column representing stock prices
data = df[['Open', 'High', 'Low', 'Adj Close']]  # Exclude 'Close' from predictors
target = df['Close'].values.reshape(-1, 1)

# Normalize the data
scaler_data = MinMaxScaler(feature_range=(0, 1))
scaler_target = MinMaxScaler(feature_range=(0, 1))

data_normalized = scaler_data.fit_transform(data)
target_normalized = scaler_target.fit_transform(target)

# Function to prepare the data for linear regression
def create_dataset(data, target, look_back=1):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), :])
        y.append(target[i + look_back, 0])
    return np.array(X), np.array(y)

# Set the look-back period (number of time steps to look back)
look_back = 20

# Create the dataset
X, y = create_dataset(data_normalized, target_normalized, look_back)

# Set the number of folds for cross-validation
k_folds = 5

# Create a k-fold cross-validation object
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

# Initialize the linear regression model
model = LinearRegression()

# Initialize lists to store MSE and R-squared scores
mse_scores = []
r2_scores = []

# Perform cross-validation and obtain scores
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Train the model on the training set
    model.fit(X_train.reshape(-1, look_back * data.shape[1]), y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test.reshape(-1, look_back * data.shape[1]))
    
    # Denormalize the predictions and actual values
    y_pred_denormalized = scaler_target.inverse_transform(y_pred.reshape(-1, 1))
    y_test_denormalized = scaler_target.inverse_transform(y_test.reshape(-1, 1))
    
    # Calculate Mean Squared Error (MSE) on the test set
    mse_test = mean_squared_error(y_test_denormalized, y_pred_denormalized)
    
    # Calculate R-squared on the test set
    r2_test = r2_score(y_test_denormalized, y_pred_denormalized)
    
    mse_scores.append(mse_test)
    r2_scores.append(r2_test)

# Print the mean and standard deviation of the MSE and R-squared scores
print(f'Mean MSE: {np.mean(mse_scores)}')
print(f'Standard Deviation MSE: {np.std(mse_scores)}')
print(f'Mean R-squared: {np.mean(r2_scores)}')
print(f'Standard Deviation R-squared: {np.std(r2_scores)}')

# No need to train the model again on the entire dataset after cross-validation

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Make predictions on the test set
y_pred = model.predict(X_test.reshape(-1, look_back * data.shape[1]))

# Denormalize the predictions and actual values
y_pred_denormalized = scaler_target.inverse_transform(y_pred.reshape(-1, 1))
y_test_denormalized = scaler_target.inverse_transform(y_test.reshape(-1, 1))

# Calculate Mean Squared Error (MSE) on the test set
mse_test = mean_squared_error(y_test_denormalized, y_pred_denormalized)

# Calculate R-squared on the test set
r2_test = r2_score(y_test_denormalized, y_pred_denormalized)

print(f'Mean Squared Error (MSE) on Test Set: {mse_test}')
print(f'R-squared on Test Set: {r2_test}')

# Plot actual vs. predicted values on the test set
# Plot actual vs. predicted values on the test set

df['Date'] = pd.to_datetime(df['Date'])

# Plot for actual values and the predictions.
plt.figure(figsize=(12, 6))
plt.plot(df['Date'][-len(y_test_denormalized):], y_test_denormalized, label='Actual Closing Price', color='blue')
plt.plot(df['Date'][-len(y_test_denormalized):], y_pred_denormalized, label='Predicted Closing Price', color='orange')
plt.title('Apple Stock Close Price Over Time')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()


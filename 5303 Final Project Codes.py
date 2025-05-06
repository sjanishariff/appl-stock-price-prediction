#!/usr/bin/env python
# coding: utf-8

# # Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler


# # Load Data and Descriptive Statistics

# In[2]:


df = pd.read_csv('AAPL.csv')


# In[3]:


df


# In[4]:


df.info()


# In[5]:


print(df.head())


# In[6]:


print(df.describe())


# # Check for Missing Values

# In[7]:


print(df.isnull().sum())


# # Time Series Analysis

# In[8]:


df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
plt.figure(figsize=(12, 6))
plt.plot(df['Close'], label='Apple Stock Close Price')
plt.title('Apple Stock Close Price Over Time')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()
print(df.corr()['Close'])


# # Correlation Matrix

# In[9]:


correlation_matrix = df.corr()
plt.figure(figsize=(15, 11))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()


# # Scatter Plots

# In[10]:


plt.scatter(df['Volume'], df['Close'])
plt.title('Scatter Plot of Volume vs. Close Price')
plt.xlabel('Volume')
plt.ylabel('Close Price')
plt.show()


# In[11]:


plt.scatter(df['Open'], df['Close'])
plt.title('Scatter Plot of Open vs. Close Price')
plt.xlabel('Open')
plt.ylabel('Close Price')
plt.show()


# In[12]:


plt.scatter(df['High'], df['Close'])
plt.title('Scatter Plot of High vs. Close Price')
plt.xlabel('High')
plt.ylabel('Close Price')
plt.show()


# In[13]:


plt.scatter(df['Low'], df['Close'])
plt.title('Scatter Plot of Low vs. Close Price')
plt.xlabel('Low')
plt.ylabel('Close Price')
plt.show()


# In[14]:


plt.scatter(df['Adj Close'], df['Close'])
plt.title('Scatter Plot of Adj Close vs. Close Price')
plt.xlabel('Adj Close')
plt.ylabel('Close Price')
plt.show()


# #  Linear Regression Model Training and Evaluation

# In[15]:


# Reload the dataset as the 'Date' index was modified earlier
df = pd.read_csv('AAPL.csv')
# Selecting features and target
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

    # Calculate Mean Squared Error (MSE) and R-squared on the test set
    mse_test = mean_squared_error(y_test_denormalized, y_pred_denormalized)
    r2_test = r2_score(y_test_denormalized, y_pred_denormalized)

    mse_scores.append(mse_test)
    r2_scores.append(r2_test)

# Print the mean and standard deviation of the MSE and R-squared scores
mean_mse = np.mean(mse_scores)
std_mse = np.std(mse_scores)
mean_r2 = np.mean(r2_scores)
std_r2 = np.std(r2_scores)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train the model on the training set
model.fit(X_train.reshape(-1, look_back * data.shape[1]), y_train)

# Make predictions on the test set
y_pred = model.predict(X_test.reshape(-1, look_back * data.shape[1]))

# Denormalize the predictions and actual values
y_pred_denormalized = scaler_target.inverse_transform(y_pred.reshape(-1, 1))
y_test_denormalized = scaler_target.inverse_transform(y_test.reshape(-1, 1))

# Calculate Mean Squared Error (MSE) and R-squared on the test set
mse_test = mean_squared_error(y_test_denormalized, y_pred_denormalized)
r2_test = r2_score(y_test_denormalized, y_pred_denormalized)


# # Plotting Predictions vs. Actual Values

# In[16]:


# Convert 'Date' column to datetime for plotting purposes
df['Date'] = pd.to_datetime(df['Date'])
# Plot for actual values and the predictions
plt.figure(figsize=(12, 6))
plt.plot(df['Date'].iloc[-len(y_test_denormalized):], y_test_denormalized, label='Actual Closing Price', color='blue')
plt.plot(df['Date'].iloc[-len(y_test_denormalized):], y_pred_denormalized, label='Predicted Closing Price', color='orange')
plt.title('Apple Stock Close Price Over Time - Actual vs Predicted')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()


# # Model Performance Summary

# In[17]:


mean_mse, std_mse, mean_r2, std_r2, mse_test, r2_test


# In[ ]:





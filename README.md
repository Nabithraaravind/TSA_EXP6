# Ex.No: 6               HOLT WINTERS METHOD
### Date: 20.10.2025
### Name: A.Nabithra


### AIM:

### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions
### PROGRAM:
#  Importing necessary modules
```
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
```
# Load the dataset,perform data exploration
```
data = pd.read_csv('/content/Tesla.csv', parse_dates=['Date'], index_col='Date')
print(data.columns)
data.head()
```
# Resample and plot data
```
data_monthly = data.resample('MS').sum()   
data_monthly.head()
```
#  Scale the data and check for seasonality
```
data_monthly.plot()
scaler = MinMaxScaler()
scaled_data = pd.Series(scaler.fit_transform(data_monthly['Close'].values.reshape(-1, 1)).flatten(), index=data_monthly.index)
scaled_data.plot()

 from statsmodels.tsa.seasonal import seasonal_decompose
 decomposition = seasonal_decompose(data_monthly['Close'], model="additive")
 decomposition.plot()
 plt.show()
```
# Split test,train data,create a model using Holt-Winters method, train with train data and Evaluate the model predictions against test data
 ```
 scaled_data=scaled_data+1 # multiplicative seasonality cant handle non postive values, ye
 train_data = scaled_data[:int(len(scaled_data) * 0.8)]
 test_data = scaled_data[int(len(scaled_data) * 0.8):]
 model_add = ExponentialSmoothing(train_data, trend='add', seasonal='mul').fit()
 test_predictions_add = model_add.forecast(steps=len(test_data))
 ax=train_data.plot()

test_predictions_add.plot(ax=ax)
test_data.plot(ax=ax)
ax.legend(["train_data", "test_predictions_add","test_data"])
ax.set_title('Visual evaluation')
np.sqrt(mean_squared_error(test_data, test_predictions_add))
np.sqrt(scaled_data.var()),scaled_data.mean()
```
# Apply the same scaling and transformation as used for the test/train split
```
final_scaled_data = pd.Series(scaler.fit_transform(data_monthly['Close'].values.reshape(-1, 1)).flatten(), index=data_monthly.index)
final_scaled_data = final_scaled_data + 1
```
#  Create teh final model and predict future data and plot it
```
final_model = ExponentialSmoothing(final_scaled_data, trend='add', seasonal='mul', seasonal_periods=12).fit()
final_predictions_scaled = final_model.forecast(steps=int(len(data_monthly)/4)) #for next year
```
# Inverse transform the predictions back to the original scale
```
final_predictions = pd.Series(scaler.inverse_transform(final_predictions_scaled.values.reshape(-1, 1) - 1).flatten(), index=final_predictions_scaled.index)

ax=data_monthly['Close'].plot()
final_predictions.plot(ax=ax)
ax.legend(["data_monthly", "final_predictions"])
ax.set_xlabel('Date')
ax.set_ylabel('Close Price')
ax.set_title('Tesla Stock Price Prediction')
```
### OUTPUT:

<img width="836" height="375" alt="image" src="https://github.com/user-attachments/assets/52ffe9bd-e156-47cd-b08a-6f129d6a7345" />
<img width="813" height="469" alt="image" src="https://github.com/user-attachments/assets/582bca44-11c5-4771-84cc-51f8243db600" />
<img width="814" height="654" alt="image" src="https://github.com/user-attachments/assets/a6503456-2cea-4759-b736-2f101dc9fb84" />

Decomposed plot:

<img width="829" height="629" alt="image" src="https://github.com/user-attachments/assets/64f56fc1-8739-4280-9311-16ce84976111" />

 Test prediction:
 
<img width="781" height="605" alt="image" src="https://github.com/user-attachments/assets/beb5de37-1d02-4396-ad82-ea6f1346eb97" />

 Final prediction:
 
<img width="829" height="672" alt="image" src="https://github.com/user-attachments/assets/150bf549-cb6f-4736-9130-152888ac4b5a" />


### RESULT:
Thus the program run successfully based on the Holt Winters Method model.

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

## 1. Data Loading and Preprocessing
print("Loading data...")
df = pd.read_csv('final_dataset.csv')
df.rename(columns={'Date': 'day', 'Month': 'month', 'Year': 'year'}, inplace=True)
df['Date_Time'] = pd.to_datetime(df[['year', 'month', 'day']])
df = df.set_index('Date_Time').sort_index()
df = df.drop(['day', 'month', 'year', 'Days'], axis=1)
for col in df.select_dtypes(include=np.number).columns:
    mean, std = df[col].mean(), df[col].std()
    df[col] = df[col].clip(lower=mean - 3*std, upper=mean + 3*std)
print("Data preprocessed successfully.")

## 2. Feature Engineering
print("Performing feature engineering...")
# Time-based features
df['day_of_year'] = df.index.dayofyear
df['month'] = df.index.month
df['quarter'] = df.index.quarter

# Lag features
for col in ['PM2.5', 'PM10', 'NO2', 'AQI']:
    for lag in range(1, 4):
        df[f'{col}_lag_{lag}'] = df[col].shift(lag)

### --- SOLUTION 1: ADD ADVANCED ROLLING FEATURES --- ###
# Create 3-day rolling averages to help the model capture trends
for col in ['PM2.5', 'PM10', 'NO2', 'AQI']:
    df[f'{col}_roll_mean_3'] = df[col].shift(1).rolling(window=3).mean()
### --------------------------------------------------- ###

df.dropna(inplace=True)
print("Feature engineering complete.")

## 3. Model Training with Final Refinements
print("Starting XGBoost model training with final refinements...")
X = df.drop('AQI', axis=1)
y = df['AQI']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Initialize the model with final refined hyperparameters
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    tree_method='hist',
    device='cuda',
    ### --- SOLUTION 2: FINAL PARAMETER REFINEMENTS --- ###
    n_estimators=2000,
    learning_rate=0.02,
    max_depth=4,                # Further reduced depth to combat overfitting
    subsample=0.7,              # Increased randomness
    colsample_bytree=0.7,       # Increased randomness
    gamma=0.1,                  # Added stronger regularization
    ### ----------------------------------------------- ###
    reg_alpha=0.1,
    random_state=42,
    early_stopping_rounds=50
)

# Fit the model
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=100
)

## 4. Evaluation
print("\nEvaluating final refined model performance...")
y_pred = xgb_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R-squared (R²): {r2:.4f}")

## 5. Visualization of Results
# Graph 1: Actual vs. Predicted AQI
print("Visualizing prediction results...")
plt.figure(figsize=(15, 7))
plt.plot(y_test.index, y_test, label='Actual AQI', color='blue', alpha=0.7)
plt.plot(y_test.index, y_pred, label='Predicted AQI', color='red', linestyle='--')
plt.title('AQI Forecasting: Actual vs. Predicted (Final Model)')
plt.xlabel('Date')
plt.ylabel('AQI')
plt.legend()
plt.grid(True)
plt.show()

# Graph 2: Training & Validation Loss Curve
print("Visualizing model accuracy during training...")
results = xgb_model.evals_result()
epochs = len(results['validation_0']['rmse'])
x_axis = range(0, epochs)
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(x_axis, results['validation_0']['rmse'], label='Train RMSE')
ax.plot(x_axis, results['validation_1']['rmse'], label='Test (Validation) RMSE')
ax.legend()
plt.ylabel('RMSE')
plt.xlabel('Boosting Round (Epoch)')
plt.title('XGBoost Training and Validation Loss (Final Model)')
plt.grid(True)
plt.show()
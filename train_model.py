import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import joblib  # For saving the model

# Step 1: Load the Stock Data
stock_data = pd.read_csv("stock_data.csv", index_col=0, parse_dates=True)

# Step 2: Feature Engineering (Previous day's closing price)
stock_data['Prev Close'] = stock_data['Close'].shift(1)  # Create the lag feature
stock_data = stock_data.dropna()  # Drop rows with NaN values (created by shifting)

# Step 3: Define Features and Target
X = stock_data[['Prev Close']]  # Feature (Previous day's Close price)
y = stock_data['Close']  # Target (Today's Close price)

# Step 4: Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Make Predictions on the Test Set
y_pred = model.predict(X_test)

# Step 7: Evaluate the Model
mae = mean_absolute_error(y_test, y_pred)  # Mean Absolute Error
r2 = r2_score(y_test, y_pred)  # R² Score

# Print evaluation metrics
print(f"Mean Absolute Error: {mae}")
print(f"R² Score: {r2}")

# Step 8: Visualize the Results
plt.figure(figsize=(12, 5))
plt.plot(y_test.index, y_test, label="Actual Prices", color='blue')
plt.plot(y_test.index, y_pred, label="Predicted Prices", color='red', linestyle='--')
plt.title('Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Stock Price (USD)')
plt.legend()
plt.show()

# Step 9: Save the Trained Model
joblib.dump(model, 'stock_price_predictor.pkl')
print("Model saved to 'stock_price_predictor.pkl'.")

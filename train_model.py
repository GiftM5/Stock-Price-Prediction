import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import joblib  # For saving the model

# Load the Stock Data
stock_data = pd.read_csv("stock_data.csv", index_col=0, parse_dates=True)

# Feature Engineering (Previous day's closing price)
stock_data['Prev Close'] = stock_data['Close'].shift(1)
stock_data = stock_data.dropna()
# Define Features and Target
X = stock_data[['Prev Close']]
y = stock_data['Close']

# Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Make Predictions on the Test Set
y_pred = model.predict(X_test)

# Evaluate the Model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)  

# Print evaluation metrics
print(f"Mean Absolute Error: {mae}")
print(f"R² Score: {r2}")

# Visualize the Results
plt.figure(figsize=(12, 5))
plt.plot(y_test.index, y_test, label="Actual Prices", color='blue')
plt.plot(y_test.index, y_pred, label="Predicted Prices", color='red', linestyle='--')
plt.title('Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Stock Price (USD)')
plt.xticks(rotation=45, ha="right")
plt.legend()
plt.tight_layout()
plt.show()

# Save the Trained Model
joblib.dump(model, 'stock_price_predictor.pkl')
print("Model saved to 'stock_price_predictor.pkl'.")

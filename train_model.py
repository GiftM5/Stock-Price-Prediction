import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import joblib  # For saving the model

# Load the Stock Data
import pandas as pd

# Load the data again (skip the first two rows and set 'Date' as the index)
stock_data = pd.read_csv("stock_data.csv", header=2, parse_dates=[0], index_col=0)

# Rename the columns for clarity
stock_data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

# Check the data after renaming the columns
print(stock_data.head())



# Feature Engineering (Previous day's closing price)
stock_data['Prev Close'] = stock_data['Close'].shift(1)
stock_data = stock_data.dropna()
# Define Features and Target
X = stock_data[['Prev Close']]
y = stock_data['Close']

# Split Data into Training and Testing Sets
# Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression Model
# Train the Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Make Predictions on the Test Set
# Make Predictions on the Test Set
y_pred = model.predict(X_test)
y_test = pd.to_numeric(y_test, errors='coerce') 
y_pred = pd.to_numeric(y_pred, errors='coerce')

# Now apply plt.ylim after ensuring both are numeric
plt.ylim([min(y_test.min(), y_pred.min()) - 10, max(y_test.max(), y_pred.max()) + 10])


# Evaluate the Model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)  

# Print evaluation metrics
print(f"Mean Absolute Error: {mae}")
print(f"R² Score: {r2}")

# Visualize the Results
print(stock_data['Close'].isnull().sum())


stock_data = stock_data.dropna(subset=['Close'])


plt.figure(figsize=(10, 6))
plt.plot(stock_data.index, stock_data['Close'], label='Close Price', color='b')
plt.title('Stock Close Price Over Time')
plt.xlabel('Date')
plt.ylabel('Close Price (USD)')
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Save the Trained Model
# Save the Trained Model
joblib.dump(model, 'stock_price_predictor.pkl')
print("Model saved to 'stock_price_predictor.pkl'.")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load preprocessed data
df = pd.read_csv('nyc_taxi_preprocessed_data.csv')

# Define features and target
X = df[['trip_distance', 'fare_amount', 'passenger_count', 'pickup_longitude', 'pickup_latitude']]
y = df['total_amount']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'linear_regression_model.pkl')

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Plotting predictions vs actual values
import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Total Amount")
plt.ylabel("Predicted Total Amount")
plt.title("Actual vs Predicted Total Amount")
plt.show()



# Extract coefficients
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print(coefficients)

# Insights
print("Insights:")
print(f"The model's R^2 score of {r2:.2f} suggests that {r2*100:.2f}% of the variability in the total amount can be explained by the features used.")
print(f"Key features influencing the prediction are trip distance, fare amount, and passenger count.")


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

### planet b
# transit_indices = np.array([24, 28, 35, 127, 135, 144, 337, 339, 340, 341, 342, 343, 430, 432])
# transit_times = np.array([2218.0041, 2249.6955, 2305.1505, 3033.8604, 3097.2502, 3168.5368, 4697.28077082, 
#                           4713.12175491, 4721.04224696, 4728.96273901, 4736.88323106, 4744.80372311, 5433.88653139, 5449.72751549])

### planet c
transit_indices = np.array([84,99,227,287])
transit_times = np.array([3019.4774, 3197.8645, 4720.12794251, 5430.91037079])

# Reshape the data for sklearn
X = transit_indices.reshape(-1, 1)
y = transit_times

# Create and fit the model
model = LinearRegression()
model.fit(X, y)

# Get the coefficients
slope = model.coef_[0]
intercept = model.intercept_

# Predict the y values
y_pred = model.predict(X)

# Calculate residuals
residuals = y - y_pred

# Plot the original data and the fitted line
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(transit_indices, transit_times, color='blue', label='Original data')
plt.plot(transit_indices, y_pred, color='red', label='Fitted line')
plt.xlabel('Transit Index')
plt.ylabel('Transit Time')
plt.title('Original Data with Fitted Line')
plt.legend()

# Plot the residuals (detrended data)
plt.subplot(1, 2, 2)
plt.scatter(transit_indices, residuals, color='green', label='Residuals')
plt.axhline(0, color='red', linestyle='--', label='Zero Line')
plt.xlabel('Transit Index')
plt.ylabel('Residuals (Detrended Transit Time)')
plt.title('Detrended Data')
plt.legend()

plt.tight_layout()
plt.show()

# Print the slope and intercept
print(f'Slope: {slope}')
print(f'Intercept: {intercept}')
# Print the residuals
print('Residuals:', residuals)
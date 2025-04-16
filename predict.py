import numpy as np
from sklearn.linear_model import LinearRegression
import random

# Data - Hum yahan 10 sample data points use kar rahe hain jisme weight aur age ka relationship diya gaya hai
# Example: Weight (in kg) vs Age (in years)
X = np.array([50, 60, 70, 80, 90, 100, 110, 120, 130, 140]).reshape(-1, 1)  # Weight
y = np.array([18, 22, 25, 30, 35, 40, 45, 50, 55, 60])  # Age

# Model create karte hain
model = LinearRegression()
model.fit(X, y)

# Prediction karna: Hum random weight value lenge aur uska age predict karenge
random_weight = random.randint(50, 140)  # Random weight between 50 and 140
predicted_age = model.predict([[random_weight]])

print(f"Predicted age for weight {random_weight} kg is: {predicted_age[0]:.2f} years")

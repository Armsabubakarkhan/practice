# deep_learning_example.py

import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# Training data: x → y = 2 * x
x_train = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]], dtype=float)
y_train = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20], dtype=float)

# Define a simple feedforward neural network
model = Sequential([
    Dense(64, activation="relu", input_shape=(1,)),
    Dense(64, activation="relu"),
    Dense(1)  # Output layer
])

# Compile the model
model.compile(optimizer="adam", loss="mean_squared_error")

# Train the model
model.fit(x_train, y_train, epochs=200, verbose=0)

# Test prediction
x_test = np.array([[12]], dtype=float)
prediction = model.predict(x_test)

print(f"Input: {x_test.flatten()[0]} → Predicted Output: {prediction.flatten()[0]:.2f}")

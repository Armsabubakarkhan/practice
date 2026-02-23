import numpy as np

x = np.array([[4],[5],[6],[7],[8],[9],[10]], dtype=float)
y = np.array([8, 10 , 12, 14, 16, 18, 20], dtype=float)

model.fit(x, y, epochs=200)

# Prediction
x_test = np.array([[12.0]], dtype=float)
print(model.predict(x_test))

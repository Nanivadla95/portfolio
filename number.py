import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the MNIST dataset
mnist = fetch_openml("mnist_784")
X, y = mnist.data / 255.0, mnist.target.astype(int)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train an MLP classifier
model = MLPClassifier(hidden_layer_sizes=(128,), max_iter=5, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
test_accuracy = model.score(X_test, y_test)
print(f'Test accuracy: {test_accuracy}')

# Make predictions
predictions = model.predict(X_test)

# Display some sample predictions
plt.figure(figsize=(8, 8))
for i in range(5):
  plt.subplot(1, 5, i + 1)
  for i in range(len(X_test)): 
        if 0 < i < len(X_test):
            plt.imshow(X_test[i].reshape(28, 28), cmap='binary', interpolation='nearest')
            plt.show()


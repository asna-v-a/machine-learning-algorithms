# K-Nearest Neighbors (KNN) Algorithm

## Overview

K-Nearest Neighbors (KNN) is a simple, versatile, and widely used algorithm for both classification and regression tasks. It is a non-parametric method, meaning it makes no assumptions about the underlying data distribution. KNN works by finding the 'K' closest data points to a given data point and making predictions based on these neighbors.

- **Classification**: For classification tasks, the label of the majority class among the K nearest neighbors is assigned to the data point.
- **Regression**: For regression tasks, the prediction is the average of the target values of the K nearest neighbors.

## Key Concepts

- **Distance Metric**: The most common distance metric used in KNN is the Euclidean distance, but other metrics like Manhattan, Minkowski, or cosine similarity can also be used depending on the data.
- **K (Number of Neighbors)**: The user defines the number of neighbors 'K'. A small K can lead to a noisy model, whereas a large K can smooth out the prediction, potentially underfitting the data.

## Algorithm Steps

1. **Choose the number of neighbors (K)**: Select the number of neighbors to consider for making predictions.
2. **Compute the distance**: Calculate the distance between the query point and all data points in the training set.
3. **Sort the distances**: Sort the distances in ascending order.
4. **Identify the K nearest neighbors**: Select the K data points with the smallest distances.
5. **Make prediction**: 
   - For classification, assign the class that is the most common among the K neighbors.
   - For regression, compute the average of the target values of the K neighbors.

## Hyperparameters

- **K**: The number of neighbors to consider.
- **Distance Metric**: The method used to calculate distance between data points (e.g., Euclidean, Manhattan).
- **Weights**: Weights can be assigned to neighbors. A common approach is to weight the contribution of each neighbor by its distance, where closer neighbors have more influence.

## Pros
- Simple and easy to understand.
- No training phase, making it fast to implement.
- Works well with small datasets and in situations where the decision boundary is highly non-linear.

## Cons
- Computationally expensive at prediction time, especially with large datasets.
- Performance degrades with high-dimensional data (curse of dimensionality).
- Sensitive to irrelevant or redundant features.

## Use Cases

- **Classification**: Used in image recognition, recommendation systems, and document classification.
- **Regression**: Applied in real-estate price prediction, stock market analysis, and weather forecasting.

## Example Code

Here’s an implementation of the KNN algorithm with **K = 5**, based on a small dataset:

```python
from sklearn.neighbors import KNeighborsClassifier

# Data points and their target labels
x1 = [7, 7, 3, 1]
y1 = [7, 4, 4, 4]
target = ['BAD', 'BAD', 'GOOD', 'GOOD']

# Combine features into a list of tuples
feature = list(zip(x1, y1))

# Create KNN model with K = 5
model = KNeighborsClassifier(n_neighbors=5)

# Train the model with features and targets
model.fit(feature, target)

# Predict the target label for a new point
prediction = model.predict([[3, 7]])

# Output the prediction
print(f"Prediction for point (3, 7): {prediction[0]}")

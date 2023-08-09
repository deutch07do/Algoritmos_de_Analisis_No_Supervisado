import numpy as np
from sklearn.cluster import KMeans

# Load the data
data = np.loadtxt("data.csv", delimiter=",")

# Create the KMeans object
kmeans = KMeans(n_clusters=5)

# Fit the model to the data
kmeans.fit(data)

# Get the cluster labels
labels = kmeans.predict(data)

# Print the cluster labels
print(labels)
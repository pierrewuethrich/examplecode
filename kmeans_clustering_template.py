"""
Created on Thu Oct 11 22:14:14 2018

@author: Wuethrich Pierre

Sample code for kmeans-clustering algorithm
"""

#Importing the necessary libraries 

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

#Creating fake data for illustration purposes and visualizing it

data = make_blobs(n_samples=200, n_features=2, centers=4, cluster_std=1.2,random_state=33)
plt.scatter(data[0][:,0],data[0][:,1],c=data[1],cmap='Set2')

#Using the k-means clustering algorithm

kmeans = KMeans(n_clusters=4)
kmeans.fit(data[0])

#Showing the centroids of the clusters and predicted labels

kmeans.cluster_centers_
kmeans.labels_

#We can show the difference between the 'actual' labels and predicted labels
plt.show()
f, (axis1, axis2) = plt.subplots(1, 2, sharey=True,figsize=(18,6))
axis1.set_title('K Means')
axis1.scatter(data[0][:,0],data[0][:,1],c=kmeans.labels_,cmap='rainbow')
axis2.set_title("Original")
axis2.scatter(data[0][:,0],data[0][:,1],c=data[1],cmap='rainbow')

plt.show()

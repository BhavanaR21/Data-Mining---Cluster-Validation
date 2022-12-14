# Data-Mining---Cluster-Validation
**Purpose**
To apply the cluster validation technique to data extracted from a provided data set.
**Technology Requirements**
Python 2.7. 
scikit-learn==0.21.2
pandas==0.25.1
Python pickle
**Project Description**
write a program, using Python, that takes a dataset and performs clustering. Using the provided training data set you will perform cluster validation to determine the amount of carbohydrates in each 
meal.  
**Directions**
There are two main parts to the process: 
1. Extract features from Meal data
2. Cluster Meal data based on the amount of carbohydrates in each meal
**Data:**
CGMData.csv
InsulinData.csv
**Extracting Ground Truth: **

Derive the max and min value of meal intake amount from the Y column of the Insulin data. Discretize 
the meal amount in bins of size 20. Consider each row in the meal data matrix that you generated in 
Project 2. Put them in the respective bins according to their meal amount label.
In total you should have n = (max-min)/20 bins.
**Performing clustering:**

Use the features in your Project 2 to cluster the meal data into n clusters. Use DBSCAN and KMeans. 
Report your accuracy of clustering based on SSE, entropy and purity metrics.




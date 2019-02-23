"""
This program applies k mean clustering to college data and finds the silhouette score.

Bonus Point - If there was with null values I would apply complete case analysis with the nearest neighbor assignment
for the null values. This method is common when there are more complete data than not. Since there seems to be no 'Age'
column in the csv file I cannot fill in values. I also think filling in the null values with the mean value of all the
ages is a viable option and that would be my go to method.
"""

# Import libraries
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.metrics import silhouette_score

NUMBER_OF_CLUSTERS = 5 # Number of clusters desired for k means clustering
CATEGORICAL_FEATURES = ['Private'] # Categorical features to dummify for standardization

# Read in raw csv college data with pandas in to perform the k means clustering on
raw_college_data = pd.read_csv("./College.csv", sep=",")

# Remove the colleges column as it is not needed for standardization.
del raw_college_data['Colleges']

# Collect unstandardized data from the raw college data. Drop the first column, which numbers the row.
# dummify the categorical feature column as a 0 or 1.
unstandardized_data = pd.get_dummies(
    raw_college_data,
    columns=CATEGORICAL_FEATURES,
    drop_first=True
)

# Standardize the columns that are useful
columns_to_standardize = [
  column for column in raw_college_data.columns
    if column not in CATEGORICAL_FEATURES
]

# First we will scale the data and then standardize the data.
data_to_standardize = unstandardized_data[columns_to_standardize]
data_to_standardize = data_to_standardize.astype(float)
scaler = preprocessing.StandardScaler().fit(data_to_standardize)
standardized_data = unstandardized_data.copy()
standardized_columns = scaler.transform(data_to_standardize)
standardized_data[columns_to_standardize] = standardized_columns

# Apply K means clustering to fit the now standardized data
model = KMeans(n_clusters=NUMBER_OF_CLUSTERS).fit(standardized_data)
unstandardized_data['cluster'] = model.predict(standardized_data)


# Print the summary of the
print('Cluster summary:')
cluster_summary = unstandardized_data.groupby(['cluster']).mean()
cluster_summary['count'] = unstandardized_data['cluster'].value_counts()
cluster_summary = cluster_summary.sort_values(by='count', ascending=False)
print(cluster_summary)

# Calculate the silhouette score for the clustering.
print("For n_clusters =", NUMBER_OF_CLUSTERS, "The average silhouette score is :",
      silhouette_score(standardized_data, unstandardized_data['cluster']))

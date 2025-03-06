from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from spatial_viz import spatial_viz
import pandas as pd


# Implement GMM model to generate clusters based on Latitude and Longitude.

def gmm(data, n_clusters):
    '''
    This function is used to cluster the data based on longitude and latitude.
    Input: data: the data to be clustered
        n_clusters: the number of clusters
    Output: the dataframe with a new column 'cluster' which contains the cluster labels of each data point
    '''
    # Standardize the data
    scaler = StandardScaler()
    data_std = scaler.fit_transform(data)
    # Use GNN to cluster the data
    gnn = GaussianMixture(n_components=n_clusters, random_state=0).fit(data_std)
    # Get the cluster labels
    labels = gnn.predict(data_std)
    # Add cluster labels as a new column to the dataframe
    data['cluster'] = labels
    return data  # return the entire dataframe instead of just labels

# df = pd.read_csv('data/crash_reporting_drivers_data_sanitized.csv')
# new_df = gnn(df[['Longitude', 'Latitude']], 9)
# df["cluster"] = new_df["cluster"]
# # print(new_df.head(50))

# spatial_viz(
#     df=df, 
#     hover_name = 'Municipality',
#     hover_data = ['Latitude', 'Longitude'], 
#     color_column = 'cluster',
#     title = "Clusters of Crashes in Montgomery County, MD"
#     )
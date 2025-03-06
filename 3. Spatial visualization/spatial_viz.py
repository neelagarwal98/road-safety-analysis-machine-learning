import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px

df = pd.read_csv('data/crash_reporting_drivers_data_sanitized.csv')
SHAPES_FILE = 'data/street_centerline_montgomery/street_centerline.shp'

# def spatial_viz(df, shp_file):
#     # Create a geopandas dataframe from the .shp file
#     gdf = gpd.read_file(shp_file)

#     # Create a geopandas dataframe from the df
#     gdf2 = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['Longitude'], df['Latitude']))

#     # Create a scatter plot using plotly
#     # fig = px.scatter_mapbox(gdf2, lat='Latitude', lon='Longitude', hover_name='Injury Severity', color_discrete_sequence=['red'], zoom=10, hover_data=['Weather', 'Surface Condition', 'Light','Driver Substance Abuse','Driver At Fault'],)
#     fig = px.scatter_mapbox(gdf2, lat='Latitude', lon='Longitude', 
#                             hover_name='Injury Severity', 
#                             color="Injury Severity", 
#                             color_discrete_sequence=px.colors.qualitative.Plotly, 
#                             zoom=10, 
#                             hover_data=['Weather', 'Surface Condition', 'Light','Driver Substance Abuse','Driver At Fault'])
#     fig.update_layout(mapbox_style='open-street-map')
#     fig.update_layout(title="Crashes in the County of Montgomery, State of Maryland, United States")
#     fig.show()

# spatial_viz(df, SHAPES_FILE)


def spatial_viz(df, shp_file=SHAPES_FILE, hover_name=None, hover_data=None, color_column=None, title="Crashes in the County of Montgomery, State of Maryland, United States"):
    # Create a geopandas dataframe from the .shp file
    gdf = gpd.read_file(shp_file)

    # Create a geopandas dataframe from the df
    gdf2 = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['Longitude'], df['Latitude']))

    if hover_name is None:
        hover_name = df.columns[1]  # Use the second column as hover_name
    if hover_data is None:
        hover_data = df.columns.tolist()  # Use all columns for hover_data
    if color_column is None:
        color_column = df.columns[1]  # Use the second column for color_column

    # Create a scatter plot using plotly
    fig = px.scatter_mapbox(gdf2, lat='Latitude', lon='Longitude', 
                            hover_name=hover_name, 
                            color=color_column, 
                            color_discrete_sequence=px.colors.qualitative.Plotly, 
                            zoom=10, 
                            hover_data=hover_data)
    fig.update_layout(mapbox_style='open-street-map')
    fig.update_layout(title=title)
    # fig.write_html('eda_plots/spatial_vizualization.html')
    fig.show()

# spatial_viz(df, SHAPES_FILE, 'Injury Severity', ['Weather', 'Surface Condition', 'Light','Driver Substance Abuse','Driver At Fault'], 'Injury Severity', "Crashes in the County of Montgomery, State of Maryland, United States")

# temp_df = df[df['Municipality'] == 'CHEVY CHASE VILLAGE']
# temp_df.dropna(subset=['Municipality'], inplace=True)

# df.dropna(subset=['Municipality'], inplace=True)
# spatial_viz(
#     df=df,
#     hover_name='Municipality',
#     hover_data='Latitude',
#     color_column='Municipality',
#     title="Clusters of Crashes based on Municipality in Montgomery County, MD"
# )
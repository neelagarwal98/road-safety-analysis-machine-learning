# Road Safety Analysis using Machine Learning (Clustering, Classification, Frequent Pattern Mining, Time Series Analysis - Prophet)
A data mining project exploring insights from crash reporting data gathered within Montgomery County, Maryland. Uncover patterns, analyze driver behavior, and enhance road safety through advanced analytics. #DataMining #RoadSafety #Analytics" 
___

## Instructions:
To run the project: 

### 1)  Get the dataset
Download the dataset (in<b>.csv</b>) from [here](https://catalog.data.gov/dataset/crash-reporting-drivers-data). Please keep the downloaded directory in the same repository.

### 2)  Install all the required packages
To run the project, install the required packages from requirements-conda.txt and requirements-pip.txt by running the following commands in your terminal:
```bash
conda install --file requirements-conda.txt
pip install -r requirements-pip.txt
```

### 3) Data Pre-Processing
Use the preprocess.ipynb to preprocess the data and generate a sanitized dataset in the form of csv file:

### 4) Exploratory Data Analysis
Use the eda.ipynb to visualize different plots.

### 5) Visualize the crash locations on a map
For Plotting the Crash Site Locations on a Map, run the spatial_viz.py file using the following command:
```bash
python3 spatial_viz.py
```

### 6) Find Cluster Specific Frequent Patterns
Run the **main.py** file using the following command:
```bash
python3 main.py
```
You can run the clustering and frequent pattern mining standalone by the following method:

#### i) Clustering: Gaussian Mixture Model
Run the gmm.py file. Cluster the dataset based on "Lattitude" and "Longitude" of crash sites. You can set the number of clusters using `NUMBER_OF_CLUSTERS` from **consts.py**.

#### ii) Frequent Pattern Mining: FPgrowth
Run the fpgrowth.py file. Set the parameters in for FPgrowth in **consts.py**

### 7) Injury Severity Classifier
Run the **random_forest.py** file using the following command:
```bash
python3 random_forest.py
```

### 8) Time Series Analysis using Prophet
Run the **time_series_analysis.py** file using the following command:
```bash
python3 time_series_analysis.py
```

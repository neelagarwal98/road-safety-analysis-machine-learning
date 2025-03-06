# Implement Random Forest Classification Algorithm to predict the severity of the crash.

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from spatial_viz import spatial_viz
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
from imblearn.over_sampling import SMOTE
from tqdm import tqdm
from datetime import datetime

print(f"Starting at {datetime.now()}...")

# Read the data
df = pd.read_csv('data/crash_reporting_drivers_data_sanitized.csv')

selected_cat_features = [
    "Collision Type", "Weather", "Surface Condition", "Light", "Traffic Control", "Driver Substance Abuse",
    "Non-Motorist Substance Abuse", "Driver At Fault",  "Circumstance", "Driver Distracted By",
    "Latitude", "Longitude"
]

# selected_cat_features = [
#     "Collision Type", "Vehicle Damage Extent", "Vehicle First Impact Location", "Vehicle Second Impact Location",
#     "Vehicle Body Type", "Vehicle Movement", "Vehicle Continuing Dir", "Vehicle Going Dir",
#     "Vehicle Year", "Vehicle Make", "Vehicle Model", "Equipment Problems", "Latitude", "Longitude"
# ]

# Create a StratifiedKFold object
skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
data = df[selected_cat_features].copy()

# Convert categorical features to numerical features
data = pd.get_dummies(data)

# Split the data into training and testing sets
X = data
y = df['Injury Severity']

# print("Unique values in y:")
# print(y.value_counts())

# Apply SMOTE to address class imbalance
smote = SMOTE(random_state=0)
X_resampled, y_resampled = smote.fit_resample(X, y)

# print("Unique values in y:")
# print(y_resampled.value_counts())

# Split the resampled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, random_state=0)

# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Create a random forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)

# Initialize lists to hold accuracy scores and confusion matrices for each fold
accuracy_scores = []
confusion_matrices = []

# Loop over each fold
for train_index, test_index in tqdm(skf.split(X_resampled, y_resampled)):
    # Split the data into training and testing sets
    X_train, X_test = X_resampled.iloc[train_index], X_resampled.iloc[test_index]
    y_train, y_test = y_resampled.iloc[train_index], y_resampled.iloc[test_index]

# for train_index, test_index in skf.split(X, y):
#     # Split the data into training and testing sets
#     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#     y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Train the classifier
    clf.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = clf.predict(X_test)

    # Calculate the accuracy of the classifier and add it to the list of accuracy scores
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)

    # Create a confusion matrix and add it to the list of confusion matrices
    cm = confusion_matrix(y_test, y_pred)
    confusion_matrices.append(cm)

print(f"Trained at {datetime.now()}...")

# Calculate the mean accuracy score
mean_accuracy = np.mean(accuracy_scores)
print(f"Mean Accuracy: {mean_accuracy}")

# Calculate the mean confusion matrix
mean_cm = np.mean(confusion_matrices, axis=0)
print(f"Mean Confusion Matrix:\n{mean_cm}")

# Visualize the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')

#### CHANGE THE NAME OF THE PLOT HERE ####

# Save the confusion matrix plot
plt.savefig('vehicle_factors_confusion_matrix.png')

# Show the plot
plt.show()

# Save the model
pickle.dump(clf, open('vehicle_factors_random_forest_model.pkl', 'wb'))

# Save the confusion matrix
np.save('vehicle_factors_confusion_matrix.npy', cm)
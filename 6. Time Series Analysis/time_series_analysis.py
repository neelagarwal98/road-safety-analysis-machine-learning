# Use Facebook's prophet package to analyze the Crash Time/Date column of the df. The column is converted to a datetime object and the date is set as the index. The column is then resampled to a daily frequency and the number of crashes per day is calculated. The prophet package is then used to fit a model to the data and make predictions for the next 365 days. The model is then plotted and the components of the model are plotted as well. The model is then cross validated using cross_validation and the results are plotted. Finally, the performance of the model is evaluated using performance_metrics and the results are plotted.
# Also split the data into training and testing sets and plot the results of the model on the testing set.

import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric


def time_series_analysis(df, column_name):
    # Convert the column to a datetime object and set the date as the index
    df[column_name] = pd.to_datetime(df[column_name], format='%Y-%m-%d %H:%M:%S')
    df = df.set_index(column_name)

    # Resample the data to a daily frequency and calculate the number of crashes per day
    df = df.resample('D').size().reset_index(name='Number of Crashes')
    df = df.rename(columns={column_name: 'ds', 'Number of Crashes': 'y'})  # Rename columns

    # Fit a model to the data and make predictions for the next 365 days
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)
    forecast = forecast[forecast['ds'] > df['ds'].max()]

    # Plot the model
    model.plot(forecast)
    plt.title(f"Daily Number of Crashes")
    plt.ylabel("Number of Crashes")
    plt.xlabel("Year")
    plt.legend()  # Add legend
    plt.show()

    # Plot the components of the model
    model.plot_components(forecast, figsize=(15, 10))
    plt.legend()  # Add legend
    plt.show()

    # Cross validate the model
    cross_validation_results = cross_validation(model, initial='730 days', period='180 days', horizon='365 days')
    cross_validation_results.head()
    performance_metrics_results = performance_metrics(cross_validation_results)
    performance_metrics_results.head()
    plot_cross_validation_metric(cross_validation_results, metric='mape')
    plt.title("Cross Validation Results")
    plt.legend()  # Add legend
    plt.show()

    # Split the data into training and testing sets
    train = df[:int(len(df) * 0.8)]
    test = df[int(len(df) * 0.8):]

    # Fit a model to the training data and make predictions for the testing data
    model = Prophet()
    model.fit(train)
    future = model.make_future_dataframe(periods=len(test))
    forecast = model.predict(future)

    # Plot the results of the model on the testing data
    plt.plot(test['ds'], test['y'], label='Predicted')  # Use renamed columns
    plt.plot(forecast['ds'], forecast['yhat'], label='Actual')  # Use renamed columns
    plt.title(f"Number of Crashes per Day in {column_name}")
    plt.ylabel("Number of Crashes")
    plt.xlabel("Date")
    plt.legend()  # Add legend
    plt.show()


df = pd.read_csv('data/crash_reporting_drivers_data_sanitized.csv')
time_series_analysis(df, 'Crash Date/Time')
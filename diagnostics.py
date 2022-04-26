import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle
import subprocess
import sys

##################Load config.json and get environment variables
with open("config.json", "r") as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config["output_folder_path"])
test_data_path = os.path.join(config["test_data_path"])
prod_deployment_path = os.path.join(config["prod_deployment_path"])

##################Function to get model predictions
def model_predictions(testdata):
    # read the deployed model and a test dataset, calculate predictions
    with open(f"{prod_deployment_path}/trainedmodel.pkl", "rb") as file:
        model = pickle.load(file)

    X = testdata.loc[
        :, ["lastmonth_activity", "lastyear_activity", "number_of_employees"]
    ].values.reshape(-1, 3)

    y = testdata["exited"].values.reshape(-1, 1).ravel()

    predicted = model.predict(X)

    # return value should be a list containing all predictions
    return list(predicted)


##################Function to get summary statistics
def dataframe_summary():
    # calculate summary statistics here

    numerical_columns = [
        "lastmonth_activity",
        "lastyear_activity",
        "number_of_employees",
    ]
    thedata = pd.read_csv(f"{dataset_csv_path}/finaldata.csv")

    summary = []
    for column in numerical_columns:
        summary.append([column, "mean", thedata[column].mean()])
        summary.append([column, "median", thedata[column].median()])
        summary.append([column, "standard deviation", thedata[column].std()])

    # return value should be a list containing all summary statistics
    return summary


def dataframe_missing_data():
    # calculate missing data here

    thedata = pd.read_csv(f"{dataset_csv_path}/finaldata.csv")
    nas = list(thedata.isna().sum())
    napercents = [nas[i] / len(thedata.index) for i in range(len(nas))]

    # return value should be a list with the percent of NA values
    return napercents


##################Function to get timings
def execution_time():
    # calculate timing of training.py and ingestion.py
    starttime_ingestion = timeit.default_timer()
    os.system("python ingestion.py")
    timing_ingestion = timeit.default_timer() - starttime_ingestion

    starttime_training = timeit.default_timer()
    os.system("python training.py")
    timing_training = timeit.default_timer() - starttime_training

    list_times = [["ingestion", timing_ingestion], ["training", timing_training]]

    # return a list of 2 timing values in seconds
    return list_times


##################Function to check dependencies
def outdated_packages_list():
    # get a list of outdated
    outdated = subprocess.check_output(["pip", "list", "--outdated"]).decode(
        sys.stdout.encoding
    )
    return outdated


if __name__ == "__main__":
    model_predictions(pd.read_csv(f"{test_data_path}/testdata.csv"))
    dataframe_summary()
    dataframe_missing_data()
    execution_time()
    outdated_packages_list()

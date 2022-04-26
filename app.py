from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
from diagnostics import (
    model_predictions,
    dataframe_summary,
    dataframe_missing_data,
    execution_time,
    outdated_packages_list,
)
from scoring import score_model
import json
import os


######################Set up variables for use in our script
app = Flask(__name__)

with open("config.json", "r") as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config["output_folder_path"])
test_data_path = os.path.join(config["test_data_path"])

prediction_model = None


#######################Prediction Endpoint
@app.route("/prediction", methods=["POST", "GET"])
def predict():
    # call the prediction function you created in Step 3
    filename = request.args.get("filename")
    testdata = pd.read_csv(f"{test_data_path}/{filename}")
    predictions = model_predictions(testdata)
    return str(predictions) + "\n"  # add return value for prediction outputs


#######################Scoring Endpoint
@app.route("/scoring", methods=["GET", "OPTIONS"])
def scoring():
    # check the score of the deployed model
    score = score_model()
    return str(score) + "\n"  # add return value (a single F1 score number)


#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=["GET", "OPTIONS"])
def stats():
    # check means, medians, and modes for each column
    summarystats = dataframe_summary()
    # return a list of all calculated summary statistics
    return str(summarystats) + "\n"


#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=["GET", "OPTIONS"])
def diagnostics():
    # check timing and percent NA values
    timing = execution_time()
    missing_data = dataframe_missing_data()
    dependency = outdated_packages_list()
    # add return value for all diagnostics
    return f"timing:{str(timing)} \n missing_data:{str(missing_data)} \n dependency:\n{str(dependency)} \n"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True, threaded=True)

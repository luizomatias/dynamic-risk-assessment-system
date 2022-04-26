from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
import shutil


##################Load config.json and correct path variable
with open("config.json", "r") as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config["output_folder_path"])
prod_deployment_path = os.path.join(config["prod_deployment_path"])
model_path = os.path.join(config["output_model_path"])


####################function for deployment
def store_model_into_pickle():
    # copy the latest pickle file, the latestscore.txt value, and the ingestedfiles.txt file into the deployment directory

    model_original = os.getcwd() + "/" + model_path + "/trainedmodel.pkl"
    model_production = os.getcwd() + "/" + prod_deployment_path + "/trainedmodel.pkl"
    shutil.copyfile(model_original, model_production)

    score_original = os.getcwd() + "/" + model_path + "/latestscore.txt"
    score_production = os.getcwd() + "/" + prod_deployment_path + "/latestscore.txt"
    shutil.copyfile(score_original, score_production)

    ingest_original = os.getcwd() + "/" + dataset_csv_path + "/ingestedfiles.txt"
    ingest_production = os.getcwd() + "/" + prod_deployment_path + "/ingestedfiles.txt"
    shutil.copyfile(ingest_original, ingest_production)


if __name__ == "__main__":
    store_model_into_pickle()

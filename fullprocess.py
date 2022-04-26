import training
import scoring
import deployment
import diagnostics
import reporting
import ast
import os
import json
from ingestion import merge_multiple_dataframe
import pandas as pd


#############Load config.json and get input and output paths
with open("config.json", "r") as f:
    config = json.load(f)

input_folder_path = config["input_folder_path"]
dataset_csv_path = os.path.join(config["output_folder_path"])
prod_deployment_path = os.path.join(config["prod_deployment_path"])
test_data_path = os.path.join(config["test_data_path"])

##################Check and read new data
# first, read ingestedfiles.txt
with open(prod_deployment_path + "/ingestedfiles.txt", "r") as file:
    ingestedfiles_list = ast.literal_eval(file.read())


# second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
filenames = os.listdir(os.getcwd() + "/" + input_folder_path)

# check_new_data = set(ingestedfiles_list) == set(filenames)

check_new_data = set(ingestedfiles_list) == set(filenames)
##################Deciding whether to proceed, part 1
# if you found new data, you should proceed. otherwise, do end the process here
if not check_new_data:
    merge_multiple_dataframe()
else:
    print("No new files")
    exit(0)
##################Checking for model drift
# check whether the score from the deployed model is different from the score from the model that uses the newest ingested data

# read latestscore.txt from the deployment
with open(prod_deployment_path + "/latestscore.txt", "r") as file:
    lastestscoref1 = float(file.read())

newscoref1 = scoring.score_model(prod_environment=True)

##################Deciding whether to proceed, part 2
# if you found model drift, you should proceed. otherwise, do end the process here
if newscoref1 >= lastestscoref1:
    print("New f1 score is better or equal than the lastest. No model drift occured.")
    exit(0)
else:
    print("New f1 score is worst than the lastest. Model drift occured.")

##################Re-deployment
# if you found evidence for model drift, re-run the deployment.py script
training.train_model()
deployment.store_model_into_pickle()
##################Diagnostics and reporting
# run diagnostics.py and reporting.py for the re-deployed model
diagnostics.model_predictions(pd.read_csv(f"{test_data_path}/testdata.csv"))
diagnostics.dataframe_summary()
diagnostics.dataframe_missing_data()
diagnostics.execution_time()
diagnostics.outdated_packages_list()
reporting.score_model()

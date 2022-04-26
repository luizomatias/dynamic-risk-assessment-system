import pandas as pd
import numpy as np
import os
import json
from datetime import datetime


#############Load config.json and get input and output paths
with open("config.json", "r") as f:
    config = json.load(f)

input_folder_path = config["input_folder_path"]
output_folder_path = config["output_folder_path"]


#############Function for data ingestion
def merge_multiple_dataframe():

    #############final dataframe columns
    finaldata = pd.DataFrame(
        columns=[
            "corporation",
            "lastmonth_activity",
            "lastyear_activity",
            "number_of_employees",
            "exited",
        ]
    )
    # check for datasets, compile them together, and write to an output file
    filenames = os.listdir(os.getcwd() + "/" + input_folder_path)

    with open(f"{output_folder_path}/ingestedfiles.txt", "w") as file:
        file.write(str(filenames))

    for each_data in filenames:
        currentdf = pd.read_csv(os.getcwd() + "/" + input_folder_path + "/" + each_data)
        finaldata = finaldata.append(currentdf).reset_index(drop=True)

    # drop duplicates
    finaldata = finaldata.drop_duplicates()
    # save final data
    finaldata.to_csv(f"{output_folder_path}/finaldata.csv", index=False)


if __name__ == "__main__":
    merge_multiple_dataframe()

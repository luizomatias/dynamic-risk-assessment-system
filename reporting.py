import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from diagnostics import model_predictions


###############Load config.json and get path variables
with open("config.json", "r") as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config["output_folder_path"])
model_path = os.path.join(config["output_model_path"])
test_data_path = os.path.join(config["test_data_path"])

##############Function for reporting
def score_model():
    # calculate a confusion matrix using the test data and the deployed model
    # write the confusion matrix to the workspace
    testdata = pd.read_csv(f"{test_data_path}/testdata.csv")
    y_test = testdata["exited"].values.reshape(-1, 1).ravel()

    y_pred = model_predictions(testdata)
    cm = metrics.confusion_matrix(y_test, y_pred)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix", fontsize=18)

    plt.savefig(os.path.join(model_path, "confusionmatrix2.png"))


if __name__ == "__main__":
    score_model()

import requests
import json
import os

# Specify a URL that resolves to your workspace
URL = "http://127.0.0.1"

with open("config.json", "r") as f:
    config = json.load(f)

model_path = os.path.join(config["output_model_path"])

# Call each API endpoint and store the responses
response1 = requests.get(URL + ":8000/prediction?filename=testdata.csv").content
response2 = requests.get(URL + ":8000/scoring").content
response3 = requests.get(URL + ":8000/summarystats").content
response4 = requests.get(URL + ":8000/diagnostics").content

# combine all API responses
responses = (
    response1 + "\n" + response2 + "\n" + response3 + "\n" + response4 + "\n"
)  # combine reponses here

# write the responses to your workspace
with open(model_path + "/apireturns2.txt", "w") as file:
    file.write(responses)

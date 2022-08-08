#!/usr/bin/env python3
import base64
import json
import logging
import os
import numpy as np
import requests
import sys
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.dataset.util import to_pandas

logger = logging.getLogger("__mymodelclient__")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
if __name__ == "__main__":
    url = sys.argv[1]
    # path = sys.argv[2]

    # dataset = get_dataset("airpassengers")
    with open('test_ds.pkl', 'rb') as f:
        dataset = pickle.load(f)
    
    true_values = to_pandas(list(dataset.test)[0])
    json_dict = {}
    # time time series to predict
    json_dict[0] = true_values[:-36].to_json()
    json_dict[1] = true_values[:-24].to_json()
    json_dict[2] = true_values[:-12].to_json()
    
    # __import__("pdb").set_trace()
    data = {}
    # seldon accepts input with keywords data, bindata, strdata or jsonData
    data["jsonData"] = json.dumps(json_dict)
    # __import__("pdb").set_trace()
    response = requests.post(url, json=data, timeout=None)
    # __import__("pdb").set_trace()

    logger.info("caught response {}".format(response))
    status_code = response.status_code
    js = response.json()
    if response.status_code == requests.codes["ok"]:
        logger.info("converting tensor to array of sample forecasts")
        data = js.get("data")
        tensor = data.get("tensor")
        shape = tensor.get("shape")
        values = tensor.get("values")
        prediction_samples = np.array(values).reshape(shape)
        logger.info(prediction_samples)
        logger.info(prediction_samples.shape)
    elif response.status_code == requests.codes["service_unavailable"]:
        logger.error("Model service is not available.")
    elif response.status_code == requests.codes["internal_server_error"]:
        logger.error("Internal model error.")

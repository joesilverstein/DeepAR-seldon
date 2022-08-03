#!/usr/bin/env python3
import logging
import pickle
import json
import pandas as pd
import numpy as np
from gluonts.dataset.pandas import PandasDataset
from gluonts.model.deepar import DeepAREstimator

logger = logging.getLogger("__mymodel__")


class MyModel(object):
    def __init__(self):
        logger.info("initializing...")
        logger.info("load model here...")
        with open('gluonts_model.pkl', 'rb') as f:
            self._model = pickle.load(f)
        logger.info("model has been loaded and initialized...")

    def predict(self, X, features_names):
        """
        Seldon Core Prediction API
        X: list of 3 Pandas monthly series (will predict 12 months ahead for each)
           numpy 3D array of 3 Pandas monthly series
        """
        
        logger.info("predict called...")
        # logger.info(X)
        # logger.info(type(X))
        
        X_dict = json.loads(X)
        logger.info(X_dict)
        logger.info(type(X_dict))
        
        logger.info(X_dict.keys())
        logger.info(X_dict['0'])
        logger.info(type(X_dict['0']))
        
        input_list = [None] * len(X_dict)
        for i in range(len(input_list)):
            input_list[i] = pd.read_json(X_dict[str(i)], typ='series', orient='records')
            
        logger.info(input_list[0])
        logger.info(type(input_list[0]))
        
        prediction_input = PandasDataset(input_list)
        logger.info("perform inference here...")
        predictions = self._model.predict(prediction_input)
        logger.info("returning prediction...")
        prediction_samples_list = [prediction.samples for prediction in predictions]
        prediction_samples = np.rollaxis(np.dstack(prediction_samples_list), -1)
        # must return a numpy array
        # https://docs.seldon.io/projects/seldon-core/en/latest/python/python_component.html
        return prediction_samples

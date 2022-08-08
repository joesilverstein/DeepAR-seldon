#!/usr/bin/env python3
import logging
import pickle
import json
import pandas as pd
import numpy as np
from gluonts.model.deepar import DeepAREstimator
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName

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
        
        test_ds_json_dict = json.loads(X)
        logger.info('initial json loading completed')
        target_list = [None] * len(test_ds_json_dict)
        start_list = [None] * len(test_ds_json_dict)
        feat_dynamic_real_list = [None] * len(test_ds_json_dict)
        feat_static_cat_list = [None] * len(test_ds_json_dict)
        for i in range(len(test_ds_json_dict)):
            entry_json_dict = json.loads(test_ds_json_dict[str(i)])
            logger.info('initial ts entry json loading completed')
            target_list[i] = pickle.loads(json.loads(entry_json_dict['target']).encode('latin-1'))
            start_list[i] = pd.Period(entry_json_dict['start'], 'H')
            feat_dynamic_real_list[i] = pickle.loads(json.loads(entry_json_dict['feat_dynamic_real']).encode('latin-1'))
            feat_static_cat_list[i] = pickle.loads(json.loads(entry_json_dict['feat_static_cat']).encode('latin-1'))
            logger.info('finished deserializing entry ' + str(i))
        
        prediction_input = ListDataset(
            [
                {
                    FieldName.TARGET: target,
                    FieldName.START: start,
                    FieldName.FEAT_DYNAMIC_REAL: fdr,
                    FieldName.FEAT_STATIC_CAT: fsc,
                }
                for (target, start, fdr, fsc) in zip(
                    target_list,
                    start_list,
                    feat_dynamic_real_list,
                    feat_static_cat_list,
                )
            ],
            freq='1H',
        )
            
        logger.info("performing inference here...")
        predictions = self._model.predict(prediction_input)
        logger.info("returning prediction...")
        prediction_samples_list = [prediction.samples for prediction in predictions]
        prediction_samples = np.rollaxis(np.dstack(prediction_samples_list), -1)
        # must return a numpy array
        # https://docs.seldon.io/projects/seldon-core/en/latest/python/python_component.html
        return prediction_samples

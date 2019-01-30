#!/usr/bin/env python


from __future__ import print_function, division

import os
import argparse
import importlib
from time import strftime

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from neuralnilm.data.loadactivations import load_nilmtk_activations
from neuralnilm.data.syntheticaggregatesource import SyntheticAggregateSource
from neuralnilm.data.realaggregatesource import RealAggregateSource
from neuralnilm.data.datapipeline import DataPipeline
from neuralnilm.data.processing import DivideBy, IndependentlyCenter
from neuralnilm.utils import select_windows, filter_activations

from dataprocess.mySQL_connect import sql4Keras
from dataprocess.appliance_metrics import Metrics
from dataprocess.plot_module import Figure
from dataprocess.appliance_validation import Validation

from lib import dirs
# Config
START_DATE = None
BUILDING_ID = None

# Parameters
DATASET = None
NILMTK_FILENAME = None
WINDOWS = None
VALIDATE_RANGE = None
SAMPLE_PERIOD = None
MODEL_NAME = None
VALIDATE_LENGTH = None
BUILDING = None
BUILDING_ID = None
NUM_SEQ_PER_BATCH = None

os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Main
def main():
    set_log_level()
    parse_args()
    load_config()

    model_metrics = Validation(NILMTK_FILENAME, SAMPLE_PERIOD, '', WINDOWS, BUILDINGS, format='HDF')
    for building_id in BUILDING_ID:
        if VALIDATE_RANGE and VALIDATE_LENGTH and NUM_SEQ_PER_BATCH:
            model_metrics.validate_model(START_DATE, int(building_id), VALIDATE_RANGE, VALIDATE_LENGTH, NUM_SEQ_PER_BATCH)
        else:
            model_metrics.validate_model(START_DATE, int(building_id))

# Argument parser
def parse_args():
    global DATASET, NILMTK_FILENAME
    global SAMPLE_PERIOD, NUM_SEQ_PER_BATCH
    global MODEL_NAME, START_DATE, BUILDING_ID, VALIDATE_RANGE, VALIDATE_LENGTH, WINDOWS

    parser = argparse.ArgumentParser()

    # required
    required_named_arguments = parser.add_argument_group('required named arguments')
    required_named_arguments.add_argument('-d', '--dataset',
                                          help='Dataset\'s name. For example, \'redd\'.',
                                          required=True)
    required_named_arguments.add_argument('-s', '--sample-period',
                                          help='Sample period (in seconds).',
                                          type=int,
                                          required=True)
    required_named_arguments.add_argument('-m', '--model-name',
                                          help='model\'s name. For example, \'III_2017_12_22_TV\'.',
                                          required=True)
    required_named_arguments.add_argument('-t', '--start-date',
                                          help='Time for start date. For example, \'2018/01/01\'.',
                                          required=True)
    required_named_arguments.add_argument('-b', '--building-id',
                                          help='Building id for validating',
                                          required=True)

    # optional
    optional_named_arguments = parser.add_argument_group('optional named arguments')
    optional_named_arguments.add_argument('-n', '--num-seq-per-batch',
                                          help='Size of batch',
                                          type=int)
    optional_named_arguments.add_argument('-r', '--validate-range',
                                          help='Range for validating',
                                          type=int)
    optional_named_arguments.add_argument('-l', '--validate-length',
                                          help='sequence length for validating',
                                          type=int)

    # start parsing
    args = parser.parse_args()

    DATASET = args.dataset
    NILMTK_FILENAME = os.path.join(dirs.DATA_DIR, DATASET + '.h5')
    SAMPLE_PERIOD = args.sample_period
    START_DATE = args.start_date
    MODEL_NAME = args.model_name
    BUILDING_ID = args.building_id.split(',')

    NUM_SEQ_PER_BATCH = args.num_seq_per_batch
    VALIDATE_RANGE= args.validate_range
    VALIDATE_LENGTH = args.validate_length


# Config loader
def load_config():
    global WINDOWS, BUILDINGS

    # dataset-dependent config
    config_module = importlib.import_module(dirs.CONFIG_DIR + '.' + DATASET, __name__)
    WINDOWS = config_module.WINDOWS
    BUILDINGS = config_module.BUILDINGS

def set_log_level():
    # hide warning log
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # ignore UserWarning log
    import warnings
    warnings.filterwarnings("ignore")

if __name__ == '__main__':
    main()

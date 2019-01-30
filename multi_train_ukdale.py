#!/usr/bin/env python


from __future__ import print_function, division

import argparse
import importlib
from time import strftime
from os import environ
from os import path
from os import makedirs

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from neuralnilm.data.loadactivations import load_nilmtk_activations
from neuralnilm.data.syntheticaggregatesource import SyntheticAggregateSource
from neuralnilm.data.realaggregatesource import RealAggregateSource
from neuralnilm.data.stridesource import StrideSource
from neuralnilm.data.datapipeline import DataPipeline
from neuralnilm.data.processing import DivideBy, IndependentlyCenter
from neuralnilm.utils import select_windows, filter_activations

from dataprocess.mySQL_connect import sql4Keras
from dataprocess.appliance_metrics import Metrics
from dataprocess.plot_module import Figure
from collections import defaultdict
from dataprocess.data_preprocess import Preprocess
from dataprocess.data_preprocess import input_median_filter, appliance_median_filter
import math
import pickle as cPickle
import types
import tempfile

from lib import dirs
# Config
WINDOWS = None
BUILDINGS = None
SEQ_PERIODS = None


# Parameters
DATASET = None
NILMTK_FILENAME = None
TARGET_APPLIANCE = None
SAMPLE_PERIOD = None
NUM_STEPS = None
TOPOLOGY_NAME = None
OVERRIDE = None
seq_length = None
APPLIANCES = None
NUM_APPLIANCE = None
BUILDINGS_APPLIANCES = None
LOAD_PIPELINE = None

# Constants
NUM_SEQ_PER_BATCH = 24
BRIEFING_NUM_STEPS = 1000

environ["CUDA_VISIBLE_DEVICES"] = ""

# Main
def main():
    global BUILDINGS_APPLIANCES, seq_length
    set_log_level()
    parse_args()
    load_config()

    pipeline = None
    pipe_path = path.join(dirs.MODELS_DIR,
                                        'pipe_' + DATASET + '_[' + TARGET_APPLIANCE + ']' + '.pkl')

    if LOAD_PIPELINE:
        print('Loading pipeline ...')
        with open(pipe_path, 'rb') as fp:
            pipeline = cPickle.load(fp)
        seq_period = SEQ_PERIODS[APPLIANCES[0]]
        seq_length = seq_period // SAMPLE_PERIOD
    else:
        # load the activations
        print('Loading activations ...')
        BUILDINGS_APPLIANCES = BUILDINGS.keys()
        activations = load_nilmtk_activations(
            appliances=BUILDINGS_APPLIANCES,
            filename=NILMTK_FILENAME,
            sample_period=SAMPLE_PERIOD,
            windows=WINDOWS
        )
#        activations_processor = Preprocess(activations)
#        activations_processor.activations_pruning()
#        activations_processor.median_filter(k_factor=3)
#        activations = activations_processor.get_activations()

        # generate pipeline
        pipeline, input_std, target_std = get_pipeline(activations)
        with open(pipe_path, 'wb') as fp:
            cPickle.dump(pipeline, fp, True)

    # determine the input shape
    print('Determining input shape ... ', end='')
    batch = pipeline.get_batch()
    input_shape = batch.input.reshape(NUM_SEQ_PER_BATCH, seq_length, 1).shape[1:]
    print(input_shape)

    # look for an existing model only when OVERRIDE is not on; if none, then
    # build a new one
    print('Looking for an existing model ... ', end='')
    model_filename = path.join(dirs.MODELS_DIR, DATASET + '_[' + TARGET_APPLIANCE + ']_' + strftime('%Y-%m-%d_%H_%m') + '.h5')
    if not OVERRIDE and path.exists(model_filename):
        print('Found; loading it ...')
        from keras.models import load_model
        model = load_model(model_filename)
    else:
        if OVERRIDE:
            print('Overridden; building a new one with the specified topology ...')
        else:
            print('Not found; building a new one with the specified topology ...')

        # define accuracy
        #import keras.backend as K
        #ON_POWER_THRESHOLD = DivideBy(target_std)(10)
        #def acc(y_true, y_pred):
        #    return K.mean(K.equal(K.greater_equal(y_true, ON_POWER_THRESHOLD),
        #                          K.greater_equal(y_pred, ON_POWER_THRESHOLD)))

        # build model
        topology_module = importlib.import_module(dirs.TOPOLOGIES_DIR + '.' + TOPOLOGY_NAME, __name__)
        model = topology_module.build_model(input_shape, APPLIANCES)
        print (model.summary())

    # train
    print('Preparing the training process ...')
    train(pipeline, model)

    # save the model
    print('Saving the model to ' + model_filename + ' ...')
    model.save(model_filename)


# Argument parser
def parse_args():
    global DATASET, NILMTK_FILENAME, TARGET_APPLIANCE, APPLIANCES
    global SAMPLE_PERIOD, NUM_SEQ_PER_BATCH, NUM_STEPS, NUM_APPLIANCE
    global BRIEFING_NUM_STEPS, TOPOLOGY_NAME, OVERRIDE, LOAD_PIPELINE

    parser = argparse.ArgumentParser()

    # required
    required_named_arguments = parser.add_argument_group('required named arguments')
    required_named_arguments.add_argument('-d', '--dataset',
                                          help='Dataset\'s name. For example, \'redd\'.',
                                          required=True)
    required_named_arguments.add_argument('-a', '--target-appliance',
                                          help='Target appliance. For example, \'fridge,television\'.',
                                          required=True)
    required_named_arguments.add_argument('-s', '--sample-period',
                                          help='Sample period (in seconds).',
                                          type=int,
                                          required=True)
    required_named_arguments.add_argument('-t', '--num-steps',
                                          help='Number of steps.',
                                          type=int,
                                          required=True)
    required_named_arguments.add_argument('-m', '--topology-name',
                                          help='Topology\'s name. For example, \'dae\'.',
                                          required=True)

    # optional
    optional_named_arguments = parser.add_argument_group('optional named arguments')
    optional_named_arguments.add_argument('-o', '--override',
                                          help='Flag to override existing model (if there\'s one).',
                                          action='store_false')

    optional_named_arguments = parser.add_argument_group('optional named arguments')
    optional_named_arguments.add_argument('-p', '--pipeline',
                                          help='Flag to load existing pipeline (if there\'s one).',
                                          action='store_false')

    # start parsing
    args = parser.parse_args()

    DATASET = args.dataset
    NILMTK_FILENAME = path.join(dirs.DATA_DIR, DATASET + '.h5')
    TARGET_APPLIANCE = args.target_appliance
    SAMPLE_PERIOD = args.sample_period
    NUM_STEPS = args.num_steps
    TOPOLOGY_NAME = args.topology_name
    OVERRIDE = args.override
    LOAD_PIPELINE = args.pipeline

    APPLIANCES = TARGET_APPLIANCE.split(',')
    NUM_APPLIANCE = len(APPLIANCES)

# Config loader
def load_config():
    global WINDOWS, BUILDINGS, SEQ_PERIODS

    # dataset-dependent config
    config_module = importlib.import_module(dirs.CONFIG_DIR + '.' + DATASET, __name__)
    WINDOWS = config_module.WINDOWS
    BUILDINGS = config_module.BUILDINGS

    # sequence periods (shared among datasets)
    seq_periods_module = importlib.import_module(dirs.CONFIG_DIR + '.seq_periods', __name__)
    SEQ_PERIODS = seq_periods_module.SEQ_PERIODS


# Pipeline
def get_pipeline(activations):
    global seq_length
    agg_source = []
    prob = []
    target_inclusion_prob = 0.48 + len(APPLIANCES) * 0.1

    for task_appliance in APPLIANCES:
        seq_period = SEQ_PERIODS[task_appliance]
        seq_length = seq_period // SAMPLE_PERIOD

        # buildings
        buildings = BUILDINGS[task_appliance]
        train_buildings = buildings['train_buildings']
        unseen_buildings = buildings['unseen_buildings']

        # windows
        filtered_windows = select_windows(
            train_buildings, unseen_buildings, WINDOWS)
        filtered_activations = filter_activations(
            filtered_windows, activations, BUILDINGS_APPLIANCES)

        # data sources
        real_source_prob=min(0.82, target_inclusion_prob)
        if task_appliance=='fridge':
            real_source_prob=1.0

        agg_source.append(RealAggregateSource(
            activations=filtered_activations,
            target_appliance=task_appliance,
            appliances = APPLIANCES,
            target_inclusion_prob = real_source_prob,
            seq_length=seq_length,
            filename=NILMTK_FILENAME,
            windows=filtered_windows,
            sample_period=SAMPLE_PERIOD
        ))
        prob.append(1.0/NUM_APPLIANCE)

        """agg_source.append(SyntheticAggregateSource(
            activations=filtered_activations,
            appliances=APPLIANCES,
            seq_length=seq_length,
            distractor_inclusion_prob=0.3,
            target_inclusion_prob=min(0.5, target_inclusion_prob),
            sample_period=SAMPLE_PERIOD
        ))

        agg_source.append(StrideSource(
            target_appliance=task_appliance,
            appliances=APPLIANCES,
            seq_length=seq_length,
            filename=NILMTK_FILENAME,
            windows=filtered_windows,
            sample_period=SAMPLE_PERIOD,
            stride=None
        ))
        prob.append(0.5/NUM_APPLIANCE)"""

    # look for existing processing parameters only when OVERRIDE is not on; if
    # none, generate new ones
    print('Looking for existing processing parameters ... ')
    proc_params_filename = path.join(dirs.MODELS_DIR, 'proc_params_' + DATASET + '_[' + TARGET_APPLIANCE + ']_' + strftime('%Y-%m-%d_%H_%m') + '.npz')
    if not OVERRIDE and path.exists(proc_params_filename):
        print('Found; using them ...')
        multi_input_std = np.load(proc_params_filename)['multi_input_std']
        multi_target_std = np.load(proc_params_filename)['multi_target_std']
    else:
        if OVERRIDE:
            print('Overridden; generating new ones ...')
        else:
            print('Not found; generating new ones ...')
        multi_input_std = np.array([])
        multi_target_std = np.array([])

        for sample_source in agg_source:
            batch_size = 1024
            sample = sample_source.get_batch(num_seq_per_batch=batch_size).next()
            sample = sample.before_processing

            multi_input_std = np.append(multi_input_std, sample.input.flatten().std())
            multi_target_std = np.append(multi_target_std, [sample.target[:,idx].flatten().std() for idx in range(NUM_APPLIANCE)])

        multi_input_std = np.mean(multi_input_std)
        multi_target_std = multi_target_std.reshape(-1, NUM_APPLIANCE)
        multi_target_std = np.mean(multi_target_std, axis=0)

        print('='*10)
        print('Input std = ', multi_input_std)
        for idx, appliance in enumerate(APPLIANCES):
            print(appliance, 'std = ', multi_target_std[idx])
        print('='*10)

        print('Saving the processing parameters ...')
        np.savez(proc_params_filename, multi_input_std = [multi_input_std], multi_target_std = multi_target_std)

    # generate pipeline
    pipeline = DataPipeline(
        agg_source,
        num_seq_per_batch=NUM_SEQ_PER_BATCH,
        input_processing=[DivideBy(multi_input_std), IndependentlyCenter()],
        target_processing=[DivideBy(multi_target_std)],
        source_probabilities=prob,
    )

    return pipeline, multi_input_std, multi_target_std


# Trainer
def train(pipeline, model):
    # create output directory
    print('Creating output directory ... ', end='')
    output_dir = path.join( dirs.OUTPUT_DIR,
                              DATASET + '_[' + TARGET_APPLIANCE + ']_' + strftime('%Y-%m-%d_%H-%M-%S'))

    makedirs(output_dir)
    plot_fig = Figure(output_dir, pipeline, APPLIANCES, 20)
    print(output_dir)

    # run
    log = []
    for step in range(NUM_STEPS + 1):
        # generate batch
        batch = pipeline.get_batch()
        while batch is None:
            batch = pipeline.get_batch()

        # train on a single batch (except for step 0, so that the code below can
        # generate briefing on the initial state of the model at step 0)
        if (step != 0):
            batch_input = batch.input.reshape((NUM_SEQ_PER_BATCH, seq_length, 1))
            batch_input = input_median_filter(batch_input)
            batch_target = batch.target.reshape((NUM_SEQ_PER_BATCH, NUM_APPLIANCE, seq_length, 1))
            batch_target = appliance_median_filter(batch_target)
            batch_target = batch_target.transpose((1, 0, 2, 3))
            train_metrics = model.train_on_batch(x=batch_input,
                                                 y=[batch_target[idx] for idx in range(NUM_APPLIANCE)])

        # generate briefing
        if (step % BRIEFING_NUM_STEPS == 0 or step == NUM_STEPS):
            valid_batch = pipeline.get_batch(fold='unseen_appliances',
                                             reset_iterator=True,
                                             validation=True)
            while valid_batch is None:
                valid_batch = pipeline.get_batch(fold='unseen_appliances',
                                             reset_iterator=True,
                                             validation=True)

            valid_batch_input = valid_batch.input.reshape((NUM_SEQ_PER_BATCH, seq_length, 1))
            valid_batch_input = input_median_filter(valid_batch_input)

            valid_series_input = valid_batch.before_processing.input
            valid_series_target = valid_batch.before_processing.target.reshape((NUM_SEQ_PER_BATCH, NUM_APPLIANCE, seq_length))
            valid_series_target = appliance_median_filter(valid_series_target)

            valid_series_pred = np.array(model.predict_on_batch(valid_batch_input)).reshape((NUM_APPLIANCE, NUM_SEQ_PER_BATCH, seq_length))
            valid_series_pred = appliance_median_filter(valid_series_pred)

            valid_series_pred = valid_series_pred.transpose(1, 0, 2)

            if (step == 0):
                print('===============================================================================')
                print('============================== Start of training ==============================')
                print('===============================================================================')
            else:
                print('*'*25)
                print('Step : {} , Time : {}\n'.format(step,strftime('%Y-%m-%d_%H_%M')))
                print('Training metrics: ')
                for i, metrics_name in enumerate(model.metrics_names):
                    print('{}={:.4f}, '.format(metrics_name, train_metrics[i]))
                print('')

                plot_fig.plot_figures(valid_series_input, valid_series_target, valid_series_pred, step)
                valid_series_pred = pipeline.apply_inverse_processing(valid_series_pred, 'target')

                appliance_metrics = Metrics(APPLIANCES, valid_series_target, valid_series_pred)
                appliance_metrics.compute_metrics()
                appliance_metrics.print_metrics()

                # append to log
                log.append([step] + train_metrics)

    print('===============================================================================')
    print('=============================== End of training ===============================')
    print('===============================================================================')

    # write the log
    print('Writing log ...')
    log_df = pd.DataFrame(log, columns=(['step'] +
                                        ['train_'+metrics_name for metrics_name in model.metrics_names]))

    log_df.to_csv(path.join(output_dir, 'log_' + TARGET_APPLIANCE + '_' + strftime('%Y-%m-%d_%H_%m') + '.csv'), index=False, float_format='%.4f')

def set_log_level():
    # hide warning log
    from os import environ
    environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # ignore UserWarning log
    import warnings
    warnings.filterwarnings("ignore")

if __name__ == '__main__':
    main()

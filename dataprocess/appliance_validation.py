#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function, division
import numpy as np
import nilmtk
import os

from neuralnilm.data.loadactivations import load_nilmtk_activations
from neuralnilm.data.datapipeline import DataPipeline
from neuralnilm.data.processing import DivideBy, IndependentlyCenter
from neuralnilm.utils import select_windows, filter_activations

from dataprocess.appliance_metrics import Metrics
from dataprocess.plot_module import Figure
from dataprocess.source4validation import ValidationSource
from collections import defaultdict
from keras.models import load_model
from time import strftime

from lib import dirs

class Validation():
    def __init__(self, nilm_filename, sample_period, model_name, window, building, format='HDF'):
        self.nilm_filename = nilm_filename
        self.sample_period = sample_period
        self.model_name = model_name
        self.window = window
        self.building = building
        self.model = load_model(os.path.join(dirs.MODELS_DIR, self.model_name + '.h5'))
        self.format = format
        print(self.model.summary())
        self.appliances = [str(layer.name.replace("_", " ")) for layer in self.model.layers[1:] if layer.output_shape == self.model.layers[-1].output_shape ]

    def get_pipeline(self, num_seq_per_batch, start_date, building_id, valid_range, validate_length):
        valid_agg_source = []

        for task_appliance in self.appliances:
            # buildings
            buildings = self.building[task_appliance]
            train_buildings = buildings['train_buildings']
            unseen_buildings = buildings['unseen_buildings']

            # windows
            filtered_windows = select_windows(train_buildings, unseen_buildings, self.window)

            # data sources
            valid_agg_source.append(ValidationSource(
                appliances=self.appliances,
                filename=self.nilm_filename,
                windows = filtered_windows,
                sample_period=self.sample_period,
                start_date=start_date,
                valid_range=valid_range,
                building_id = building_id,
                validate_length = validate_length,
                format=self.format
            ))

    	# look for existing processing parameters only when OVERRIDE is not on; if
    	# none, generate new ones
    	print('Looking for existing processing parameters ... ')
    	proc_params_filename = os.path.join(dirs.MODELS_DIR, 'proc_params_' + self.model_name + '.npz')

        print('Found; using them ...')
        multi_input_std = np.load(proc_params_filename)['multi_input_std']
        multi_target_std = np.load(proc_params_filename)['multi_target_std']
        """multi_input_std = 1000
        multi_target_std = 400"""

    	# generate pipeline
    	pipeline = DataPipeline(
            valid_agg_source,
            num_seq_per_batch=num_seq_per_batch,
            input_processing=[DivideBy(multi_input_std), IndependentlyCenter()],
            target_processing=[DivideBy(multi_target_std)]
    	)

        return pipeline, multi_input_std, multi_target_std

    def validate_model(self, start_date, building_id, valid_range = 45, validate_length=1440, num_seq_per_batch = 30):
        valid_pipeline, multi_input_std, multi_target_std = self.get_pipeline(num_seq_per_batch, start_date, building_id, valid_range, validate_length)
        valid_batch = valid_pipeline.get_batch(fold='unseen_appliances', reset_iterator=True, validation=True)
        while valid_batch is None:
            valid_batch = valid_pipeline.get_batch(num_seq_per_batch = num_seq_per_batch, fold='unseen_appliances',
                                             reset_iterator=True,
                                             validation=True)

        # create output directory
        print('Creating output directory ... ')
        output_dir = os.path.join(dirs.OUTPUT_DIR,
                                  self.model_name + '_validation of building' + str(building_id) + '_' + strftime('%Y-%m-%d_%H-%M-%S'))

        os.makedirs(output_dir)
        plot_fig = Figure(output_dir, valid_pipeline, self.appliances, num_figures=num_seq_per_batch)
        print(output_dir)

        input_length = self.model.layers[0].input_shape[1]
        valid_batch_input = valid_batch.input.reshape((num_seq_per_batch, -1, input_length, 1))
        num_appliances = len(self.appliances)
        valid_series_pred = []

        for days in range(num_seq_per_batch):
            valid_input = valid_batch_input[days]
            seq_per_day = int(validate_length / input_length)
            valid_pred = np.array(self.model.predict_on_batch(valid_input)).reshape(
                (num_appliances, seq_per_day, input_length))
            valid_pred.transpose(1, 0, 2)
            valid_series_pred.append(valid_pred)

        valid_series_pred = np.array(valid_series_pred).reshape(
            (num_seq_per_batch, num_appliances, validate_length))
        valid_series_target = valid_batch.before_processing.target.reshape(
            (num_seq_per_batch, num_appliances, validate_length))
        valid_series_input = valid_batch.before_processing.input

        print('===============================================================================')
        print('============================== Start of testing ==============================')
        print('===============================================================================')

        plot_fig.plot_figures(valid_series_input, valid_series_target, valid_series_pred)
        valid_series_pred = valid_pipeline.apply_inverse_processing(valid_series_pred, 'target')

        appliance_metrics = Metrics(self.appliances, valid_series_target, valid_series_pred)
        appliance_metrics.compute_metrics()
        appliance_metrics.print_metrics()

        print('===============================================================================')
        print('=============================== End of testing ===============================')
        print('===============================================================================')

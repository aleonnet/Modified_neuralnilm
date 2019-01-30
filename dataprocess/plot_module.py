#!/usr/bin/env python

import os
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

class Figure():
    def __init__(self, output_dir, pipeline, appliances, num_figures=10):
        self.file_path = output_dir
        self.pipeline = pipeline
        self.appliances = appliances
        self.num_figures = num_figures
        self.create_dir()

    def create_dir(self):
        for appliance_idx in range(len(self.appliances)):
            for i in range(1, self.num_figures + 1):
                os.makedirs(os.path.join(self.file_path, self.appliances[appliance_idx], str(i)))

    def init_series(self, input_series, target_series, pred_series, sample_index, appliance_idx):
        self.input_series = input_series[sample_index].flatten()

        if self.num_appliance > 1:
            self.target_series = target_series[sample_index][appliance_idx].flatten()
            self.pred_series = self.pipeline.apply_inverse_processing(pred_series, 'target')
            self.pred_series = self.pred_series.reshape(self.num_seq, self.num_appliance, -1)[sample_index][appliance_idx].flatten()
        else:
            self.target_series = target_series[sample_index].flatten()
            self.pred_series = self.pipeline.apply_inverse_processing(pred_series, 'target')[sample_index].flatten()

        if self.target_series.shape != self.pred_series.shape:
            raise ValueError("pred.shape != target.shape")

    def plot_figures(self, input_series, target_series, pred_series, step=0):
        self.num_seq = target_series.shape[0]
        self.num_appliance = target_series.shape[1]

        valid_sample_indices = np.random.choice(range(self.num_seq), size=self.num_figures, replace=False)

        for appliance_idx in range(len(self.appliances)):
            for sample_no, sample_index in enumerate(valid_sample_indices):
                self.init_series(input_series, target_series, pred_series, sample_index, appliance_idx)
                self.plot_series(appliance_idx, sample_no, step)

    def plot_series(self, appliance_idx, sample_no, step):
        seq_length = self.input_series.shape[0]
        x_tick = range(0, seq_length, int(seq_length/6))

        p1 = plt.subplot(131)
        p1.set_title('Input #{}'.format(sample_no + 1))
        p1.set_xlabel('Timestamp (min)')
        p1.set_ylabel('Power (w)')
        p1.set_xticks(x_tick)

        p2 = plt.subplot(132, sharey=p1)
        p2.set_title('Target #{} '.format(sample_no + 1) + self.appliances[appliance_idx])
        p2.set_xlabel('Timestamp (min)')
        p2.set_xticks(x_tick)

        p3 = plt.subplot(133, sharey=p1)
        p3.set_title('Prediction #{}'.format(sample_no + 1, step))
        p3.set_xlabel('Timestamp (min)')
        p3.set_xticks(x_tick)

        p1.plot(self.input_series)
        p2.plot(self.target_series)
        p3.plot(self.pred_series)

        plt.savefig(os.path.join(self.file_path, self.appliances[appliance_idx], str(sample_no + 1),
                                 'Step_{}.png'.format(step)))
        plt.clf()

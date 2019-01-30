#!/usr/bin/env python

from __future__ import division
import os
import numpy as np
from scipy.signal import medfilt

def input_median_filter(batch, k_factor=3):
    for idx, seq in enumerate(batch):
        batch[idx] = medfilt(seq.flatten(), k_factor).reshape(seq.shape)
    return batch

def appliance_median_filter(batch, k_factor=3):
    for idx, appliances in enumerate(batch):
        for appliance_idx, seq in enumerate(appliances):
            batch[idx][appliance_idx] = medfilt(seq.flatten(), k_factor).reshape(seq.shape)
    return batch

class Preprocess():
    def __init__(self, activations):
        self.activations = activations

    def activations_pruning(self, upper_factor=4.2, lower_factor=0.15, window_size = 120, fold='train'):
        for applaince in self.activations[fold].keys():
            applaince_activations = self.activations[fold][applaince]
            avg_power, avg_len, num_activations = self.appliance_statistics(applaince_activations)

            for buliding_id, tar_activations in applaince_activations.iteritems():
                remove_list = []
                for idx, activation in enumerate(tar_activations):
                    seq_length = len(activation)
                    seq_power = activation.values.mean()
                    is_drop = False

                    if seq_power > upper_factor*avg_power or seq_power < lower_factor*avg_power:
                        is_drop = True
                    if seq_length > (upper_factor*0.75)*avg_len or seq_length < lower_factor*avg_len:
                        is_drop = True
                    if seq_length > window_size * 2:
                        is_drop = True
                    if is_drop:
                        remove_list.append(idx)

                for idx in reversed(remove_list):
                    del self.activations[fold][applaince][buliding_id][idx]

            print('*'*10)
            print(applaince)
            print('avg_power = {:.1f}'.format(avg_power))
            print('avg_duration = {:.1f}'.format(avg_len))
            print('num_activations = ' + str(num_activations))
            print('*'*10)

    def appliance_statistics(self, activations):
        avg_duration = 0.
        avg_power = 0.
        count = 0.
        total_activations = 0
        for buliding_id, activations in activations.iteritems():
            num_activations = len(activations)
            total_activations += num_activations
            count += 1
            for seq in activations:
                avg_duration += len(seq) / num_activations
                avg_power += seq.values.mean() / num_activations

        return avg_power/count, avg_duration/count, total_activations

    def median_filter(self, fold='train', k_factor=3):
        for applaince in self.activations[fold].keys():
            applaince_activations = self.activations[fold][applaince]
            for buliding_id, tar_activations in applaince_activations.iteritems():
                for idx, activation in enumerate(tar_activations):
                    self.activations[fold][applaince][buliding_id][idx].loc[:] = medfilt(activation.values, k_factor)

    def get_activations(self):
        return self.activations



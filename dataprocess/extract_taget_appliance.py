#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function, division
from collections import defaultdict
import numpy as np


def get_target_series(multi_appliance_series, target_appliance, appliances, category):
    multi_appliance_series = reshape_multi_series (multi_appliance_series)
    batch_size = multi_appliance_series[1]

    for idx, appliance in enumerate(appliances):
        if appliance == target_appliance:
            return np.squeeze(multi_appliance_series[idx], axis=(0,)).reshepe(batch_size, -1)

def reshape_multi_series(multi_appliance_series, category):
    if category == 'tar' or 'target':
        multi_appliance_series.transpose((1, 0, 2))

    if multi_appliance_series.shape[-1] == 1:
        multi_appliance_series = np.squeeze(multi_appliance_series, axis=(-1,))

    return multi_appliance_series
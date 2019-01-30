from __future__ import print_function, division
from copy import copy
from datetime import timedelta
import numpy as np
import pandas as pd
import nilmtk
from nilmtk.timeframegroup import TimeFrameGroup
from nilmtk.timeframe import TimeFrame
from neuralnilm.data.source import Sequence
from neuralnilm.utils import check_windows
from neuralnilm.data.activationssource import ActivationsSource
from neuralnilm.consts import DATA_FOLD_NAMES
from collections import defaultdict
from neuralnilm.data.source import Source
import datetime as dt

import logging
logger = logging.getLogger(__name__)

from datetime import timedelta, date

"""
appliance_in_code:
    main : 0
    television : 2
    fridge : 3
    air conditioner : 4
    bottle warmer : 5
    washing machine : 6
"""
meters_in_code = ['main', 'other', 'television', 'fridge', 'air conditioner', 'bottle warmer', 'washing machine']

class ValidationSource(Source):
    """
    Attributes
    ----------
    mains : dict
        Structure example:
        {<train | unseen_appliances | unseen_activations_of_seen_appliances>: {
             <building_name>: pd.Series of raw data
        }}
    mains_good_sections : dict
        Same structure as `mains`.
    """
    def __init__(self, appliances, filename, windows, sample_period, start_date, valid_range, building_id,
		validate_length, fold=['unseen_appliances'], rng_seed=None, format='HDF'):
        self.appliances = appliances
        self.seq_length = validate_length
        self.filename = filename
        self.sample_period = sample_period
        self.start_date = dt.datetime.strptime(start_date, "%Y/%m/%d")
        self.valid_date = self.start_date
        self.valid_range = valid_range
        self.building_id = building_id
        check_windows(windows)
        self.windows = windows
        self.fold = fold
        self.format = format
        super(ValidationSource, self).__init__(rng_seed=rng_seed)

        self._load_mains_into_memory()

    def _load_mains_into_memory(self):
        logger.info("Loading NILMTK mains...")
        # Load dataset
        dataset = nilmtk.DataSet(self.filename, self.format)
        self.mains = {}
        self.mains_good_sections = {}
        self.target = {}
        for fold in self.fold:
            window = self.windows[fold][self.building_id]
            dataset.set_window(*window)
            elec = dataset.buildings[self.building_id].elec
            self.building_name = (dataset.metadata['name'] + '_building_{}'.format(self.building_id))
            logger.info("Loading mains for {}...".format(self.building_name))

            mains_meter = elec.mains()
            mains_data = mains_meter.power_series_all_data(sample_period=self.sample_period)
            target_data = defaultdict(lambda: np.array())

            for label in self.appliances:
                target_data[label] = elec[label].power_series_all_data(sample_period=self.sample_period)

            def set_mains_data(dictionary, data):
                dictionary.setdefault(fold, {})[self.building_name] = data

            if not mains_data.empty and len(target_data.keys()):
                set_mains_data(self.mains, mains_data)
                set_mains_data(self.target, target_data)
            else:
                print('no available data')

            logger.info("Loaded mains data from building {} for fold {} from {} to {}."
                        .format(self.building_name, fold, mains_data.index[0], mains_data.index[-1]))

        dataset.store.close()
        logger.info("Done loading NILMTK mains data.")

    def _get_sequence_which_includes_target(self, fold, valid_date):
        seq = Sequence(self.seq_length, [len(self.appliances), self.seq_length])

        # Check neighbouring activations
        mains_start = valid_date

        # Get mains
        mains_for_building = self.mains[fold][self.building_name]
        # load some additional data to make sure we have enough samples
        mains_end_extended = mains_start + timedelta(days=2)
        mains = mains_for_building[mains_start:mains_end_extended].dropna()
        seq.input = mains.values[:self.seq_length]

        # Get targets
        targets_for_building = self.target[fold][self.building_name]
        seq.target = np.array([targets_for_building[label][mains_start:mains_end_extended].dropna().values[:self.seq_length]
                                for label in self.appliances])
        return seq


    def _get_sequence(self, fold='unseen_appliances', enable_all_appliances=False):
        if enable_all_appliances:
            raise ValueError("`enable_all_appliances` is not implemented yet"
                             " for ValidationSource!")

        _seq_getter_func = self._get_sequence_which_includes_target

        for forward_time in range(self.valid_range):
            tar_success = 0
            self.valid_date = self.valid_date + timedelta(days=1)
            seq = _seq_getter_func(fold=fold, valid_date=self.valid_date)
            if seq is None:
                continue
            if len(seq.input) != self.seq_length:
                continue
            for i in range(len(self.appliances)):
                if len(seq.target[i]) == self.seq_length:
                    tar_success += 1
            if tar_success == len(self.appliances):
                return seq
        print('No valid seq data')
        print('validate date:', self.valid_date)

    def get_sequence(self, fold='unseen_appliances', enable_all_appliances=False):
        while True:
            yield self._get_sequence(
                fold=fold, enable_all_appliances=enable_all_appliances)

    @classmethod
    def _attrs_to_remove_for_report(cls):
        return ['activations', 'rng', 'mains', 'mains_good_sections']

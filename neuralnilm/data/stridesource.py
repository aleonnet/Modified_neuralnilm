from __future__ import print_function, division
from datetime import timedelta
import numpy as np
import pandas as pd
import nilmtk
from nilmtk.timeframegroup import TimeFrameGroup
from nilmtk.timeframe import TimeFrame
from neuralnilm.data.source import Sequence
from neuralnilm.utils import check_windows
from neuralnilm.data.source import Source
from neuralnilm.consts import DATA_FOLD_NAMES
from collections import defaultdict

import logging
logger = logging.getLogger(__name__)


class StrideSource(Source):
    """
    Attributes
    ----------
    data : dict
        Structure example:
        {<train | unseen_appliances | unseen_activations_of_seen_appliances>: {
             <building_name>: pd.DataFrame of with 2 cols: mains, target
        }}
    _num_seqs : pd.Series with 2-level hierarchical index
        L0 : train, unseen_appliances, unseen_activations_of_seen_appliances
        L1 : building_names
    """
    def __init__(self, target_appliance, appliances,
                 seq_length, filename, windows, sample_period,
                 stride=None,
                 rng_seed=None):
        self.target_appliance = target_appliance
        self.appliances = appliances
        self.seq_length = seq_length
        self.filename = filename
        check_windows(windows)
        self.windows = windows
        self.sample_period = sample_period
        self.stride = self.seq_length if stride is None else stride
        self._reset()
        super(StrideSource, self).__init__(rng_seed=rng_seed)
        # stop validation only when we've gone through all validation data
        self.num_batches_for_validation = None

        self._load_data_into_memory()
        self._compute_num_sequences_per_building()


    def _reset(self):
        self.data = {}
        self._num_seqs = pd.Series()

    def _load_data_into_memory(self):
        logger.info("Loading NILMTK data...")

        # Load dataset
        dataset = nilmtk.DataSet(self.filename)

        for fold, buildings_and_windows in self.windows.iteritems():
            for building_i, window in buildings_and_windows.iteritems():
                dataset.set_window(*window)
                elec = dataset.buildings[building_i].elec

                """appliances = elec.get_labels(list(elec.identifier.meters))
                meter_complete = True
                for appliance_name in self.appliances:
                    if appliance_name.title() not in appliances:
                        meter_complete=False
                if not meter_complete:
                    continue"""

                building_name = (
                    dataset.metadata['name'] +
                    '_building_{}'.format(building_i))

                # Mains
                logger.info(
                    "Loading data for {}...".format(building_name))

                mains_meter = elec.mains()
                good_sections = mains_meter.good_sections()
                good_sections = elec[self.target_appliance].good_sections(sections=good_sections)

                if len(good_sections) < 1:
                    continue

                def load_data(meter):
                    return meter.power_series_all_data(
                        sample_period=self.sample_period,
                        sections=good_sections)

                power_series_data = defaultdict(lambda: np.array())
                power_series_data['mains'] = load_data(mains_meter)
                main_index = power_series_data['mains'].index
                is_valid = True

                for appliance_name in self.appliances:
                    appliance_meter = elec[appliance_name]
                    power_series_data[appliance_name] = load_data(appliance_meter)

                    if power_series_data[appliance_name] is None:
                        is_valid = False
                        break

                    power_series_data[appliance_name] = power_series_data[appliance_name].loc[main_index]
                    appliance_index = power_series_data[appliance_name].index
                    main_index = main_index.intersection(appliance_index)

                if not is_valid:
                    continue

                for meter in power_series_data.keys():
                    power_series_data[meter] = power_series_data[meter].astype(np.float32).loc[main_index].values

                for meter in power_series_data.keys():
                    if power_series_data[meter].shape != power_series_data['mains'].shape:
                        is_valid = False
                        break

                if not is_valid:
                    continue

                df = pd.DataFrame(
                    power_series_data,
                    dtype=np.float32).dropna()

                if not df.empty:
                    self.data.setdefault(fold, {})[building_name] = df

                logger.info(
                    "Loaded data from building {} for fold {}"
                    " from {} to {}."
                    .format(building_name, fold, df.index[0], df.index[-1]))

        dataset.store.close()
        logger.info("Done loading NILMTK mains data.")

    def _compute_num_sequences_per_building(self):
        index = []
        all_num_seqs = []
        for fold, buildings in self.data.iteritems():
            for building_name, df in buildings.iteritems():
                remainder = len(df) - self.seq_length
                num_seqs = np.ceil(remainder / self.stride) + 1
                num_seqs = max(0 if df.empty else 1, int(num_seqs))
                index.append((fold, building_name))
                all_num_seqs.append(num_seqs)

        multi_index = pd.MultiIndex.from_tuples(
            index, names=["fold", "building_name"])
        self._num_seqs = pd.Series(all_num_seqs, multi_index)

    def get_sequence(self, fold='train', enable_all_appliances=False):
        if enable_all_appliances:
            raise ValueError("`enable_all_appliances` is not implemented yet"
                             " for StrideSource!")

        # select building
        building_divisions = self._num_seqs[fold].cumsum()
        total_seq_for_fold = self._num_seqs[fold].sum()
        building_row_i = 0
        building_name = building_divisions.index[0]
        prev_division = 0
        for seq_i in range(total_seq_for_fold):
            if seq_i == building_divisions.iloc[building_row_i]:
                prev_division = seq_i
                building_row_i += 1
                building_name = building_divisions.index[building_row_i]

            seq_i_for_building = seq_i - prev_division
            start_i = seq_i_for_building * self.stride
            end_i = start_i + self.seq_length
            data_for_seq = self.data[fold][building_name].iloc[start_i:end_i]

            def get_data(col):
                data = data_for_seq[col].values
                n_zeros_to_pad = self.seq_length - len(data)
                data = np.pad(
                    data, pad_width=(0, n_zeros_to_pad), mode='constant')
                return data

            seq = Sequence(self.seq_length, [len(self.appliances), self.seq_length])
            seq.input = get_data('mains')
            for idx, appliance_name in enumerate(self.appliances):
                seq.target[idx] = get_data(appliance_name)

            # Set mask
            seq.weights = np.ones((self.seq_length), dtype=np.float32)
            n_zeros_to_pad = self.seq_length - len(data_for_seq)
            if n_zeros_to_pad > 0:
                seq.weights[-n_zeros_to_pad:] = 0

            # Set metadata
            seq.metadata = {
                'seq_i': seq_i,
                'building_name': building_name,
                'total_num_sequences': total_seq_for_fold,
                'start_date': data_for_seq.index[0],
                'end_date': data_for_seq.index[-1]
            }

            if len(seq.input) != self.seq_length:
                continue

            seq_count = 0
            for i in range(len(self.appliances)):
                if len(seq.target[i]) == self.seq_length:
                    seq_count += 1

            if seq_count != len(self.appliances):
                continue

            yield seq

    @classmethod
    def _attrs_to_remove_for_report(cls):
        return ['data', '_num_seqs', 'rng']

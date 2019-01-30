from __future__ import print_function, division
from copy import copy
import numpy as np
import pandas as pd

from .batch import Batch

import logging
logger = logging.getLogger(__name__)


class Sequence(object):
    """
    Attributes
    ----------
    input : np.ndarray
    target : np.ndarray
    all_appliances : pd.DataFrame
        Column names are the appliance names.
    metadata : dict
    weights : np.ndarray or None
    """
    def __init__(self, shape, tar_shape=None):
	if tar_shape is None:tar_shape=shape
        self.input = np.zeros(shape, dtype=np.float32)
        self.target = np.zeros(tar_shape, dtype=np.float32)
        self.time = np.zeros(shape, dtype=np.float32)
        self.all_appliances = pd.DataFrame()
        self.metadata = {}
        self.weights = None

    def __getstate__(self):
        odict = self.__dict__.copy()  # get attribute dictionary
        return odict

    def __setstate__(self, dict):
        self.__dict__ = dict


class Source(object):
    def __init__(self, rng_seed=None, num_batches_for_validation=16):
        logger.info("------------- INITIALISING {} --------------"
                    .format(self.__class__.__name__))
        self.rng_seed = rng_seed
        self.rng = np.random.RandomState(rng_seed)
        self.num_batches_for_validation = num_batches_for_validation

    def get_sequence(self, validation=False):
        """
        Returns
        -------
        sequence : Sequence
        """
        raise NotImplementedError()

    def get_batch(self, num_seq_per_batch, fold='train',
                  enable_all_appliances=False, validation=False):
        """
        Returns
        -------
        iterators of Batch objects
        """
        seq_iterator = self.get_sequence(
            fold=fold,
            enable_all_appliances=enable_all_appliances)
        stop = False
        batch_i = 0
        while not stop:
            if validation and batch_i == self.num_batches_for_validation:
                break
            input_sequences = []
            target_sequences = []
            time_sequences = []
            weights = []
            all_appliances = {}
            for i in range(num_seq_per_batch):
                try:
                    seq = seq_iterator.next()
                except StopIteration:
                    stop = True
                    if self.appliances is None:
                        seq = Sequence((self.seq_length))
                    else:
                        seq = Sequence(self.seq_length, [len(self.appliances), self.seq_length])
                    seq.weights = np.zeros(
                        (self.seq_length), dtype=np.float32)
                if enable_all_appliances:
                    all_appliances[i] = seq.all_appliances
                input_sequences.append(seq.input[np.newaxis, :])
                time_sequences.append(seq.time[np.newaxis, :])
                target_sequences.append(seq.target[np.newaxis, :, :])
                if seq.weights is not None:
                    weights.append(seq.weights[np.newaxis, :])

            batch = Batch()
            batch.metadata['fold'] = fold
            batch.metadata['source_name'] = self.__class__.__name__
            batch.before_processing.input = np.concatenate(input_sequences)
            del input_sequences
            batch.before_processing.time = np.concatenate(time_sequences)
            del time_sequences
            batch.before_processing.target = np.concatenate(target_sequences)
            del target_sequences
            if enable_all_appliances:
                batch.all_appliances = pd.concat(
                    all_appliances, axis=1, names=['sequence', 'appliance'])
            if weights:
                batch.weights = np.concatenate(weights)
            yield batch
            batch_i += 1

    @classmethod
    def _attrs_to_remove_for_report(cls):
        return ['activations', 'rng']

    def report(self):
        report = copy(self.__dict__)
        report['name'] = self.__class__.__name__
        for attr in self._attrs_to_remove_for_report():
            report.pop(attr, None)
        return report

    def __getstate__(self):
        odict = self.__dict__.copy()  # get attribute dictionary
        return odict

    def __setstate__(self, dict):
        self.__dict__ = dict

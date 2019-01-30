from __future__ import print_function, division
import numpy as np
import pandas as pd
from neuralnilm.data.source import Sequence
from neuralnilm.data.activationssource import ActivationsSource

import logging
logger = logging.getLogger(__name__)


class SyntheticAggregateSource(ActivationsSource):
    def __init__(self, activations, appliances, seq_length,
                 sample_period,
                 distractor_inclusion_prob=0.3,
                 target_inclusion_prob=0.6,
                 uniform_prob_of_selecting_each_building=True,
                 allow_incomplete_target=True,
                 allow_incomplete_distractors=True,
                 include_incomplete_target_in_output=True,
                 rng_seed=None):
        self.activations = activations
        self.appliances = appliances
        self.seq_length = seq_length
        self.sample_period = sample_period
        self.distractor_inclusion_prob = distractor_inclusion_prob
        self.target_inclusion_prob = target_inclusion_prob
        self.uniform_prob_of_selecting_each_building = (
            uniform_prob_of_selecting_each_building)
        self.allow_incomplete_target = allow_incomplete_target
        self.allow_incomplete_distractors = allow_incomplete_distractors
        self.include_incomplete_target_in_output = (
            include_incomplete_target_in_output)
        super(SyntheticAggregateSource, self).__init__(rng_seed=rng_seed)

    def _get_sequence(self, fold='train', enable_all_appliances=False):
        seq = Sequence(self.seq_length, [len(self.appliances), self.seq_length])

        for idx, appliance in enumerate(self.appliances):
            # Target appliance
            if self.rng.binomial(n=1, p=self.target_inclusion_prob):
                building_name = self._select_building(fold, appliance)
                activations = (
                    self.activations[fold][appliance][building_name])
                activation_i = self._select_activation(activations)
                activation = activations[activation_i]
                positioned_activation, is_complete = self._position_activation(
                    activation, is_target_appliance=True)
                positioned_activation = positioned_activation.values

                seq.input += positioned_activation
                seq.target[idx] += positioned_activation

            all_appliances = set(self.activations[fold].keys())
            distractor_appliances = all_appliances - set(self.appliances)

            # Distractor appliances
            distractor_appliances = [
                appliance for appliance in distractor_appliances
                if self.rng.binomial(n=1, p=self.distractor_inclusion_prob)]

            for appliance in distractor_appliances:
                building_name = self._select_building(fold, appliance)
                activations = self.activations[fold][appliance][building_name]
                if len(activations) == 0:
                    continue

                activation_i = self._select_activation(activations)
                activation = activations[activation_i]
                positioned_activation, is_complete = self._position_activation(
                    activation, is_target_appliance=False)
                positioned_activation = positioned_activation.values
                seq.input += positioned_activation
                if enable_all_appliances:
                    all_appliances[appliance] = positioned_activation

        assert len(seq.input) == self.seq_length
        for i in range(len(self.appliances)):
            assert len(seq.target[i]) == self.seq_length

        return seq

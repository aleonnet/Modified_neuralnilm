#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function, division
from collections import defaultdict
import numpy as np
import sklearn.metrics as metrics

class metric_calculation():
    def __init__(self, MULTI_APPLIANCE, target_series, pred_series, acc_threshold = 15):
        # parameter initialization
        self.target = target_series.flatten()
        self.pred = pred_series.flatten()
        self.appliances = MULTI_APPLIANCE
        self.NUM_SEQ_PER_BATCH = target_series.shape[0]
        self.num_appliance = target_series.shape[1]
        self.seq_length = target_series.shape[2]
        self.acc_threshold = acc_threshold

        # data structure for metrics calculation
        self.single_acc_metric = defaultdict(lambda: 0)
        self.energy_metric = defaultdict(lambda: 0)
        self.mse_metric = defaultdict(lambda: 0)
        self.mabs_metric = defaultdict(lambda: 0)
        self.single_confusion_mat = defaultdict(lambda: [0, 0, 0, 0])

    """
    Multi-Task for Relative Error in Total Energy:
        1. relative_error_in_total_energy() is used to get the multi-task energy metric
        2. energy_calculation() is used for calculating the energy metrics of each appliance
        3. print_energy_metrics() present the calculation result after relative_error_in_total_energy() is done
    """
    def relative_error_in_total_energy(self):
        for target_idx in range(self.NUM_SEQ_PER_BATCH - 1):
            power_idx_start = target_idx * self.seq_length * self.num_appliance
            power_idx_end = (target_idx + 1) * self.seq_length * self.num_appliance

            target_series = self.target[power_idx_start:power_idx_end]
            pred_series = self.pred[power_idx_start:power_idx_end]
            for appliance_idx in range(self.num_appliance):
                appliance_energy_error = self.energy_calculation(target_series, pred_series, appliance_idx)
                label = self.appliances[appliance_idx]
                self.energy_metric[label] = self.energy_metric[label] + abs(appliance_energy_error)

        self.print_energy_metrics()

    def energy_calculation(self, target_series, pred_series, appliance_idx):
        start_idx = self.seq_length*appliance_idx
        end_idx = self.seq_length * (appliance_idx+1)

        sum_target = np.sum(target_series[start_idx:end_idx])
        sum_pred = np.sum(pred_series[start_idx:end_idx])
        relative_energy_error = (sum_pred - sum_target) / (max(sum_pred, sum_target) + 0.01)

        return relative_energy_error/self.NUM_SEQ_PER_BATCH

    def print_energy_metrics(self):
        for label in self.appliances:
            print('Relative energy error of', label, '= {:.4f}'.format(self.energy_metric[label]))

    """
    Multi-Task for Precision and Recall Calculation:
        1. precision_calculation() is used for calculating precision of each appliance by confusion metrix
        2. recall_calculation() is used for calculating recall of each appliance by confusion metrix
        3. ROC is used to present the relative F1-score performance
    """
    def print_single_point_precision(self):
        for appliance_idx in range(self.num_appliance):
            label = self.appliances[appliance_idx]
            precision = self.single_confusion_mat[label][0] / (
                        self.single_confusion_mat[label][0] + self.single_confusion_mat[label][2] + 0.001)
            print('Precision of', label,
                  '= {:.4f}'.format(precision))

    def print_single_point_recall(self):
        for appliance_idx in range(self.num_appliance):
            label = self.appliances[appliance_idx]
            recall = self.single_confusion_mat[label][0]/(self.single_confusion_mat[label][0] + self.single_confusion_mat[label][3] + 0.001)
            print('Recall of', label,
                  '= {:.4f}'.format(recall))

    """
    Multi-Task for Loss Function Calculation:
        1. MSE() is used to print the multi-task mse metric
        2. MSE_calculation() is used for calculating multi-task mse metric
        3. appliance_mse() is used for calculating mse metrics of each appliance
        
        4. MABS() is used to print the multi-task mae metric
        5. MABS_calculation() is used for calculating multi-task mae metric
        6. appliance_mabs() is used for calculating mae metrics of each appliance
    """
    def MSE(self):
        self.MSE_calculation(self.target, self.pred)
        for appliance_idx in range(self.num_appliance):
            label = self.appliances[appliance_idx]
            print('MSE of', label, '= {:.4f}'.format(self.mse_metric[label]))
        print('')

    def MSE_calculation(self, target_series, pred_series):
        for target_idx in range(self.NUM_SEQ_PER_BATCH):
            power_idx_start = target_idx * self.seq_length * self.num_appliance
            power_idx_end = (target_idx + 1) * self.seq_length * self.num_appliance
            target = target_series[power_idx_start:power_idx_end]
            pred = pred_series[power_idx_start:power_idx_end]

            for appliance_idx in range(self.num_appliance):
                mse = self.appliance_mse(target, pred, appliance_idx)
                label = self.appliances[appliance_idx]
                self.mse_metric[label] = self.mse_metric[label] + mse

    def appliance_mse(self, target_series, pred_series, appliance_idx):
        mse = 0
        power_idx = self.seq_length*appliance_idx

        for timeStamp in range(self.seq_length):
            label_idx = timeStamp + power_idx
            mse = mse + (target_series[label_idx]-pred_series[label_idx])**2
        return mse/(self.seq_length*self.NUM_SEQ_PER_BATCH)

    def MABS(self):
        self.MABS_calculation(self.target, self.pred)

        for appliance_idx in range(self.num_appliance):
            label = self.appliances[appliance_idx]
            print('MABS of', label, '= {:.4f}'.format(self.mabs_metric[label]))
        print('')

    def MABS_calculation(self, target_series, pred_series):
        for target_idx in range(self.NUM_SEQ_PER_BATCH):
            power_idx_start = target_idx * self.seq_length * self.num_appliance
            power_idx_end = (target_idx + 1) * self.seq_length * self.num_appliance
            target = target_series[power_idx_start:power_idx_end]
            pred = pred_series[power_idx_start:power_idx_end]

            for appliance_idx in range(self.num_appliance):
                mabs = self.appliance_mabs(target, pred, appliance_idx)
                label = self.appliances[appliance_idx]
                self.mabs_metric[label] = self.mabs_metric[label] + mabs

    def appliance_mabs(self, target_series, pred_series, appliance_idx):
        mabs = 0
        power_idx = self.seq_length*appliance_idx

        for timeStamp in range(self.seq_length):
            label_idx = timeStamp + power_idx
            mabs = mabs + abs(target_series[label_idx]-pred_series[label_idx])
        return mabs/(self.seq_length*self.NUM_SEQ_PER_BATCH)

    """
    Multi-Task for the Classification Result:
        1. single_point_acc() print the classification metrics according the one-to-one point result between label and target
           the metrics related to the classification result
           >> accuracy
           >> precision
           >> recall
           >> confusion matrix (CF)
           
        2. single_acc_calculation() is used to calculate the on_off accuracy and confusion matrix
           >> on_off accuracy : if label and target is all above or below the threshold
           >> confusion matrix : a list data for [TP, TN, FP, FN]
           
        3. print_single_point_CF() is used to print the confusion metrix result by one-to-one point method
    """
    def single_point_acc(self):
        for target_idx in range(self.NUM_SEQ_PER_BATCH):
            power_idx_start = target_idx * self.seq_length * self.num_appliance
            power_idx_end = (target_idx + 1) * self.seq_length * self.num_appliance
            target_series = self.target[power_idx_start:power_idx_end]
            pred_series = self.pred[power_idx_start:power_idx_end]

            for appliance_idx in range(self.num_appliance):
                acc = self.single_acc_calculation(target_series, pred_series, appliance_idx)
                label = self.appliances[appliance_idx]
                self.single_acc_metric[label] = self.single_acc_metric[label] + acc

        self.print_single_acc_metric()
        print('')
        self.print_single_point_precision()
        print('')
        self.print_single_point_recall()
        print('')
        self.print_single_point_CF()

    def single_acc_calculation(self, target_series, pred_series, appliance_idx):
        accuracy = 0
        power_idx = self.seq_length*appliance_idx

        for timeStamp in range(self.seq_length):
            label_idx = timeStamp + power_idx
            label = self.appliances[appliance_idx]

            if target_series[label_idx] >= self.acc_threshold:
                if pred_series[label_idx] >= self.acc_threshold:
                    accuracy =  accuracy + 1
                    self.single_confusion_mat[label][0] = self.single_confusion_mat[label][0] + 1
                else:
                    self.single_confusion_mat[label][3] = self.single_confusion_mat[label][3] + 1

            if target_series[label_idx] < self.acc_threshold:
                if pred_series[label_idx] < self.acc_threshold:
                    accuracy =  accuracy + 1
                    self.single_confusion_mat[label][1] = self.single_confusion_mat[label][1] + 1
                else:
                    self.single_confusion_mat[label][2] = self.single_confusion_mat[label][2] + 1

        return accuracy/(self.NUM_SEQ_PER_BATCH*self.seq_length)

    def print_single_acc_metric(self):
        for appliance_idx in range(self.num_appliance):
            label = self.appliances[appliance_idx]
            print('Single point acc of', label, '= {:.4f}%'.format(self.single_acc_metric[label]*100))

    def print_single_point_CF(self):
        for appliance_idx in range(self.num_appliance):
            label = self.appliances[appliance_idx]
            print('TP of', label, ' = ', self.single_confusion_mat[label][0])
            print('TN of', label, ' = ', self.single_confusion_mat[label][1])
            print('FP of', label, ' = ', self.single_confusion_mat[label][2])
            print('FN of', label, ' = ', self.single_confusion_mat[label][3])
            print('')
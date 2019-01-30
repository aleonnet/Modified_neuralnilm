from __future__ import division
from copy import copy
import numpy as np


class Processor(object):
    def report(self):
        report = copy(self.__dict__)
        report['name'] = self.__class__.__name__
        return report

    def inverse(self, data):
        raise NotImplementedError("To be implemented by subclass.")

    def __call__(self, data):
        raise NotImplementedError("To be implemented by subclass.")

class IndependentlyCenter(Processor):
    def __call__(self, data):
        means = data.mean(axis=1, keepdims=True)
        self.metadata = {'IndependentlyCentre': {'means': means}}
        return data - means

class DivideBy(Processor):
    def __init__(self, divisor):
        self.divisor = divisor
        self.target_num = 1

        if not isinstance(self.divisor, (int, float)):
            self.target_num = len(self.divisor)

    def __call__(self, dividend):
        if self.target_num == 1:
            divided_result = dividend / self.divisor
        else:
            batch_size = int(dividend.shape[0] / self.target_num)
            divided_result = [dividend[i * self.target_num + j] / self.divisor[j] for i in range(batch_size) for j in range(self.target_num)]

        divided_result = np.array(divided_result)
        return divided_result.reshape(dividend.shape)

    def inverse(self, quotient):
        if self.target_num == 1:
            inverse_result = quotient * self.divisor
        else:
            batch_size = int(quotient.shape[0] / self.target_num)
            inverse_result = [quotient[i * self.target_num + j] * self.divisor[j] for i in range(batch_size) for j in range(self.target_num)]

        inverse_result = np.array(inverse_result)
        return inverse_result.reshape(quotient.shape)




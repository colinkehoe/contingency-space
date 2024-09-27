import copy
import numpy as np
from utils.confusion_matrix import CM
from utils.confusion_matrix_generalized import CMGeneralized


class DLI:
    def __init__(self, cm: CM):
        """
        Calculates the Doolittle Index (GSR) based on the true classes and the
        predicted ones.

        .. math::

            DLI = ((TP * TN) - (FP * FN))^2 / ((TP + FP) * (TP + FN) * P * N)


        :param cm: an instance of a confusion matrix for which value is required.
        """
        self.cm: CM = cm
        self.value = self.__measure()

    def __measure(self):
        """
        :return: Doolittle Index. Following sklearn's implementation, when
        the denominator is zero, it returns zero.
        """
        cm: CM = copy.deepcopy(self.cm)
        num = np.power(cm.tp * cm.tn - cm.fp * cm.fn, 2)
        den = (cm.tp + cm.fp) * (cm.tp + cm.fn) * cm.p * cm.n
        return num / den if den != 0 else 0

def doolittle_index(cm: CMGeneralized) -> float:
    matrix = copy.deepcopy(cm)
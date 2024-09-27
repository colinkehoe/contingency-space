import copy
import numpy as np
from utils.confusion_matrix import CM
from utils.confusion_matrix_generalized import CMGeneralized


class BAC:
    def __init__(self, cm: CM | CMGeneralized):
        """
        Calculates the balanced accuracy (BAC) based on the true classes and the
        predicted ones.

        .. math::

            BAC = (TPR + TNR) / 2

        Args:
            cm: an instance of a confusion matrix for which value is required.
        """
        self.cm: CM | CMGeneralized = cm
        self.value = self.__measure()

    def __measure(self):
        """
        Returns:
            balanced accuracy. Following sklearn's implementation, when
            the denominator is zero, it returns zero.
        """
        matrix: CM | CMGeneralized = copy.deepcopy(self.cm)
        
        match matrix:
            case CM():
                tpr = matrix.tp / matrix.p if matrix.p != 0 else 0
                tnr = matrix.tn / matrix.n if matrix.n != 0 else 0
                return (tpr + tnr) / 2
            case CMGeneralized():
                positive_rates: list[float] = matrix.positive_rates(return_type=list)
                
                return np.sum(positive_rates) / matrix.num_classes
            case _:
                raise TypeError('Type must be CM or CMGeneralized')

def balanced_accuracy(matrix: CMGeneralized):
    
    positive_rates: list[float] = matrix.positive_rates(return_type=list)
    
    return np.sum(positive_rates) / matrix.num_classes
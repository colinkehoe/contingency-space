import copy
import numpy as np
import utils
from utils.confusion_matrix import CM
from utils.confusion_matrix_generalized import CMGeneralized


class ACC:
    def __init__(self, cm: CM | CMGeneralized):
        """
        Calculates the accuracy (ACC) based on the true classes and the
        predicted ones.

        .. math::

            TSS = (TP + TN) / (P + N)

        Args:
            cm (CM | CMGeneralized): an instance of a confusion matrix for which value is required.
        """
        self.cm: CM | CMGeneralized = cm
        self.value = self.__measure()

    def __measure(self):
        """
        Returns:
            accuracy. Following sklearn's implementation, when
            the denominator is zero, it returns zero.
        """
        matrix: CM | CMGeneralized = copy.deepcopy(self.cm)
        
        
        match matrix:
            case CM():
                
                return (matrix.tp + matrix.tn) / (matrix.p + matrix.n) if matrix.p + matrix.n > 0 else 0
            case utils.confusion_matrix_generalized.CMGeneralized():
                matrix = self.cm.array()
                
                true_values = np.sum(matrix.diagonal())
                total_values = np.sum(matrix)

                return true_values / total_values if total_values > 0 else 0
            case _:
                raise TypeError('Type must be CM or CMGeneralized')

if __name__ == "__main__":
    gen = CMGeneralized({'a': [3, 1, 2, 3],
                         'b': [2, 5, 1, 1],
                         'c': [3, 3, 5, 2],
                         'd': [2, 1, 2, 4]})
    
    test = ACC(gen)
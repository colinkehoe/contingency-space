import copy
import numpy as np
from utils.confusion_matrix import CM
from utils.confusion_matrix_generalized import CMGeneralized


class GSR:
    def __init__(self, cm: CM | CMGeneralized):
        """
        Calculates the Gilbert's success ratio (GSR) based on the true classes and the
        predicted ones.

        .. math::

            GSR = (TP - R) / (TP + FP + FN - R)

            R = ((TP + FP) * (TP + FN)) / (P + N)


        :param cm: an instance of a confusion matrix for which value is required.
        """
        self.cm: CM | CMGeneralized = cm
        self.value = self.__measure()

    def __measure(self):
        """
        :return: Gilbert's success ratio. Following sklearn's implementation, when
        the denominator is zero, it returns zero.
        """
        matrix: CM | CMGeneralized = copy.deepcopy(self.cm)
        
        match matrix:
            case CM():
                r = ((matrix.tp + cm.fp) * (matrix.tp + matrix.fn)) / (matrix.p + matrix.n)
                gs = (matrix.tp - r) / (matrix.tp + matrix.fp + matrix.fn - r)
                return gs
            case CMGeneralized():
                arr = matrix.array()
                
                num_classes: int = matrix.get_num_classes()
                hits: list[int] = arr.diagonal() #hits per class
                misses: list[int] = matrix.get_missed_predictions() #misses per class
                false_alarms: list[int] = matrix.get_false_predictions() #false positives per class
                num_samples: list[int] = matrix.num_samples(per_class=True)
                
                return 0 #NOT DONE
                
if __name__ == "__main__":
    gen = CMGeneralized({'a': [3, 1, 2, 3],
                         'b': [2, 5, 1, 1],
                         'c': [3, 3, 5, 2],
                         'd': [2, 1, 2, 4]})
    
    test = GSR(gen)

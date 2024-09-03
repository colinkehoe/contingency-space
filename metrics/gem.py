import copy
import numpy as np
from utils.confusion_matrix import CM
from utils.confusion_matrix_generalized import CMGeneralized

class GEM:
    def __init__(self, cm: CM):
        """
        Calculates the geometric mean (GEM) based on the true classes and the
        predicted ones.

        .. math::

            GEM = sqrt(TPR * TNR)

        :param cm: an instance of a confusion matrix for which value is required.
        """
        self.cm: CM | CMGeneralized = cm
        self.value = self.__measure()

    def __measure(self):
        """
        :return: geometric mean. Following sklearn's implementation, when
        the denominator is zero, it returns zero.
        """
        matrix: CM | CMGeneralized = copy.deepcopy(self.cm)
        
        geometric_mean = 0
        
        match matrix:
            case CM():
                tpr = matrix.tp / matrix.p if matrix.p != 0 else 0
                tnr = matrix.tn / matrix.n if matrix.n != 0 else 0
                geometric_mean = np.sqrt(tpr + tnr)
            case CMGeneralized():
                positive_rates: list[float] = matrix.positive_rates(return_type=list)
                geometric_mean = np.prod(positive_rates) ** (1 / matrix.num_classes)
            case _:
                raise TypeError('Type must be CM or CMGeneralized.')
            
        return geometric_mean
    
if __name__ == "__main__":
    gen = CMGeneralized({'a': [3, 1, 2, 3],
                         'b': [2, 5, 1, 1],
                         'c': [3, 3, 5, 2],
                         'd': [2, 1, 2, 4]})
    
    test = GEM(gen)
    
    print(test.value)

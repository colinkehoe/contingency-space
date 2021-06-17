import copy
import numpy as np
from utils.confusion_matrix import CM
from metrics.pre import PRE
from metrics.rec import REC
from utils.confusion_matrix_generalized import CMGeneralized

# Ref:
#   1. https://towardsdatascience.com/multi-class-metrics-made-simple-part-i-precision-and-recall-9250280bddc2
#   2. https://towardsdatascience.com/multi-class-metrics-made-simple-part-ii-the-f1-score-ebe8b2c2ca1


class FBS:
    def __init__(self, cm: any([CM, CMGeneralized]), b=1, average: str = None):
        """
        Calculates the fB-Score based on the true classes and the predicted ones. The
        formula for fB-Score is:

        .. math::

            PRE = ((1+B^2) * PRE * REC) / ((B^2 * PRE) + REC)

        :param cm: an instance of a confusion matrix for which value is required.
        :param b: b in fb-score: if b > 1 it weighs recall (b times) higher than precision,
        and if 0 < b < 1, it weighs precision (1/b) times higher than recall.
        """
        self.cm: any([CM, CMGeneralized]) = cm
        self.value = self.__measure(b, average)

    def __measure(self, b=1, average: str = None):
        """
        :param b: see the same argument in the class constructor.
        :param average: default is `None` for binary case. The other two options are `micro` and
        `macro`.

        Note: When using 'micro', total_fp is equal to total_fn, thus pre = rec = f1score = acc.
        Note: Only when `CMGeneralized` is used, `average` would be effective.

        :return: fB-score. Following sklearn's implementation, when the denominator is zero,
        it returns zero.
        """
        cm: any([CM, CMGeneralized]) = copy.deepcopy(self.cm)
        f = 0
        if not average or average.endswith('micro'):
            pre = PRE(cm, average)
            pr = pre.value
            rec = REC(cm, average)
            re = rec.value
            f = ((1 + b ** 2) * pr * re) / ((b ** 2 * pr) + re) if (b ** 2 * pr) + re > 0 else 0
        elif average.endswith('macro'):
            with np.errstate(divide='ignore', invalid='ignore'):
                m = cm.get_matrix()
                col_sum = np.sum(m, axis=0)
                row_sum = np.sum(m, axis=1)
                pr_macro: np.array = m.diagonal() / col_sum
                # pr_macro[pr_macro == np.inf] = 0
                re_macro: np.array = m.diagonal() / row_sum
                # re_macro[re_macro == np.inf] = 0
                numerator = ((1 + b ** 2) * pr_macro * re_macro)
                denominator = ((b ** 2 * pr_macro) + re_macro)

                f = numerator / denominator
                f[f == np.inf] = 0
                f = np.nanmean(f)
        return f

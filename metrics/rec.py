import copy
import numpy as np
from utils.confusion_matrix import CM
from utils.confusion_matrix_generalized import CMGeneralized


class REC:
    def __init__(self, cm: any([CM, CMGeneralized]), average: str = None):
        """
        Calculates the Recall based on the true classes and the predicted ones. The
        formula for Recall is:

        .. math::

            PRE = TP / (TP + FN)

        :param cm: an instance of a confusion matrix for which value is required.
        :param average: default is `None` for binary case. The other two options are `micro` and
        `macro`. When `micro` is used, we get the total_tp, total_fp, and total_fn from
        `CMGeneralized` and then using the above formula, calculate `micor_recall`. When
        `macro` is used, first per-class recall is calculated; for i-th class, i-th entry on
        diagonal divided by the sum of i-th row. Then, the average of those recalls will be
        returned as the `macro_recall`.
        """
        self.cm: any([CM, CMGeneralized]) = cm
        self.value = self.__measure(average)

    def __measure(self, average: str):
        """
        :param average: default is `None` for binary case. The other two options are `micro` and
        `macro`.

        Note: When using 'micro', total_fp is equal to total_fn, thus pre = rec = f1score = acc.
        Note: Only when `CMGeneralized` is used, `average` would be effective.

        :return: recall. Following sklearn's implementation, when
        the denominator is zero, it returns zero.
        """
        cm: any([CM, CMGeneralized]) = copy.deepcopy(self.cm)
        re = 0
        if isinstance(cm, CM):
            re = cm.tp / (cm.tp + cm.fn) if cm.tp + cm.fn > 0 else 0
        elif isinstance(cm, CMGeneralized):
            if average.endswith('micro'):
                tp = cm.get_total_t()
                fn = cm.get_total_fn()
                re = tp / (tp + fn) if tp + fn > 0 else 0
            elif average.endswith('macro'):
                m = cm.get_matrix()
                row_sum = np.sum(m, axis=1)
                rec_macro: np.array = m.diagonal() / row_sum
                rec_macro[rec_macro == np.inf] = 0
                re = np.nanmean(rec_macro)

        return re

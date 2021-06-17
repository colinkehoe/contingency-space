import copy
import numpy as np
from utils.confusion_matrix import CM
from utils.confusion_matrix_generalized import CMGeneralized


class PRE:
    def __init__(self, cm: any([CM, CMGeneralized]), average: str = None):
        """
        Calculates the Precision based on the true classes and the predicted ones. The
        formula for Precision in binary case is:

        .. math::

            PRE = TP / (TP + FP)

        :param cm: an instance of a confusion matrix for which value is required.
        :param average: default is `None` for binary case. The other two options are `micro` and
        `macro`. When `micro` is used, we get the total_tp, total_fp, and total_fn from
        `CMGeneralized` and then using the above formula, calculate `micor_precision`. When
        `macro` is used, first per-class precision is calculated; for i-th class, i-th entry on
        diagonal divided by the sum of i-th column. Then, the average of those precisions will be
        returned as the `macro_precision`.
        """
        self.cm: any([CM, CMGeneralized]) = cm
        self.value = self.__measure(average)

    def __measure(self, average: str):
        """
        :param average: default is `None` for binary case. The other two options are `micro` and
        `macro`.

        Note: When using 'micro', total_fp is equal to total_fn, thus pre = rec = f1score = acc.
        Note: Only when `CMGeneralized` is used, `average` would be effective.

        :return: precision. Following sklearn's implementation, when
        the denominator is zero, it returns zero.
        """
        cm: any([CM, CMGeneralized]) = copy.deepcopy(self.cm)
        pr = 0
        if isinstance(cm, CM):
            pr = cm.tp / (cm.tp + cm.fp) if cm.tp + cm.fp > 0 else 0
        elif isinstance(cm, CMGeneralized):
            if average.endswith('micro'):
                tp = cm.get_total_tp()  # todo: get_total_tp was changed to get_total_t as its implementations was wrong!
                fp = cm.get_total_fp()
                pr = tp / (tp + fp) if tp + fp > 0 else 0
            elif average.endswith('macro'):
                m = cm.get_matrix()
                col_sum = np.sum(m, axis=0)
                pre_macro: np.array = m.diagonal() / col_sum
                pre_macro[pre_macro == np.inf] = 0
                pr = np.nanmean(pre_macro)
        return pr

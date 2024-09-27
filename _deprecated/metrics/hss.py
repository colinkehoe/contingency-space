import numpy as np
from utils.confusion_matrix import CM


class HSS:
    def __init__(self, cm: CM):
        """
        Calculates the Heidke Skill Score (HSS) based on the formula employed by the Space
        Weather Prediction Center for flare forecasting. See Balch 2008 for more details.

        .. math::

            HSS2 = 2[(TP * TN) - (FN * FP)] / [(P * (FN + TN)] + [(TP + FP) * N)]

        :param cm: an instance of a confusion matrix for which value is required.
        """
        self.cm: CM = cm
        self.value = self.__measure()

    def __measure(self):
        """
        :return: Heidke skill score. Following sklearn's implementation of other metrics, when
        the denominator is zero, it returns zero.
        """
        tp, tn, fp, fn = self.cm.tp, self.cm.tn, self.cm.fp, self.cm.fn
        numer = 2 * ((tp * tn) - (fn * fp))
        denom = ((tp + fn) * (fn + tn)) + ((tp + fp) * (fp + tn))
        hss = (numer / float(denom)) if denom != 0 else 0
        return hss


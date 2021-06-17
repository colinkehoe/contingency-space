import copy
import numpy as np
from utils.confusion_matrix import CM


class TSS:
    def __init__(self, cm: CM):
        """
        Calculates the True Skill Statistic (TSS) based on the true classes and the predicted ones.
        TSS is also called  Hansen-Kuipers Skill Score or Peirce Skill Score. For more details,
        see Bobra & Couvidat (2015), or Bloomfield et al. (2012).

        .. math::

            TSS = (TP / (TP + FN)) - (FP / (FP + TN))

        :param cm: an instance of a confusion matrix for which value is required.
        """
        self.cm: CM = cm
        self.value = self.__measure()

    def __measure(self):
        """
        :return: true skill statistic. Following sklearn's implementation, when
        the denominator is zero, it returns zero.
        """
        cm: CM = copy.deepcopy(self.cm)
        tp_rate = cm.tp / cm.p if cm.p > 0 else 0
        fp_rate = cm.fp / cm.n if cm.n > 0 else 0
        return tp_rate - fp_rate


# https://bitbucket.org/gsudmlab/swan_features/src/master/util/metrics.py
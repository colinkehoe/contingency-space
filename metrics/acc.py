import copy
from utils.confusion_matrix import CM


class ACC:
    def __init__(self, cm: CM):
        """
        Calculates the accuracy (ACC) based on the true classes and the
        predicted ones.

        .. math::

            TSS = (TP + TN) / (P + N)

        :param cm: an instance of a confusion matrix for which value is required.
        """
        self.cm: CM = cm
        self.value = self.__measure()

    def __measure(self):
        """
        :return: accuracy. Following sklearn's implementation, when
        the denominator is zero, it returns zero.
        """
        cm: CM = copy.deepcopy(self.cm)
        return (cm.tp + cm.tn) / (cm.p + cm.n) if cm.p + cm.n > 0 else 0

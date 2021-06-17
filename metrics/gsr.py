import copy
from utils.confusion_matrix import CM


class GSR:
    def __init__(self, cm: CM):
        """
        Calculates the Gilbert's success ratio (GSR) based on the true classes and the
        predicted ones.

        .. math::

            GSR = (TP - R) / (TP + FP + FN - R)

            R = ((TP + FP) * (TP + FN)) / (P + N)


        :param cm: an instance of a confusion matrix for which value is required.
        """
        self.cm: CM = cm
        self.value = self.__measure()

    def __measure(self):
        """
        :return: Gilbert's success ratio. Following sklearn's implementation, when
        the denominator is zero, it returns zero.
        """
        cm: CM = copy.deepcopy(self.cm)
        r = ((cm.tp + cm.fp) * (cm.tp + cm.fn)) / (cm.p + cm.n)
        gs = (cm.tp - r) / (cm.tp + cm.fp + cm.fn - r)
        return gs

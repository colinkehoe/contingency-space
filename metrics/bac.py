import copy
from utils.confusion_matrix import CM


class BAC:
    def __init__(self, cm: CM):
        """
        Calculates the balanced accuracy (BAC) based on the true classes and the
        predicted ones.

        .. math::

            BAC = (TPR + TNR) / 2

        :param cm: an instance of a confusion matrix for which value is required.
        """
        self.cm: CM = cm
        self.value = self.__measure()

    def __measure(self):
        """
        :return: balanced accuracy. Following sklearn's implementation, when
        the denominator is zero, it returns zero.
        """
        cm: CM = copy.deepcopy(self.cm)
        tpr = cm.tp / cm.p if cm.p != 0 else 0
        tnr = cm.tn / cm.n if cm.n != 0 else 0
        return (tpr + tnr) / 2

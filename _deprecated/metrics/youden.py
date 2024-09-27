from utils.confusion_matrix import CM


class YDN:
    def __init__(self, cm: CM):
        """
        Calculates the Youden Index (Youden, William J. "Index for rating
        diagnostic tests." Cancer 3.1 (1950): 32-35.) based on the following
        formula:

        .. math::

            ydn = sensitivity - (1 - specificity)
            ydn = tpr - fpr = (TP/P) - (FP/N)

        :param cm: an instance of a confusion matrix for which value is required.
        """
        self.cm: CM = cm
        self.value = self.__measure()

    def __measure(self):
        """
        :return: Youden index. Here for special cases, we followed sklearn's
        strategy that if the denominator is zero, a metric should give returns zero.
        """
        tp, tn, fp, fn = self.cm.tp, self.cm.tn, self.cm.fp, self.cm.fn
        tpr = (tp / (tp + fn)) if (tp + fn) != 0 else 0
        fpr = (fp / (tn + fp)) if (tn + fp) != 0 else 0
        ydn = tpr - fpr
        return ydn


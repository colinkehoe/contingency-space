import copy
import numpy as np
from utils.confusion_matrix import CM
from utils.confusion_matrix_generalized import CMGeneralized


class TauGeneralized:
    """
    This is the general form of Tau and works on multi-class verification instead of a binary
    verification.
    """
    def __init__(self, cm: any([CM, CMGeneralized]), do_normalize: bool = True):
        """
        :param cm:
        :param do_normalize: if `True`, normalize TP and FN with respect to P, and TN and FP with
        respect to N.
        """
        self.cm: any([CM, CMGeneralized]) = cm
        self.model_point, self.perfect_point, self.random_point = self.__measure(do_normalize)
        self.dist_from_random = self.__get_dist_from_random()
        self.dist_from_perfect = self.__get_dist_from_perfect()
        self.value = self.__get_tau(do_normalize)

    def __measure(self, do_normalize: bool = True):
        """
        :param do_normalize: see the docstring of the class constructor.
        :return:
        """
        model_point, perfect_point, random_point = None, None, None
        cm: any([CM, CMGeneralized]) = copy.deepcopy(self.cm)
        if do_normalize:
            cm.normalize()
            if isinstance(self.cm, CM):
                perfect_point = np.array([1, 1])
            elif isinstance(self.cm, CMGeneralized):
                perfect_point = np.ones(self.cm.dim)
        else:
            if isinstance(self.cm, CM):
                perfect_point = np.array([cm.tn + cm.fp, cm.tp + cm.fn])
            elif isinstance(self.cm, CMGeneralized):
                perfect_point = np.array(list(cm.class_freqs.values()))

        # Position of performance for "the model"
        if isinstance(self.cm, CM):
            model_point = np.array([cm.tn, cm.tp])
        elif isinstance(self.cm, CMGeneralized):
            model_point = np.array(list(cm.table.values())).diagonal()

        # Position of performance for "Random-Guess Model"
        if isinstance(self.cm, CM):
            random_point = np.array([(cm.tn + cm.fp) / 2, (cm.tp + cm.fn) / 2])
        elif isinstance(self.cm, CMGeneralized):
            random_point = np.array([r / 2 for r in cm.class_freqs.values()])

        return model_point, perfect_point, random_point

    def __get_dist_from_random(self):
        """
        :return: the euclidean distance from the model's point to random-guess's point.
        """
        return np.linalg.norm(self.model_point - self.random_point)

    def __get_dist_from_perfect(self):
        """
        :return: the euclidean distance from the model's point to the perfect model's point (i.e.
        Origin = (0, 0)).
        """
        return np.linalg.norm(self.model_point - self.perfect_point)

    def __get_tau(self, do_normalize: bool = True):
        """
        normalizes `dist_from_perfect` so that it ranges from 0 to 1.
        If the CM is not normalized, it simply returns `dist_from_perfect`
        as is.
        :return: normalized `dist_from_perfect`.
        """
        dim = 2
        if isinstance(self.cm, CMGeneralized):
            dim = self.cm.dim
        if do_normalize:
            dist_upper_bound = np.sqrt(dim)
            tau = 1 - (self.dist_from_perfect / dist_upper_bound)
        else:
            tau = self.dist_from_perfect
        return tau


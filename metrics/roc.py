import copy
import numpy as np
from utils.confusion_matrix import CM


class ROC:
    """
    This is similar to Tau, but in the ROC space. Here the distance from (0,1) accounts for the
    distance.
    """
    def __init__(self, cm: CM, do_normalize: bool = True):
        """
        :param cm:
        :param do_normalize: if `True`, normalize TP and FN with respect to P, and TN and FP with
        respect to N.
        """
        self.cm: CM = cm
        self.model_point, self.perfect_point, self.random_point = self.__measure(do_normalize)
        self.dist_from_random = self.__get_dist_from_random()
        self.dist_from_perfect = self.__get_dist_from_perfect()
        self.value = self.__get_dist(do_normalize) # ** 2

    def __measure(self, do_normalize: bool = True):
        """
        :param do_normalize: see the docstring of the class constructor.
        :return:
        """
        cm: CM = copy.deepcopy(self.cm)
        if do_normalize:
            cm.normalize()
            perfect_point = np.array([0, 1])
        else:
            perfect_point = np.array([0, cm.tp + cm.fn])
        # Position of performance for "the model"
        model_point = np.array([cm.fp, cm.tp])

        # Position of performance for "Random-Guess Model"
        random_point = np.array([(cm.fp + cm.tn) / 2, (cm.tp + cm.fn) / 2])
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

    def __get_dist(self, do_normalize: bool = True):
        """
        normalizes `dist_from_perfect` so that it ranges from 0 to 1.
        If the CM is not normalized, it simply returns `dist_from_perfect`
        as is.
        :return: normalized `dist_from_perfect`.
        """
        if do_normalize:
            dist_upper_bound = np.sqrt(2)
            d = 1 - (self.dist_from_perfect / dist_upper_bound)
        else:
            d = self.dist_from_perfect
        return d


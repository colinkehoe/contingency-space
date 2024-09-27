import copy
import numpy as np
from utils.confusion_matrix import CM


def __clean_weights(weights):
    if isinstance(weights, np.ndarray):
        return weights
    elif isinstance(weights, list):
        return np.array(weights)


class TauW:
    """
    This is the new metric I am now investigating. It forms two axes by stacking the values tp
    and fn (from the confusion matrix) as the y-axis and tn and fp as the x-axis. It normalizes
    each axis with respect to p and n, respectively. The point located at (x=tn, y=tp) represents
    the model under study in the space of all possible performances. Moreover, the model at (1, 1)
    of this 2D space represents the Perfect model, while the one at (0.5, 0.5) represents the
    Random-guess model's performance.

    This space provides a visualization for each model with respect to the Perfect model and also
    with respect to the Random-guess mode. The distance between any given model and the
    Random-guess model indicates how well the model performs. Similarly, its performance can be
    measured with respect to the Perfect model. Our investigation shows that comparing a model with
    respect to the Perfect model is more meaningful.

    Non-normalized version of tau can be formulated as follows::

            tua = sqrt((fp/n)^2 + (fn/p)^2)
    in other words::

            tau = sqrt( FPR^2 + FNR ^ 2)
    where FPR and FNR are False-Positive Rate and False-Negative Rate, respectively.

    This metric also provides means to tweak the space by introducing independent weights along
    each axis.

    The tua value this class calculates is the normalized and rescaled version of
    `dist_from_perfect`. For normalization, since each axis is normalized, it is enough to divide
    `dist_from_perfect` by `sqrt(2)`. Subtracting the result from 1 gives a value in the range
    [0, 1], where higher values shows better (closer to perfect) performances.

    When weights are used, some extra steps are needed for calculating tau. Suppose w and h are
    the width and height of the line connecting the model to the Perfect model. The scalars a and b
    are considered as the weights for w and h, respectively. Then, from::

            sqrt(w^2 + h^2) <= sqrt(2)

    we can derive::

            sqrt(aw^2 + bh^2) <= sqrt(2a + (b-a)h)

    and since h <= 1::

            sqrt(aw^2 + bh^2) <= sqrt(a + b).
    """

    def __init__(self, cm: CM, do_normalize: bool = True, weights: np.array = np.array([1, 1, 1])):
        """
        The class constructor.

        :param cm:
        :param do_normalize: if `True`, normalize TP and FN with respect to P, and TN and FP with
        respect to N.
        """
        self.weights = copy.deepcopy(weights)
        self.cm: CM = cm
        self.model_point, self.perfect_point, self.random_point = self.__measure(do_normalize)
        self.dist_from_random = self.__get_dist_from_random()
        self.dist_from_perfect = self.__get_dist_from_perfect(self.weights)
        self.value = self.__get_tau(do_normalize, self.weights)

    def __measure(self, do_normalize: bool = True):
        """
        measures the performance.

        :param do_normalize: see the docstring of the class constructor.
        :return:
        """
        cm: CM = copy.deepcopy(self.cm)
        if do_normalize:
            cm.normalize()
            perfect_point = np.array([1, 1])
        else:
            perfect_point = np.array([cm.tn + cm.fp, cm.tp + cm.fn])
        # Position of performance for "the model"
        model_point = np.array([cm.tn, cm.tp])

        # Position of performance for "Random-Guess Model"
        random_point = np.array([(cm.tn + cm.fp) / 2, (cm.tp + cm.fn) / 2])
        return model_point, perfect_point, random_point

    def __get_dist_from_random(self):
        """
        :return: the euclidean distance from the model's point to random-guess's point.
        """
        return np.linalg.norm(self.model_point - self.random_point)

    def __get_dist_from_perfect(self, weights: np.array = np.array([1, 1, 1])):
        """
        :param weights: weights of the negative and positive classes, respectively. For example
        [2.0, 1.0] puts twice as much weight on the contribution of TN on distance. So,
        the space of all possible Taus is no longer symmetric, although the concept of 'distance'
        implies the symmetry.

        :return: the euclidean distance from the model's point to the perfect model's point (i.e.
        Origin = (0, 0)).
        """
        if all(weights == np.array([1, 1, 1])):
            return np.linalg.norm(self.model_point - self.perfect_point)
        else:
            x0, y0 = self.model_point
            x1, y1 = self.perfect_point
            d = np.sqrt(np.power(y0 - y1, 2) * weights[0] +
                        np.power(x0 - x1, 2) * weights[1])
            return d

    def __get_tau(self, do_normalize: bool = True, weights: np.array = np.array([1, 1, 1])):
        """
        normalizes `dist_from_perfect` so that it ranges from 0 to 1.
        If the CM is not normalized, it simply returns `dist_from_perfect`
        as is.
        :return: normalized `dist_from_perfect`.
        """
        if do_normalize:
            a, b, c = weights
            dist_upper_bound = np.sqrt(a + b)
            tau = 1 - (self.dist_from_perfect / dist_upper_bound)
            tau = np.power(tau, c)
        else:
            tau = self.dist_from_perfect
        return tau

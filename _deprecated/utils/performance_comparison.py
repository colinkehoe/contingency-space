from utils.confusion_matrix import CM
import numpy as np
import copy


def _verify_cm(cm: CM):
    if cm.n <= 1.0 or cm.p <= 1.0:
        raise ValueError(
            """
            Error! The original (non-normalized) confusion matrices are needed.

            Normalization is a lossy process: a normalized confusion matrix represents infinitely many
            (non-identical) confusion matrices. This makes a metric to return the same value for originally different
            confusion matrices. For measuring the 2D and 3D distances between two confusion matrices, however, the
            confusion matrices may be internally normalized using the optional argument `normalize`. 
            """
        )


class PerformanceComparison:
    """
    This class provides different ways that performance of two models given by their confusion matrices can be compared.

    Note that all comparison methods are order sensitive, i.e., the order of which the confusion matrices are passed in
    the methods impact the results by a positive or negative sign.
    """
    def __init__(self, mu, normalize=True):
        """
        The class constructor only initializes the class fields.

        :param mu: the performance evaluation metric (from any of the classes in `metrics`)
        :param normalize: determines whether the confusion matrices should be normalized before their distance is
        measured or not. The default value is `True`. Regardless of this choice, the metric `mu` (in
        `compare_by_metric`, `compare_by_3d_distance`, and `compare_by_3d_triangle`) will be calculated on the
        non-normalized confusion matrices. The choice matters only on `compare_by_2d_distance` and other methods where
        it is called. An error will be returned if the confusion matrices are already normalized.
        """
        self.mu = mu
        self.normalize = normalize

    def compare_by_metric(self, cm_1: CM, cm_2: CM):
        """
        compares two CMs by calculating the given metric `mu` on each of them and finding the difference.

        Note that this method computes mu(cm_2) - mu(cm_1), so the order matters.
        :return: mu(cm_2) - mu(cm_1). The outcome could be positive or negative.
        """
        # make sure the confusion matrices are NOT normalized
        _verify_cm(cm_1)
        _verify_cm(cm_2)
        diff = self.mu(cm_2).value - self.mu(cm_1).value
        return diff

    def compare_by_2d_distance(self, cm_1: CM, cm_2: CM):
        """
        computes the distance (always positive) between the two given confusion matrices in the 2d contingency space.

        :param cm_1: the first confusion matrix of the two consecutive ones.
        :param cm_2: the second confusion matrix of the two consecutive ones.

        :return: the distance.
        """
        cm_1 = copy.deepcopy(cm_1)
        cm_2 = copy.deepcopy(cm_2)
        if self.normalize:
            cm_1.normalize()
            cm_2.normalize()
        p1 = np.array([cm_1.tn, cm_1.tp])
        p2 = np.array([cm_2.tn, cm_2.tp])
        return np.linalg.norm(p2 - p1)

    def compare_by_3d_distance(self, cm_1: CM, cm_2: CM):
        """
        computes the distance (always positive) between the two given confusion matrices in the 3d contingency space.

        :param cm_1: the first confusion matrix of the two consecutive ones.
        :param cm_2: the second confusion matrix of the two consecutive ones.

        :return: the distance.
        """
        opposite = self.compare_by_metric(cm_1, cm_2)  # could be positive or negative
        adjacent = self.compare_by_2d_distance(cm_1, cm_2)
        hypotenuse = np.sqrt(np.power(opposite, 2) + np.power(adjacent, 2))
        return hypotenuse

    def compare_by_3d_triangle(self, cm_1: CM, cm_2: CM):
        """
        compares the two given confusion matrices by considering them as two model points in the Contingency Space
        and computing the area of the right triangle they form. This triangle's opposite and adjacent sides are
        as follows:

            - adjacent: the distance between cm_1 and cm_2 determined by `metric` on the Z axis. The method
            `compare_by_metric` is called to compute this.
            - opposite: the distance between cm_1 and cm_2 determined by their `tp` and `tn` values. The method
            `compare_by_2d_distance` is called to compute this.

        :param cm_1: the first confusion matrix of type `CM` (from utils.confusion_matrix)
        :param cm_2: the second confusion matrix of type `CM`.
        :return: the area of the triangle formed by the two confusion matrices. Positive values indicate preference of
        `cm_2` over `cm_1` by the utilized metric. Negative values indicate otherwise.
        """
        opposite = self.compare_by_metric(cm_1, cm_2)
        adjacent = self.compare_by_2d_distance(cm_1, cm_2)
        return 0.5 * opposite * adjacent


def main():
    from metrics.fbs import FBS
    n, p = 4900, 100
    cm_random = CM({'tp': p / 2, 'fn': p / 2, 'tn': n / 2, 'fp': n / 2})
    cm_perfect = CM({'tp': p, 'fn': 0, 'tn': n, 'fp': 0})
    pc = PerformanceComparison(FBS, normalize=True)
    print(pc.compare_by_metric(cm_random, cm_perfect))
    print(pc.compare_by_2d_distance(cm_random, cm_perfect))
    print(pc.compare_by_3d_distance(cm_random, cm_perfect))
    print(pc.compare_by_3d_triangle(cm_random, cm_perfect))


if __name__ == "__main__":
    main()

import copy
import numpy as np
from utils.confusion_matrix import CM
from utils.performance_comparison import PerformanceComparison


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


class LearningPath:
    def __init__(self, path: list, metric, normalize=True):
        """
        The class constructor that initializes the fields.

        Note that the confusion matrices should be passed in without being normalized. A normalized confusion matrix
        represents infinitely many (non-identical) confusion matrices. The original confusion matrix is needed to
        calculate the given `metric`, and then for measuring the adjacent side of the desired triangle, the confusion
        matrices may be internally normalized using the optional argument `normalize`.

        :param path: a list of confusion matrices each of type CM.
        :param metric: the metric to be used for calculating the relative improvements.
        :param normalize: determines whether the confusion matrices should be normalized before their distance is
        measured or not. The default value is `True`. Regardless of this choice, the `metric` will be calculated on the
        non-normalized confusion matrices. An error will be returned if the confusion matrices are already normalized.
        """
        _verify_cm(path[0])
        self.normalize = normalize
        self.path = copy.deepcopy(path)
        self.metric = metric
        self.scores = []
        self.score_changes = []  # tracking of score improvements measured by `metric`
        self.cm_steps = []  # tracking of moves in CM space (only magnitudes, no directions)
        self.cs_steps = []  # tracking of moves in Contingency Space (only magnitudes, no directions)
        self.triangle_areas = []  # tracking of area of formed triangles in Contingency Space

        self.cm_path_length = 0  # sum of all `self.cm_steps`
        self.cs_path_length = 0  # sum of all `self.cs_steps`
        self.net_score_change = 0  # sum of all `self.score_changes` taking into account the signs.
        self.net_triangle_area = 0  # sum of all `self.triangle_areas` taking into account the signs.

    def compute_impact(self):
        """
        computes the impact of the given learning path.

        :return: the impact.
        """
        cm_steps, cs_steps, score_changes, scores, triangle_areas = [], [], [], [], []

        idx = 0
        for idx in range(len(self.path) - 1):
            a, b = self.path[idx], self.path[idx + 1]  # a pair of consecutive CMs
            score_a = self.metric(a).value
            scores.append(score_a)

            # calculate `metric` before normalization of CMs
            pc = PerformanceComparison(self.metric, normalize=self.normalize)
            h = pc.compare_by_metric(a, b)
            score_changes.append(h)

            # calculate 2d hypotenuse
            hypotenuse_2d = pc.compare_by_2d_distance(a, b)
            cm_steps.append(hypotenuse_2d)
            # calculate 3d hypotenuse
            hypotenuse_3d = pc.compare_by_3d_distance(a, b)
            cs_steps.append(hypotenuse_3d)

            # compare by triangle area
            area = pc.compare_by_3d_triangle(a, b)
            triangle_areas.append(area)

        scores.append(self.metric(self.path[idx + 1]).value)  # compute the last CM

        self.__update_improvements(scores, score_changes, cm_steps, cs_steps, triangle_areas)

    def __update_improvements(self, scores: list, score_changes, cm_steps, cs_steps, triangle_areas):
        """
        Copies all calculated statistics onto the class fields.

        :param scores: a list of all scores measured using the employed metric.
        :param score_changes: a list of all score changes measured by `compare_by_metric` from
        `utils.performance_comparison.py`.
        :param cm_steps: a list of all distances of 2D moves measured by `compare_by_2d_distance` from
        `utils.performance_comparison.py`.
        :param cs_steps: a list of all distances of 3D moves measured by `compare_by_3d_distance` from
        `utils.performance_comparison.py`.
        :param triangle_areas: a list of areas of all triangles measured by `compare_by_3d_triangle` from
        `utils.performance_comparison.py`.
        :return: None.
        """
        self.scores = copy.deepcopy(scores)
        self.score_changes = copy.deepcopy(score_changes)
        self.cm_steps = copy.deepcopy(cm_steps)
        self.cs_steps = copy.deepcopy(cs_steps)
        self.triangle_areas = copy.deepcopy(triangle_areas)
        self.cm_path_length = np.sum(self.cm_steps)
        self.cs_path_length = np.sum(self.cs_steps)
        self.net_score_change = np.sum(self.score_changes)
        self.net_triangle_area = np.sum(self.triangle_areas)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from metrics import *
from utils.confusion_matrix import CM
from utils.confusion_matrix_generalized import CMGeneralized
from utils.cm_generator import CMGenerator
from fractions import Fraction

def imbalance_sensitivity(imbalance_ratio: float, metric: type = ACC) -> float:
    """_summary_

    Args:
        imbalance_ratio (tuple[int, int]): _description_
        metric (type, optional): _description_. Defaults to ACC.

    Returns:
        float: _description_
    """
    
    #generate the matrices from the CMGenerator. 
    
    num_classes = 2
    fraction = Fraction(imbalance_ratio)
    
    #generate the imbalanced matrices
    n_per_class_imbalanced = {'t': fraction.numerator*1000, 'f': (fraction.denominator)*1000}
    matrices_imbalanced = CMGenerator(num_classes, (fraction.numerator+fraction.denominator)*1000, n_per_class_imbalanced)
    matrices_imbalanced.generate_cms(10)
    
    #generate the balanced matrices
    n_per_class_balanced = {'t': (fraction.denominator / 2)*1000, 'f': (fraction.denominator / 2)*1000}
    matrices_balanced = CMGenerator(num_classes, fraction.denominator*1000, n_per_class_balanced)
    matrices_balanced.generate_cms(10)
    
    #generate each metric object using the matrices we just generated. 
    imbalanced_metrics = [metric(matrix) for matrix in matrices_imbalanced.all_cms]
    balanced_metrics = [metric(matrix) for matrix in matrices_balanced.all_cms]
    
    #generate each score for every matrix. 
    imbalanced_metric_scores = [metric.value for metric in imbalanced_metrics]
    balanced_metric_scores = [metric.value for metric in balanced_metrics]
    
    #calculate the distance between each set of points.
    point_distances = []
    for i, b in zip(imbalanced_metric_scores, balanced_metric_scores):
        point_distances.append(abs(i - b))

    volume = sum(point_distances)    
    return volume

if __name__ == "__main__":
    print(imbalance_sensitivity(0.5))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from metrics import *
from utils.confusion_matrix import CM
from utils.confusion_matrix_generalized import CMGeneralized
from utils.cm_generator import CMGenerator
from fractions import Fraction
from typing import Callable

def calculate_scores(matrices: list[CMGeneralized], metric: Callable[[CMGeneralized], float]) -> float:
    all_scores = []
    for cm in matrices:
        m = metric(cm)
        all_scores.append(m)
    return all_scores

def imbalance_sensitivity(imbalance: float | int | str | tuple[int, int], metric: Callable[[CMGeneralized], float], granularity: int = 15) -> float:
    """Calculates the sensitivity of a given metric to a given imbalance ratio. 

    Args:
        imbalance (float | int): An integer representing the larger half of the imbalance ratio, or float containing the numerator and denominator
        metric (Callable[[CMGeneralized], float]): A function that calculates a metric given a Confusion Matrix. Should return a float. 
        granularity (int, optional): The number of points along each axis to generate confusion matrices from. Defaults to 15.

    Returns:
        float: A value representing the sensitivity of the given metric to the imbalance ratio. 
    """
    num_classes = 2
    numerator = 1
    denominator = 1
    
    #parse out the imbalance ratio
    match imbalance:
        #case they passed only the denom
        case int():
            numerator = 1
            denominator = imbalance
        #parse the num and denom from the decimal
        case float():
            decimal_part = str(imbalance).split('.')[1]
            
            half_len = len(decimal_part) // 2
            
            first_half = decimal_part[:half_len]
            second_half = decimal_part[half_len:]
            
            numerator = int(first_half) if first_half else 1
            denominator = int(second_half) if second_half else 1
        #parse num and denom from string
        case str():
            parts = imbalance.split(':')
            
            try:
                numerator = int(parts[0])
                denominator = int(parts[1])
            except:
                raise ValueError("Numerator and denominator must be uninterrupted integers.")
        case tuple():
            if len(imbalance) > 2:
                raise ValueError("Ratio must only contain two values")
            
            (numerator, denominator) = imbalance
            
            if not isinstance(numerator, int) or not isinstance(denominator, int):
                raise TypeError('Values in tuple must be of type int.')
        case _:
            raise TypeError("Check valid types for imbalance ratio")
        
    power: int = 0
    
    #ensure the parts of the ratio are of suitable size for the calculation
    while numerator < 1000 or not (numerator.is_integer() and denominator.is_integer()):
        numerator *= 10
        denominator *= 10
        power += 1
    
    #generate the imbalanced matrices
    n_per_class_imbalanced: dict[str, int] = {'t': numerator, 'f': denominator}
    matrices_imbalanced = CMGenerator(num_classes, (numerator + denominator), n_per_class_imbalanced)
    matrices_imbalanced.generate_cms(granularity)
    
    #generate the balanced matrices
    n_per_class_balanced: dict[str, int] = {'t': int((denominator / 2)*1000), 'f': int((denominator / 2)*1000)}
    matrices_balanced = CMGenerator(num_classes, denominator*1000, n_per_class_balanced)
    matrices_balanced.generate_cms(granularity)
    
    #calculate the scores for all cms
    imbalanced_scores = calculate_scores(matrices_imbalanced.all_cms, metric)
    balanced_scores = calculate_scores(matrices_balanced.all_cms, metric)
    
    #re-organize the matrices so that they are aligned as they belong on a contingency space
    imbalanced_scores_as_mat = np.flip(np.array(imbalanced_scores).reshape((granularity, granularity)), 0)
    balanced_scores_as_mat = np.flip(np.array(balanced_scores).reshape((granularity, granularity)), 0)

    
    #pairwise difference between points
    differences = imbalanced_scores_as_mat - balanced_scores_as_mat
    
    print(np.sum(np.abs(differences)))
    #return the 
    return np.sum(np.abs(differences)) / pow(granularity, num_classes)

if __name__ == "__main__":
    one_four = imbalance_sensitivity(4, accuracy)

    print(f'1 : 4 -> {one_four}')

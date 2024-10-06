import numpy as np
from utils import ConfusionMatrix
    
def balanced_accuracy(matrix: ConfusionMatrix) -> float:
    """Returns the balanced accuracy score for the given confusion matrix.

    Args:
        matrix (ConfusionMatrix): The matrix you want to calculate the score for. 

    Returns:
        float: A value between 0 and 1 representing the balanced accuracy of the given matrix. 
    """
    
    positive_rates: list[float] = matrix.positive_rates(return_type=list)
    
    return np.sum(positive_rates) / matrix.num_classes
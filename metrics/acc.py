import numpy as np
from utils import ConfusionMatrix

def accuracy(cm: ConfusionMatrix) -> float:
    """Returns the accuracy of a given model using the basic accuracy formula.

    Args:
        cm (CM): A confusion matrix of any size.

    Returns:
        float: The accuracy of the model, as a decimal. 
    """
    matrix = cm.array()
    
    true_values = np.sum(matrix.diagonal())
    total_values = np.sum(matrix)
    
    return true_values / total_values
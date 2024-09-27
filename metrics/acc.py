import numpy as np
from utils.confusion_matrix_generalized import CM

def accuracy(cm: CM) -> float:
    """Returns the accuracy of a given model using the basic accuracy formula.

    Args:
        cm (CMGeneralized): A confusion matrix of any size.

    Returns:
        float: The accuracy of the model, as a decimal. 
    """
    matrix = cm.array()
    
    true_values = np.sum(matrix.diagonal())
    total_values = np.sum(matrix)
    
    return true_values / total_values
from utils import ConfusionMatrix
from metrics.average import Average
import numpy as np

def recall(cm: ConfusionMatrix, average: Average = Average.MACRO) -> float:
    """Calculates the recall score of a given matrix (on either macro-averages or micro-averages)

    Args:
        cm (ConfusionMatrix): 
            The matrix to compute the score for.
        average (Average, optional): Either Average.MACRO or Average.MICRO:
            MICRO will return the sum of the total true positives divided by the total number of instances.
            MACRO will return the average the individual recall scores by class. 
            Defaults to Average.MACRO.
            
            Note: For multi-class problems, micro-precision and micro-recall will be the same.

    Raises:
        TypeError: Raised when average is not either Average.MICRO or Average.MACRO

    Returns:
        float: The recall score for the given matrix. Ranges from 0 to 1.0.
    """
    matrix = cm.array()
    result: float = 0
    
    match average:
        case Average.MACRO:
            row_sums = np.sum(matrix, axis=1)
            
            rec_macro = matrix.diagonal() / row_sums
            
            rec_macro[rec_macro == np.inf] == 0
            result = np.nanmean(rec_macro)
            
        case Average.MICRO:
            true_positives = np.sum(matrix.diagonal())
            total_instances = np.sum(matrix)
            
            result = true_positives / total_instances if total_instances > 0 else 0
        case _:
            raise TypeError("Average must be from Average Enum.")
        
    return result
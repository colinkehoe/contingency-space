from utils import ConfusionMatrix
from metrics.average import Average
import numpy as np

def precision(cm: ConfusionMatrix, average: Average = Average.MACRO) -> float:
    """Calculates the precision of a matrix (on either macro-averages or micro-averages)

    Args:
        cm (ConfusionMatrix): 
            The matrix to compute the score for.
        average (Average, optional): 
            Either Average.MACRO or Average.MICRO:
            MICRO will return the sum of the total true positives divided by the total number of instances.
            MACRO will return the average the individual precision scores by class. 
            Defaults to Average.MACRO.
            
            Note: For multi-class problems, micro-precision and micro-recall will be the same.

    Raises:
        TypeError: 
            Raised when average is not either Average.MICRO or Average.MACRO

    Returns:
        float: The precision score for the given matrix. Ranges from 0 to 1.0.
        
    """
    
    matrix = cm.array()
    result: float = 0
    
    if (cm.num_classes == 2):
        #Handle binary case. Needs to be implemented.
        
        m_dict: dict[str, list[int]] = matrix.matrix
        
        result = 1

    match average:
        case Average.MACRO:
            column_sums = np.sum(matrix, axis=0)
            
            pre_macro = matrix.diagonal() / column_sums
            
            pre_macro[pre_macro == np.inf] == 0
            result = np.nanmean(pre_macro)
            
        case Average.MICRO:
            true_positives: int = np.sum(matrix.diagonal())
            total_instances: int = np.sum(matrix)
            
            result = true_positives / total_instances if total_instances > 0 else 0
            pass;
        case _:
            raise ValueError("Average must be from Average Enum.")
        
    return result


if __name__ == "__main__":
    cm = ConfusionMatrix({'a': [1, 90, 1, 1],
                          'b': [1, 10, 0, 0],
                          'c': [0,  0, 1, 0],
                          'd': [0,  0, 0, 1]})
    print(precision(cm, Average.MICRO))
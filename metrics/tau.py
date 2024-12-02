from utils import ConfusionMatrix

import numpy as np

def tau(confusion_matrix: ConfusionMatrix, weight_vector: list[float]=None) -> float:
    """The distance between the point and the point defined by the perfect model, divided by the square
    root of the number of classes.

    Returns:
        float: The tau score.
    """
    
    model_vector: list[float] = confusion_matrix.vector(return_type=list)
    num_classes: int = confusion_matrix.num_classes
    v = 1
    
    def __dist_from_perfect(weight_vector: list[float]=None) -> float:
        """Worker function that gets the distance from the given matrix to the perfect matrix. 

        Args:
            confusion_matrix (ConfusionMatrix): _description_

        Returns:
            float: _description_
        """
        
        perfect_model_vector = [1.0] * num_classes
        
        if weight_vector is not None:
            return np.sqrt(np.sum([pow((1 - rate), 2) for rate in model_vector]))
        else:
            return np.linalg.norm(np.array(perfect_model_vector) - np.array(model_vector))
    
    if weight_vector is not None:
        if len(weight_vector) != num_classes:
            raise ValueError("Number of weights supplied to Tau must match the number of classes.")
        return v - (v/np.sqrt(num_classes)*__dist_from_perfect(weight_vector=weight_vector))
    
    tau_score = v - (v/np.sqrt(num_classes))*__dist_from_perfect()
    
    return tau_score


def sensitive_tau(confusion_matrix: ConfusionMatrix) -> float:
    """A version of tau that is sensitive to class imbalances.

    Args:
        confusion_matrix (ConfusionMatrix): _description_
    """
    
    model_vector = confusion_matrix.vector(return_type=list)
    num_classes = confusion_matrix.num_classes
    
    def __dist_from_perfect():
        return
        
        
    tau_score = 0
    return tau_score
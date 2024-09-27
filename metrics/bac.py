import numpy as np
from utils import ConfusionMatrix
    
def balanced_accuracy(matrix: ConfusionMatrix):
    
    positive_rates: list[float] = matrix.positive_rates(return_type=list)
    
    return np.sum(positive_rates) / matrix.num_classes
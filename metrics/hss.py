
from utils.confusion_matrix import ConfusionMatrix

def hss(matrix: ConfusionMatrix) -> float:
    """Calculates the Heidke Skill Score (HSS) based on the formula employed by the
    Space Weather Prediction Center for flare forecasting. See Balch 2008 for more details.

    .. Formula::
        HSS = 2[(TP * TN) - (FN * FP)] / [ [(P * (FN + TN))] + [(TP + FP) * N)] ]
    
    Args:
        matrix (ConfusionMatrix): The confusion matrix to compute the score for.

    Raises:
        ValueError: If the number of classes is not 2.
        IndexError: If the matrix does not have "true" and "false" classes.

    Returns:
        float: the Heidke Skill Score. When the denominator is zero, returns zero.
    """
    
    if matrix.num_classes != 2:
        raise ValueError('HSS is intended for binary classification problems.')
    
    t_row = matrix['t']
    f_row = matrix['f']
    
    tp, fn = t_row[0], t_row[1]
    fp, tn = f_row[0], f_row[1]
    
    numerator = 2 * ((tp * tn) - (fn * fp))
    denominator = ((tp + fn) * (fn + tn)) + ((tp + fp) * (fp + tn))
    
    hss = (numerator / float(denominator)) if denominator != 0 else 0
    
    return hss
    

if __name__ == "__main__":
    matrix = ConfusionMatrix({'t': [100, 0],
                              'f': [0, 100]})
    
    print(hss(matrix))
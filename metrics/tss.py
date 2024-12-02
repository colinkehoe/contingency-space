from utils.confusion_matrix import ConfusionMatrix

def tss(matrix: ConfusionMatrix) -> float:
    """
    Calculates the True Skill Statistic (TSS) based on the true classes and the predicted ones.
        TSS is also called  Hansen-Kuipers Skill Score or Peirce Skill Score. For more details,
        see Bobra & Couvidat (2015), or Bloomfield et al. (2012).

    .. Formula::

            TSS = [TP / (TP + FN)] - [FP / (FP + TN)]

    Args:
        matrix (ConfusionMatrix): An instance of ConfusionMatrix to calculate the
            score for.
    Raises:
        ValueError: If a multi-class matrix is passed.
        IndexError: If the matrix does not contain "true" and "false" classes.
    Returns:
        float: The TSS. If the denominator is zero, returns zero.
    """

    if matrix.num_classes != 2:
        raise ValueError('TSS is intended for binary classification problems.')

    t_row = matrix['t']
    f_row = matrix['f']
    
    tp, fn = t_row[0], t_row[1]
    fp, tn = f_row[0], f_row[1]
    
    tp_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
    fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    return tp_rate - fp_rate

if __name__ == "__main__":
    matrix = ConfusionMatrix({'t': [100, 100],
                              'f': [100, 100]})
    
    print(tss(matrix))
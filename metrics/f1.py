from utils import ConfusionMatrix
from metrics.pre import precision
from metrics.rec import recall

def f1(matrix: ConfusionMatrix):
    pre = precision(matrix)
    rec = recall(matrix)
    
    return (2 * pre * rec) / (pre + rec)
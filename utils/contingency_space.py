import copy
import pandas as pd
import numpy as np
from confusion_matrix_generalized import CMGeneralized
from confusion_matrix import CM
from enum import Enum

#visualize the contingency space:
#https://dmlab.cs.gsu.edu/metrics/contingency/

class Metrics(Enum):
    acc = 1
    ba = 2
    gm = 3
    pre = 4
    rec = 5
    f1 = 6
    gss =  7
    dss = 8
    tss = 9
    hss = 10
    j = 11
    tau = 12


def imbalance_score(metric: str | Metrics, imbalance_ratio: list[int]):
    """Return a value between 0 and 1 representing the sensitivity to class imbalance.
    Closer to 1 means more sensitivity, closer to 0 means less.
    
    Args:
        metric (str | Metrics):      The metric you wish to calculate the imbalance ratio for.
        imbalance_ratio (list[int]): The class imbalance ratios you wish to calculate for.
    
    """
        
    if type(metric) == str:
        metric = Metrics[metric]
        
    match metric:
        case Metrics.acc:
            #handle case
            return
        case Metrics.ba:
            #handle case
            return
        case Metrics.gm:
            return
        case Metrics.pre:
            return
        case Metrics.rec:
            return
        case Metrics.f1:
            return
        case Metrics.gss:
            return
        case Metrics.dss:
            return
        case Metrics.tss:
            return
        case Metrics.hss:
            return
        
    return metric

class ContingencySpace:
    """ 
    An implementation of the Contingency Space.
    """
    
    def show_history(self):
        
        #implement a method using seaborn/pyplot/etc to have a visual representation (where possible)
        #for now a text-based visual is below.
        
        for index, matrix in self.matrices.items():
            
            print(f'--[{index}]-----------------------------------------')
            print(matrix) #adapt to show 
    
    def add_history(self, values: (list | dict)):
        """Add a history entry.

        Args:
            values: the rates at which the model performed. Can either be a list with the rates of a single model or a dictionary consisting of multiple models with associated keys.
            >>> values {
                
            }
        """
        match values:
            case list():
                #add to the dict, generate a key.
                
                #if the number of classes does not match, throw an error to the user <-- to be implemented
                
                
                index: int = len(self.matrices.keys())
                self.matrices.update({str(index), values})
                return
            case dict():
                #add all rows to the dict
                
                for key, matrix in values.items():
                    self.matrices.update({key: matrix})

                return
            case _:
                print('Something has gone wrong. You must pass a list or dictionary of CMGeneralized')
                return
    
    def grab_entry(self, key: int | str) -> CMGeneralized | None:
        """_summary_

        :param key (int/str): 
            the index or key for the Confusion Matrix to retrieve.
        
            
        :return Confusion Matrix (CMGeneralized):
            The Confusion Matrix requested. If there is no matrix found with that key, returns None.
        """
        
        matrix: CMGeneralized = self.matrices[str(key)]
        
        if not matrix:
            return None
        
        return self.matrices[str(key)]
    
    
    def learning_path_length_2D(self, points: tuple[str | int, str | int]=None):
        """Calculate the learning path between the first and last points given.
        """
        
        
        
        return
    
    def learning_path_length_3D(self, points: tuple[int, int]=None):
        """Calculate the learning path between the first and last 

        Args:
            points (_type_, optional): _description_. Defaults to None.
        """
        
        return
    
    def __init__(self, metric: str, matrices: dict[str, CMGeneralized] | list[CMGeneralized] = None):
        """
        
        The constructor for the contingency space.
        
        If initialized with values, they should be passed in as a dictionary of keys and confusion matrices, or as a list
        of confusion matrices.
        
        
        Args:
            matrices: A pre-defined set of models and their values to be plotted on the contingency space. If one is not provided, an empty dictionary will be generated.
        
        """
        
        #If the user has passed in matrices, copy them to the object. Otherwise, initialize an empty dictionary.
        if not matrices:
            self.matrices = {}
        else:
            self.matrices = {}
            match matrices:
                case list():
                    #generate keys for each CM
                    for index, cm in enumerate(matrices):
                        self.matrices.update({str(index): cm})
                case dict():
                    self.matrices = copy.deepcopy(matrices)
            
if __name__ == "__main__":
    p, n = 2500, 2500
    gen = ContingencySpace([CMGeneralized({'a': [24, 23, 23],
                                           'b': [24, 33, 22],
                                           'c': [14, 23, 12]}),
                            CMGeneralized({'a': [24, 23, 23],
                                           'b': [24, 33, 22],
                                           'c': [14, 23, 12]}),
                            CMGeneralized({'a': [24, 23, 23],
                                           'b': [24, 33, 22],
                                           'c': [14, 23, 12]})])
    
    
    gen.show_history()
    
    print(ContingencySpace.imbalance_score('tau', [2, 2]))
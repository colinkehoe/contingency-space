import copy
import pandas as pd
import numpy as np
from utils.confusion_matrix_generalized import CM
from enum import Enum
from typing import Callable

from metrics import *

#visualize the contingency space:
#https://dmlab.cs.gsu.edu/metrics/contingency/

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
                print('Something has gone wrong. You must pass a list or dictionary of CM')
                return
    
    def grab_entry(self, key: int | str) -> CM | None:
        """_summary_

        :param key (int/str): 
            the index or key for the Confusion Matrix to retrieve.
        
            
        :return Confusion Matrix (CM):
            The Confusion Matrix requested. If there is no matrix found with that key, returns None.
        """
        
        matrix: CM = self.matrices[str(key)]
        
        if not matrix:
            return None
        
        return self.matrices[str(key)]
    
    def learning_path_length_2D(self, points: tuple[str, str]) -> float:
        """Calculate the learning path between the first and last points given. Currently only works for binary classification problems.
        
        Args:
            points (tuple): a tuple consisting of two keys that correspond to CMs within the contingency space.
            
        Returns:
            float: the length of the learning path from the first point to the last.
        """
        
        keys = list(self.matrices.keys())
        
        #get the keys for the user-provided matrices.
        (first, last) = points
        
        #convert the keys to an index representing their location within the space.
        first_matrix_index = keys.index(first)
        last_matrix_index = keys.index(last)
        
        #total distance traveled.
        distance_traveled = 0
        
        previous_key = keys[first_matrix_index]
        
        
        for key in keys[first_matrix_index + 1 : last_matrix_index+1]:
            
            #get the first coordinates
            (previous_x, previous_y) = self.matrices[previous_key].positive_rates()
            #get the coordinates of the previous point.
            (current_x, current_y) = self.matrices[key].positive_rates()
            
            #calculate the distance from the previous point to the current point.
            d = np.sqrt((current_x - previous_x)**2 + (current_y - previous_y)**2)
            
            distance_traveled += d
            previous_key = key
        
        return distance_traveled
    
    def learning_path_length_3D(self, points: tuple[str, str], metric: Callable[[CM], float]) -> float:
        """Calculate the learning path between the first and last points given, using an accuracy metric to determine a third dimension. Currently only works for binary classification problems. 

        Args:
            points (tuple[str, str]): The points you wish to calculate the learning path between. Defaults to None.
            metric (MetricType, optional): The metric you wish to assess the model with. Defaults to Accuracy.
        
        Returns:
            float : The distance between the first point given and the last point given across the contingency space.
        """
        
        keys = list(self.matrices.keys())
        
        #get the keys for the user-provided matrices.
        (first, last) = points
        
        #convert the keys to an index representing their location within the space.
        first_matrix_index = keys.index(first)
        last_matrix_index = keys.index(last)
        
        #total distance traveled.
        distance_traveled = 0
        
        previous_key = keys[first_matrix_index]
        
        for key in keys[first_matrix_index + 1 : last_matrix_index+1]:
            
            #get the first coordinates
            (previous_x, previous_y) = self.matrices[previous_key].positive_rates(return_type=tuple)
            previous_z = metric(self.matrices[previous_key])
            
            #get the coordinates of the next point.
            (current_x, current_y) = self.matrices[key].positive_rates()
            current_z = metric(self.matrices[key])
            
            #calculate the distance from the previous point to the current point.
            d = np.sqrt((current_x - previous_x)**2 + (current_y - previous_y)**2 + (current_z - previous_z)**2)
        
            distance_traveled += d
            previous_key = key
        
        return distance_traveled
    
    def __init__(self, matrices: dict[str, CM] | list[CM] = None):
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
    gen = ContingencySpace({'1': CM({'t': [5, 5],
                                                'f': [5, 5]}),
                            '2': CM({'t': [7, 4],
                                                'f': [3, 6]}),
                            '3': CM({'t': [7, 1],
                                                'f': [3, 9]}),
                            '4': CM({'t': [10, 0],
                                                'f': [0, 10]})
                        })
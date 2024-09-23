import copy
import pandas as pd
import numpy as np


class CMGeneralized:
    """
    Confusion matrix class for multi-class problems.
    """
    def __init__(self, table: dict[str, list[int]]={}):
        """
        The class constructor.

        An example of a confusion matrix for multiple classes ('a', 'b', and 'c')
        is given below::

                          (pred)
                         a   b   c
                       ____________
                    a  | ta  fb  fc
             (real) b  | fa  tb  fc
                    c  | fa  fb  tc

        which should be input as a dictionary as follows::

             {'a': [ta, fb, fc],
              'b': [fa, tb, fc],
              'c': [fa, fb, tc]}



        :param table: a dictionary with all class names as keys and their corresponding frequencies
        as values.
        """
        
        
        self.table = copy.deepcopy(table)
        self.num_classes = len(self.table)
        self.class_freqs = {}
        for k, v in table.items():
            self.class_freqs.update({k: int(np.sum(np.array(v)))})
        self.dim = len(self.class_freqs.keys())

    def add_class(self, cls: str, values: list[int]) -> None:
        """Adds a row to the table. Do not use this function unless you are building
        a matrix from scratch.

        Args:
            cls (str): The name of the class being added.
            values (list[int]): Values of the row.
        """
        self.table.update({cls: values})
        self.num_classes += 1
        
    def normalize(self):
        """
        normalizes all entries of the confusion matrix.

        :return: None.
        """
        
        
        cm_normalized = {}
        for k, freqs in self.table.items():
            norm_freqs = [e / self.class_freqs[k] if self.class_freqs[k] != 0 else 0 for e in freqs]
            cm_normalized.update({k: norm_freqs})
        self.__init__(cm_normalized)

    def get_total_true(self):
        """
        Returns:
            int: sum of the counts along the diagonal of the table.
        """
        a = np.array(list(self.table.values()))
        return np.sum(a.diagonal())

    def get_false_predictions(self, cls: str = None) -> dict[str, int] | int:
        """
        For each class i, the total amount of false predictions is the sum of the counts in column i, except the one on the diagonal. 
        For binary classification, this will return the number of false positives in the matrix.

        Args:
            cls (str, optional): 
                The class for which you wish to find the number of false predictions. If left blank, this function will return a list of false predictions by class. Defaults to None.

        Returns:
            list[int] | int: 
                -list[int]: List of the number of false predictions by class.
                -int:       The total number of false predictions for the specified class.
        """
        
        
        matrix = np.array(list(self.table.values()))
        keys = list(self.table.keys())
        
        diagonal_mask = np.eye(len(matrix), dtype=bool) #create a mask for the diagonal of the matrix.
        
        if cls is None:
            
            #return the sum of all values in the matrix, except for the diagonal.
            matrix_without_hits = matrix * (1 - diagonal_mask)
            
            summed_list = np.sum(matrix_without_hits, axis=0)
            
            summed_dict = {cls: value for cls, value in zip(self.table.keys(), summed_list)}
            
            return summed_dict
        else:
            try:
                column_index = keys.index(cls)
            except:
                raise ValueError(f'The class {cls} was not found in this matrix.')
            
            
            #return the sum of all values in the matrix, except for the diagonal.
            return matrix[:, column_index][~diagonal_mask[:, column_index]].sum()
            
    def get_missed_predictions(self, cls: str = None) -> list[int] | int:
        """
        For each class i, the total amount of missed predictions is the sum of the counts in row i, except the one on the diagonal. 
        For binary classification, this will return the number of false negatives in the matrix.

        Args:
            cls (str, optional): 
                The class for which you wish to find the number of missed predictions. If left blank, this function will return a list of missed predictions by class. Defaults to None.

        Returns:
            result: list[int] | int: 
                -list[int]: List of the number of missed predictions by class.
                -int:       The total number of missed predictions for the specified class.
        """
        
        matrix = np.array(list(self.table.values()))
        keys = list(self.table.keys())
        
        #create a diagonal mask for the matrix
        diagonal_mask = np.eye(len(matrix), dtype=bool)
        
        if cls is None:
            
            #return list of misses per class.
            
            matrix_without_hits = matrix * (1 - diagonal_mask)
            
            return np.sum(matrix_without_hits, axis=1)
        else:
            try:
                row_index = keys.index(cls)
            except:
                raise ValueError(f'The class {cls} was not found in this matrix.')
                
            #return sum of all values within the row, excluding the diagonal
            return matrix[row_index, :][~diagonal_mask[row_index, :]].sum()

    def get_matrix(self):
        return np.array(list(self.table.values()))

    def positive_rates(self, return_type: type = tuple) -> tuple[float, ...] | list[float]:
        """Returns a tuple representing the position of the confusion matrix within a contingency space. 

        Returns:
            c (tuple[int, ...] | list[int]): The tuple taking the form (x1, x2, ..., xk), where k is the number of classes.
        """
        
        rates = []
        cm = np.array(list(self.table.values()))
        
        total_real = np.sum(cm, axis=1) #the total # of instances of each class.
        true_pred = cm.diagonal() #the list of # of times the model predicted each class correctly.
        
        for real, pred in zip(total_real, true_pred): #create each coordinate
            rates.append(pred / real)
        
        return return_type(rates)
    
    def num_samples(self, per_class:bool = False):
        
        arr = np.array(list(self.table.values()))
        
        if per_class == True:
            return np.sum(arr, axis=1)
        return np.sum(np.array(list(self.table.values())))
    
    def array(self):
        return np.array(list(self.table.values()))
    
    def get_num_classes(self) -> int:
        """Returns the number of classes.

        Returns:
            int: The number of classes.
        """
        return self.num_classes
        

    def __repr__(self) -> str:
        #called when printing the object
        df = pd.DataFrame.from_dict(self.table, orient='index', columns=self.table.keys())
        df.index = self.table.keys()
        return str(df)
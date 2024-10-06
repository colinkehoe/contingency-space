import copy
import pandas as pd
import numpy as np
import numpy.typing as npt


class ConfusionMatrix:
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



        :param __table: a dictionary with all class names as keys and their corresponding frequencies
        as values.
        """
        
        self.__table = copy.deepcopy(table)
        self.__num_classes = len(self.__table)
        
        for row in self.__table.values():
            if len(row) != self.__num_classes:
                raise ValueError('Length of each row must be equal to the number of classes.')
        
        self.class_freqs = {}
        for k, v in self.__table.items():
            self.class_freqs.update({k: int(np.sum(np.array(v)))})
        self.dim = len(self.class_freqs.keys())

    def add_class(self, cls: str, values: list[int]) -> None:
        """Adds a row to the matrix. Do not use this function unless you are building
        a matrix from scratch.

        Args:
            cls (str): The name of the class being added.
            values (list[int]): Values of the row.
        """
        self.__table.update({cls: values})
        self.class_freqs.update({cls: int(np.sum(np.array(values)))})
        self.__num_classes += 1
        
    def normalize(self):
        """
        normalizes all entries of the confusion matrix.

        :return: None.
        """
        
        
        cm_normalized = {}
        for k, freqs in self.__table.items():
            norm_freqs = [e / self.class_freqs[k] if self.class_freqs[k] != 0 else 0 for e in freqs]
            cm_normalized.update({k: norm_freqs})
        self.__init__(cm_normalized)

    def get_total_true(self):
        """
        Returns:
            int: sum of the counts along the diagonal of the table.
        """
        a = np.array(list(self.__table.values()))
        return np.sum(a.diagonal())

    #name suggestions:
    # get_false_as_class() or get_false_as()
    def get_wrong_classifications(self, cls: str = None) -> dict[str, int] | int:
        """
        For each class i, the total amount of false classifications is the sum of the counts in column i, except the one on the diagonal. 
        For binary classification, this will return the number of false positives in the matrix.
        
        #Explain in context of classification

        Args:
            cls (str, optional): 
                The class for which you wish to find the number of false classifications. If left blank, this function will return a list of false classifications by class.

        Returns:
            list[int] | int: 
                -list[int]: List of the number of false classifications by class.
                -int:       The total number of false classifications for the specified class.
        """
        
        
        matrix = np.array(list(self.__table.values()))
        keys = list(self.__table.keys())
        
        diagonal_mask = np.eye(len(matrix), dtype=bool) #create a mask for the diagonal of the matrix.
        
        if cls is None:
            
            #return the sum of all values in the matrix, except for the diagonal.
            matrix_without_hits = matrix * (1 - diagonal_mask)
            
            summed_list = np.sum(matrix_without_hits, axis=0)
            
            summed_dict = {cls: value for cls, value in zip(self.__table.keys(), summed_list)}
            
            return summed_dict
        else:
            try:
                column_index = keys.index(cls)
            except:
                raise ValueError(f'The class {cls} was not found in this matrix.')
            
            
            #return the sum of all values in the matrix, except for the diagonal.
            return matrix[:, column_index][~diagonal_mask[:, column_index]].sum()
            
            
    #name suggestions:
    # get_misclassified_as_other()
    # get_misclassified()
    def get_missed_classifications(self, cls: str = None) -> list[int] | int:
        """
        For each class i, the total amount of missed classifications is the sum of the counts in row i, except the one on the diagonal. 
        For binary classification, this will return the number of false negatives in the matrix.

        Args:
            cls (str, optional): 
                The class for which you wish to find the number of missed classifications. If left blank, this function will return a list of missed classifications by class. Defaults to None.

        Returns:
            result: list[int] | int: 
                -list[int]: List of the number of missed classifications by class.
                -int:       The total number of missed classifications for the specified class.
        """
        
        matrix = np.array(list(self.__table.values()))
        keys = list(self.__table.keys())
        
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
        return np.array(list(self.__table.values()))

    def vector(self, return_type: npt.ArrayLike = tuple) -> tuple[float, ...] | list[float]:
        """Returns a tuple representing the position of the confusion matrix within a contingency space. 

        Returns:
            c (tuple[int, ...] | list[int]): The tuple taking the form (x1, x2, ..., xk), where k is the number of classes.
        """
        
        rates = []
        cm = np.array(list(self.__table.values()))
        
        total_real = np.sum(cm, axis=1) #the total # of instances of each class.
        true_pred = cm.diagonal() #the list of # of times the model classifications each class correctly.
        
        for real, pred in zip(total_real, true_pred): #create each coordinate
            rates.append(pred / real)
        
        return return_type(rates)
    
    def num_samples(self, per_class:bool = False):
        
        arr = np.array(list(self.__table.values()))
        
        if per_class == True:
            return np.sum(arr, axis=1)
        return np.sum(np.array(list(self.__table.values())))
    
    def array(self) -> npt.NDArray:
        return np.array(list(self.__table.values()))
    
    @property
    def matrix(self):
        return self.__table
    @matrix.getter
    def matrix(self):
        return self.__table
    @matrix.setter
    def matrix(self, new_table = dict[str, list[int]]):
        if len(new_table) != len(self.__table):
            raise ValueError("New matrix must be the same size as the old matrix.")
        if set(self.__table.keys()) != set(new_table.keys()):
            raise ValueError("New matrix must contain the same classes as the previous matrix.")
        for row in new_table.values():
            if len(row) != self.__num_classes:
                raise ValueError("Number of elements in each row must match the number of classes in the original matrix.")
            
        self.__table = new_table
    
    @property
    def num_classes(self):
        return self.__num_classes
        

    def __repr__(self) -> str:
        #called when printing the object
        df = pd.DataFrame.from_dict(self.__table, orient='index', columns=self.__table.keys())
        df.index = self.__table.keys()
        return str(df)
    
    def __eq__(self, other) -> bool:
        """Compares this CM with another CM. 
        
        Returns whether the frequencies in this matrix match the frequencies of
        another matrix.

        Args:
            other (CM): 
                The other matrix that will be compared with this one.

        Returns:
            bool: 
                Returns True if the frequencies of the given matrices match, and
                False if they do not.
        """
        if other.__class__ is self.__class__:
            for this_freq, that_freq in zip(self.class_freqs, other.class_freqs):
                if this_freq != that_freq:
                    return False
            return True
        else:
            return NotImplemented
        
    
if __name__ == "__main__":
    matrix_1 = ConfusionMatrix({'a': [500, 500],
                                'b': [500, 500]})
    matrix_2 = ConfusionMatrix({'a': [250, 250],
                                'b': [250, 250]})
    
    print(matrix_1 == matrix_2)
    
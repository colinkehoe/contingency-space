import copy
import pandas as pd
import numpy as np


class CMGeneralized:
    """
    Confusion matrix class for multi-class problems.
    """
    def __init__(self, table: dict):
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
        self.class_freqs = {}
        for k, v in table.items():
            self.class_freqs.update({k: int(np.sum(np.array(v)))})
        self.dim = len(self.class_freqs.keys())

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

    def get_total_t(self):
        """
        :return: sum of the counts on the diagonal of the table.
        """
        a = np.array(list(self.table.values()))
        return np.sum(a.diagonal())

    def get_total_fp(self):
        """
        For each class i, FP is the sum of the counts in column i, except the one on the diagonal.

        :return: column-wise sum of counts excluding the values on diagonal.
        """
        a = np.array(list(self.table.values()))
        diag = a.diagonal()
        return np.sum(np.sum(a, axis=0) - diag)

    def get_total_fn(self):
        """
        For each class i, FP is the sum of the counts in row i, except the one on the diagonal.

        :return: row-wise sum of counts excluding the values on diagonal.
        """
        a = np.array(list(self.table.values()))
        diag = a.diagonal()
        return np.sum(np.sum(a, axis=1) - diag)

    def get_matrix(self):
        return np.array(list(self.table.values()))
    
    def tuple(self):
        #implement behavior converting the matrix to the tuple (x1, x2, ... , xk) where k is # of classes
        #potentially implement returning the vector <x1, 1-x1, r2*x2, r2(1-x2),..., rk*xk, rk(1-xk)>
        return
    
    def num_classes(self) -> int:
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


import numpy as np
from utils.confusion_matrix import CM
from utils.confusion_matrix_generalized import CMGeneralized
import random
import itertools
import metrics
import math


def _verify_cm(trues, falses, expected_sum):
    res = [False for a in (trues + falses) if a != expected_sum]
    if len(res) > 0:
        raise ValueError(
            """
            At least one of the following conditions is violated:
               TP + FN = P
               TN + FP = N
            """
        )


class CMGenerator:

    #change to take any # of classes
    #change to take an imbalance ratio
    def old_constructor(self, n_p, n_n, n_cm):
        """

        :param n_p: number of positive instances
        :param n_n: number of negative instances
        :param n_cm: number of splits
        """
        self.n_cm = n_cm
        self.n_p = n_p
        self.n_n = n_n
        self.all_cms = []
        
    def __init__(self, num_classes: int, num_instances: int, instances_per_class: dict[str, int], metric=metrics.ACC):
        """Create an object capable of generating Confusion Matrices using the parameters given.

        Args:
            num_classes (int): The number of classes.
            num_instances (int): The total number of predictions made by each model.
            instances_per_class (list): The number of instances of each class.
        """
        
        self.num_classes: int = num_classes
        self.n_instances: int = num_instances
        self.n_per_class: dict[str, int] = instances_per_class
        self.all_cms: list[CMGeneralized] = []
        
        if sum(instances_per_class.values()) != num_instances:
            raise ValueError('The sum of all instances per class must be equal to the total number of instances.')
        
        #We could either use lists, or have num_instances_perclass be a dict instead, with the class names as keys.      
        return

    def generate_cms_old(self):
        all_tp = np.linspace(0, self.n_p, self.n_cm, dtype=int)
        all_fn = self.n_p - all_tp
        _verify_cm(all_tp, all_fn, self.n_p)

        all_tn = np.linspace(0, self.n_n, self.n_cm, dtype=int)
        all_fp = self.n_n - all_tn
        _verify_cm(all_tn, all_fp, self.n_n)

        for i in range(len(all_tp)):
            for j in range(len(all_tn)):
                self.all_cms.append(
                    CM({'tp': all_tp[i],
                        'fn': all_fn[i],
                        'tn': all_tn[j],
                        'fp': all_fp[j]})
                )
    
    
    def generate_cms(self, granularity: int) -> list[CMGeneralized]:
        """Generates a series of confusion matrices.

        Args:
            granularity (int): The number of values you wish to have on each axis. 
            
        Returns:
            (list[CMGeneralized]): The matrices generated. These can also by accessed by calling show_all_cms().
        """
        
        all_rates: dict[str, list] = {}
        
        for cls in self.n_per_class.keys():
            all_rates.update({cls: np.linspace(0, self.n_per_class[cls], granularity, dtype=int)})
            
        lists: list[list[int]] = all_rates.values()
        
        combinations = list(itertools.product(*lists))
        
        keys = all_rates.keys()
        
        combinations_with_keys = [dict(zip(keys, comb)) for comb in combinations]
        
        
            
        for comb in combinations_with_keys:
            
            
            matrix = CMGeneralized()
            
            
            for i, (cls, hits) in enumerate(comb.items()):
                
                row = []
                
                for j in range(self.num_classes):
                    
                    
                    if i == j:
                        
                        row.append(int(hits))
                        continue
                    
                    row.append((int(math.ceil((self.n_per_class[cls] - hits) / (self.num_classes - 1)))))
                    
                matrix.add_class(cls, row)
                
                
                
            print("\n")
            print(matrix.array())
            self.all_cms.append(matrix)
        
        return self.all_cms

    def show_all_cms(self, limit: int = None):
        limit = len(self.all_cms) if limit is None else limit
        i = 0
        while i < limit:
            print('--[{}]-----------------------------------------'.format(i))
            print(self.all_cms[i])
            i += 1



if __name__ == "__main__":
    #p, n = 2500, 2500
    #gen = CMGenerator(n_p=p, n_n=n, n_cm=6)
    gen = CMGenerator(3, 600, {'a': 200, 'b': 200, 'c': 200})
    gen.generate_cms(10)
    # n_ps = np.arange(100, 2501, 300)[::-1]
    # n_ns = np.asarray(5000 - n_ps)
    # cm_collection = []  # [[CM, ...], ...]
    # for n_p, n_n in zip(n_ps, n_ns):
    #     gen = CMGenerator(n_p=n_p, n_n=n_n, n_cm=12)
    #     gen.generate_cms()
    #     cm_collection.append(gen.all_cms)
    # print(len(cm_collection))

import numpy as np
from utils import ConfusionMatrix
import itertools


class CMGenerator:
    """Object that generates a set of confusion matrices.
    
    Constructed given a number of classes and the number of instances of each class.
    
    Object will generate a series when generate_cms() is called. 
    """
    
    def __init__(self, num_classes: int, instances_per_class: dict[str, int]):
        """Create an object capable of generating Confusion Matrices using the parameters given.

        Args:
            num_classes (int): The number of classes.
            instances_per_class (list): The number of instances of each class.
        """
        
        self.num_classes: int = num_classes
        self.n_instances: int = sum(instances_per_class.values())
        self.n_per_class: dict[str, int] = instances_per_class
        self.all_cms: list[ConfusionMatrix] = []
        
        #We could either use lists, or have num_instances_perclass be a dict instead, with the class names as keys.      
        return

    def generate_cms(self, granularity: int) -> list[ConfusionMatrix]:
        """Generates a series of confusion matrices.

        Args:
            granularity (int): The number of values you wish to have on each axis. 
            
        Returns:
            (list[ConfusionMatrix]): The matrices generated. These can also by accessed by calling show_all_cms().
        """


        #Generate every rate possible for each class.
        all_rates: dict[str, list] = {}
        for cls in self.n_per_class.keys():
            all_rates.update({cls: np.linspace(0, self.n_per_class[cls], granularity, dtype=int)})
            
        #grab the values and make a list of every possible combination of the rates.
        lists: list[list[int]] = all_rates.values()
        combinations = list(itertools.product(*lists))
        
        #zip the combinations back up with their keys. 
        keys = all_rates.keys()
        combinations_with_keys = [dict(zip(keys, comb)) for comb in combinations]
        
        
        #generate every possible matrix
        for comb in combinations_with_keys:
            matrix = ConfusionMatrix()
            
            #generate each row and insert it into the matrix
            for i, (cls, hits) in enumerate(comb.items()):
                row = []
                
                for j in range(self.num_classes):
                    
                    #if the loop is on the diagonal, append the number of successful hits.
                    if i == j:
                        row.append(int(hits))
                        continue
                    
                    #otherwise, evenly spread the instances across the other cells.
                    row.append((int((int(self.n_per_class[cls]) - int(hits)) / (self.num_classes - 1))))
                    
                matrix.add_class(cls, row)
                
                
                
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
    gen.generate_cms(3)
    
    
    # n_ps = np.arange(100, 2501, 300)[::-1]
    # n_ns = np.asarray(5000 - n_ps)
    # cm_collection = []  # [[CM, ...], ...]
    # for n_p, n_n in zip(n_ps, n_ns):
    #     gen = CMGenerator(n_p=n_p, n_n=n_n, n_cm=12)
    #     gen.generate_cms()
    #     cm_collection.append(gen.all_cms)
    # print(len(cm_collection))

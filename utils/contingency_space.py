import copy
import pandas as pd
import numpy as np
from utils import ConfusionMatrix
from typing import Callable
from metrics import *

class ContingencySpace:
    """ 
    An implementation of the Contingency Space.
    """
    
    def show_history(self):
        """
        Shows the history of the space in confusion matrices. 
        
        Args:
            None.
            
        Prints:
            The matrices.
        """
        
        for index, matrix in self.matrices.items():
            
            print(f'--[{index}]-----------------------------------------')
            print(matrix) #adapt to show 
    
    def add_history(self, values: (list | dict)):
        """Add a history entry.

        Args:
            values: 
                the rates at which the model performed. Can either be a list with the rates of a single model 
                or a dictionary consisting of multiple models with associated keys.
            
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
                print('Something has gone wrong. You must pass a list or dictionary of ConfusionMatrix')
                return
    
    def grab_entry(self, key: int | str) -> ConfusionMatrix | None:
        """
        Grabs a matrix from the space history using a string or integer key.

        Args:
            key (int/str): 
                the index or key for the Confusion Matrix to retrieve.
        
            
        Returns:
            ConfusionMatrix:
                The Confusion Matrix requested. If there is no matrix found with that key, returns None.
        """
        
        matrix: ConfusionMatrix = self.matrices[str(key)]
        
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
    
    def learning_path_length_3D(self, points: tuple[str, str], metric: Callable[[ConfusionMatrix], float]) -> float:
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
    
    def learning_path(self, points: tuple[str, str], metric: Callable[[ConfusionMatrix], float] = None) -> float:
        match metric:
            case None:
                result = self.learning_path_length_2D(points)
                return result
            
        self.learning_path_length_3D()
        
    def visualize(self, metric: Callable[[ConfusionMatrix], float] | list[Callable[[ConfusionMatrix], float]], step_size: int = 30, lines: bool = True, projection: str = '2d', fig_size: tuple[int, int]=(5, 5), ax=None, point_size: float = 1.0, title: str=None) -> None:
        
        
        import matplotlib.pyplot as plt
        from utils import CMGenerator
        
        point_size_list = [point_size for _ in range(len(self.matrices.keys()))]
        
        # Generate the space we will draw the points on.
        # ----------------------------------------------
        example_matrix: ConfusionMatrix = list(self.matrices.values())[0]
        
        # 0: positives, 1: negatives
        matrix_instances = {cls: sum(row) for (cls, row) in example_matrix.matrix.items()}
        matrix_instances_per_class_list = [x for x in matrix_instances.values()]
        
        generator = CMGenerator(self.num_classes, instances_per_class = matrix_instances)
        generator.generate_cms(granularity=step_size)
        
        base_matrices = generator.all_cms
        
        base_points = [matrix.vector(metric=metric) for matrix in base_matrices]
        base_x = [x*matrix_instances_per_class_list[1] for x, y, z in base_points]
        base_y = [y*matrix_instances_per_class_list[0] for x, y, z in base_points]
        base_z = [z for x, y, z in base_points]
        
        
        base_x = np.array(base_x[:step_size]) # every nth element
        base_y = np.array(base_y[::step_size])  # first n elements
        base_z = np.array(base_z).reshape((step_size, step_size))
        
        base_x_mesh, base_y_mesh = np.meshgrid(base_x, base_y)
        
        space_matrix_points = [matrix.vector(metric=metric) for matrix in self.matrices.values()]
        
        
        # deconstruct tuple
        model_points_x, model_points_y, model_points_z = map(list, zip(*space_matrix_points))
        
        # rescale values
        model_points_x = [x * matrix_instances_per_class_list[1] for x in model_points_x]
        model_points_y = [y * matrix_instances_per_class_list[0] for y in model_points_y]
        model_points_z = [round(z, 2) for z in model_points_z]
        
        match projection:
            case '2d':
                if ax is not None:
                    cs = ax.contourf(base_x_mesh, base_y_mesh, base_z, 30, vmin=-1, vmax=1, alpha=1, cmap='twilight_shifted')
                    cs2 = ax.contour(base_x_mesh, base_y_mesh, base_z, 30, vmin=-1, vmax=1, cmap='twilight', edgecolor='black', linewidth=9, linewidths=1.7, linestyles='solid', )
                    if title is not None: 
                        ax.set_title(title)
                else:
                    plt.figure(figsize=fig_size)
                    cs = plt.contourf(base_x_mesh, base_y_mesh, base_z, 30, vmin=-1, vmax=1, alpha=1, cmap='twilight_shifted', linestyles=None)
                    cs2 = plt.contour(base_x_mesh, base_y_mesh, base_z, 30, vmin=-1, vmax=1, cmap='twilight', linewidths=0.6, linestyles=None)
                    for c in cs.collections:
                        c.set_edgecolor("face")
                        c.set_linewidth(0.000000000001)
                    for c in cs2.collections:
                        c.set_edgecolor("face")
                        c.set_linewidth(0.000000000001)
            case '3d':
                if ax is not None:
                    base_x_mesh, base_y_mesh = np.meshgrid(base_x, base_y)
                    
                    surf = ax.plot_surface(base_x_mesh, base_y_mesh, base_z, cmap='twilight_shifted', vmin=-1, vmax=1, alpha=1)
                else:
                    fig = plt.figure(figsize=fig_size)
                    ax = fig.add_subplot(111, projection='3d')

                    base_x_mesh, base_y_mesh = np.meshgrid(base_x, base_y)

                    surf = ax.plot_surface(base_x_mesh, base_y_mesh, base_z, cmap='twilight_shifted', vmin=-1, vmax=1, alpha=1)
        
        match projection:
            case '2d':
                if ax is not None:
                    ax.plot(model_points_x, model_points_y, color='white', linewidth=1.5, linestyle='-', alpha=1.0, marker='o', markersize=point_size)
                    ax.plot(model_points_x[0], model_points_y[0], color='blue', alpha=1.0, marker='o', markersize=2*point_size)
                    ax.plot(model_points_x[-1], model_points_y[-1], color='red', alpha=1.0, marker='o', markersize=2*point_size)
                else:
                    if lines:
                        plt.plot(model_points_x, model_points_y, color='white', linewidth=1.5, linestyle='-', alpha=1.0, marker='o', markersize=point_size)
                    else:
                        plt.scatter(model_points_x, model_points_y, color='white', alpha=0.6, s=point_size)
            case '3d':
                
                ax.scatter(model_points_x, model_points_y, model_points_z, color='black', s=100)
                ax.plot(model_points_x, model_points_y, model_points_z, color='black', linewidth=1.5, linestyle='-', alpha=1)
        if ax is None:
            plt.show()
        
        
        
    def __init__(self, matrices: dict[str, ConfusionMatrix] | list[ConfusionMatrix] = None):
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
            self.num_classes = 0
        else:
            self.matrices = {}
            self.num_classes = 0
            match matrices:
                case list():
                    #generate keys for each ConfusionMatrix
                    for index, cm in enumerate(matrices):
                        self.matrices.update({str(index): cm})
                        if self.num_classes != cm.num_classes:
                            if self.num_classes == 0:
                                self.num_classes = cm.num_classes
                            else:
                                raise ValueError('Number of classes must remain the same over every matrix.')
                case dict():
                    self.matrices = copy.deepcopy(matrices)
                    # add num classes
                    
                    
            
if __name__ == "__main__":
    p, n = 2500, 2500
    space = ContingencySpace({'1': ConfusionMatrix({'t': [5, 5],
                                                'f': [5, 5]}),
                            '2': ConfusionMatrix({'t': [7, 3],
                                                'f': [4, 6]}),
                            '3': ConfusionMatrix({'t': [7, 3],
                                                'f': [1, 9]}),
                            '4': ConfusionMatrix({'t': [10, 0],
                                                'f': [0, 10]})
                        })
    
    print(space.num_classes)
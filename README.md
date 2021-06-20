# Contingency Space #


### Abstract.

In Machine Learning, a supervised model's performance is measured
using the verification metrics. In this study, we first present
our motivation by revisiting the major limitations of these metrics,
namely one-dimensionality, lack of context, lack of intuitiveness,
uncomparability, binary restriction, and uncustomizability of
metrics. In response, we propose Contingency Space, a bounded
semimetric space that provides a generic representation for any
performance verification metric. Then we showcase how this space
addresses the limitations. In this space, each metric forms a
surface using which we visually compare different verification
metrics. Taking advantage of the fact that a metric's surface
warps proportional to the degree of which it is sensitive to
the class-imbalance ratio of data, we introduce Imbalance
Sensitivity that quantifies the skew-sensitivity. Since an
arbitrary model is represented in this space by a single point, we
introduce Learning Path for qualitative and quantitative analyses
of the training process. Using the semimetric that contingency
space is endowed with, we introduce Tau as a new cost sensitive
and Imbalance Agnostic metric. Lastly, we show that contingency
space addresses multi-class problems as well. Throughout this work
we define each concept through stipulated definitions and present
every application with practical examples and visualizations.

### How To Use.

* Use the notebooks under [./notebooks/](./notebooks/) to explore some of
the main functionalities.
  
* All metrics are available under [./metrics/](./metrics/).

* One of the notebooks ([./notebooks/cnn_learning_path_on_MNIST.ipynb](./notebooks/cnn_learning_path_on_MNIST.ipynb))
requires the data of the learning paths. The data is pickled and
  available under [./pickled_data/](./pickled_data/).
  

### Metadata.
* Python 3.8
* Tested on Ubuntu 20.04.2 LTS
* Utilized libraries are listed in [./requirements.txt](./requirements.txt).
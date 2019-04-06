# Auto-PyTorch

Copyright (C) 2018  [AutoML Group](http://www.automl.org/)

This a very early pre-alpha version of our upcoming Auto-PyTorch.
So far, Auto-PyTorch only supports featurized data.

## Installation

Clone repository

```sh
$ cd install/path
$ git clone https://github.com/automl/Auto-PyTorch.git
$ cd Auto-PyTorch
```
If you want to contribute to this repository switch to our current develop branch

```sh
$ git checkout develop
```

Install pytorch: 
https://pytorch.org/
 
Install autonet

```sh
$ python setup.py install
```


## Examples

In a nutshell:

```py
from autoPyTorch import AutoNetClassification

# data and metric imports
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
X, y = sklearn.datasets.load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(X, y, random_state=1)

# running Auto-PyTorch
autoPyTorch = AutoNetClassification("tiny_cs",  # config preset
                                    log_level='info',
                                    max_runtime=300,
                                    min_budget=30,
                                    max_budget=90)

autoPyTorch.fit(X_train, y_train, validation_split=0.3)
y_pred = autoPyTorch.predict(X_test)

print("Accuracy score", sklearn.metrics.accuracy_score(y_test, y_pred))
```

How to configure Auto-PyTorch for your needs:

```py

# print all possible configuration options
autoPyTorch = AutoNetClassification().print_help()

# you pass use the constructor to configure Auto-PyTorch
autoPyTorch = AutoNetClassification(log_level='info', max_runtime=300, min_budget=30, max_budget=90)

# you can overwrite this configuration in each fit call
autoPyTorch.fit(X_train, y_train, log_level='debug', max_runtime=900, min_budget=50, max_budget=150)

# you can use presets to configure the config space
# available presets: full_cs, medium_cs (default), tiny_cs
# these are defined in configs/autonet/presets
# tiny_cs is recommended if you want fast results with few resources
# full_cs is recommended if you have many resources and a high search budget
autoPyTorch = AutoNetClassification("full_cs")

# enabling / disabling of components is done using the Auto-PyTorch config:
autoPyTorch = AutoNetClassification(networks=["shapedresnet"])

# print the search space.
# each hyperparameter belongs to a node in Auto-PyTorch's ML Pipeline.
# The names of the hyperparameters are prefixed with the name of the node: NodeName:hyperparameter_name
# If a hyperparameter belongs to a component: NodeName:component_name:hyperparameter_name
autoPyTorch.get_hyperparameter_search_space()

# You can configure the search space of every hyperparameter of every component:
from autoPyTorch import HyperparameterSearchSpaceUpdates
search_space_updates = HyperparameterSearchSpaceUpdates()

search_space_updates.append(node_name="NetworkSelector",  # refers to the node in the ML-Pipeline of Auto-PyTorch, see above
                            hyperparameter="shapedresnet:activation",
                            value_range=["relu", "sigmoid"])
search_space_updates.append(node_name="NetworkSelector",
                            hyperparameter="shapedresnet:blocks_per_group,",
                            value_range=[2,5],
                            log=False)
autoPyTorch = AutoNetClassification(hyperparameter_search_space_updates=search_space_updates)

# enable ensemble building:
autoPyTorchEnsemble = AutoNetEnsemble(AutoNetClassification, "tiny_cs", max_runtime=300, min_budget=30, max_budget=90)


```

Disable pynisher if you experience issues when using cuda:

```py
autoPyTorch = AutoNetClassification("tiny_cs", log_level='info', max_runtime=300, min_budget=30, max_budget=90, cuda=True, use_pynisher=False)

```

More examples with datasets:

```sh
$ cd examples/
```

## License

This program is free software: you can redistribute it and/or modify
it under the terms of the 3-clause BSD license (please see the LICENSE file).

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

You should have received a copy of the 3-clause BSD license
along with this program (see LICENSE file).
If not, see <https://opensource.org/licenses/BSD-3-Clause>.

## Reference

```
@incollection{mendoza-automlbook18a,
  author    = {Hector Mendoza and Aaron Klein and Matthias Feurer and Jost Tobias Springenberg and Matthias Urban and Michael Burkart and Max Dippel and Marius Lindauer and Frank Hutter},
  title     = {Towards Automatically-Tuned Deep Neural Networks},
  year      = {2018},
  month     = dec,
  editor    = {Hutter, Frank and Kotthoff, Lars and Vanschoren, Joaquin},
  booktitle = {AutoML: Methods, Sytems, Challenges},
  publisher = {Springer},
  chapter   = {7},
  pages     = {141--156},
  note      = {To appear.},
}
```

## Contact

Auto-PyTorch is developed by the [AutoML Group of the University of Freiburg](http://www.automl.org/).

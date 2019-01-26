__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"

from copy import deepcopy
import ConfigSpace
import inspect
from autoPyTorch.utils.config.config_option import ConfigOption
from autoPyTorch.utils.configspace_wrapper import ConfigWrapper
from autoPyTorch.pipeline.base.node import Node


class PipelineNode(Node):
    def __init__(self):
        """A pipeline node is a step in a pipeline.
        It can implement a fit function:
            Returns a dictionary.
            Input parameter (kwargs) are given by previous fit function computations in the pipeline.
        It can implement a predict function:
            Returns a dictionary.
            Input parameter (kwargs) are given by previous predict function computations in the pipeline or defined in fit function output of this node.

        Each node can provide a list of config options that the user can specify/customize.
        Each node can provide a config space for optimization.

        """

        super(PipelineNode, self).__init__()
        self.user_hyperparameter_range_updates = dict()
        self.pipeline = None

    @classmethod
    def get_name(cls):
        return cls.__name__
    
    def clone(self, skip=("pipeline", "fit_output", "predict_output", "child_node")):
        node_type = type(self)
        new_node = node_type.__new__(node_type)
        for key, value in self.__dict__.items():
            if key not in skip:
                setattr(new_node, key, deepcopy(value))
            else:
                setattr(new_node, key, None)
        return new_node

    # VIRTUAL
    def fit(self, **kwargs):
        """Fit pipeline node.
        Each node computes its fit function in linear order.
        All args have to be specified in a parent node fit output.
        
        Returns:
            [dict] -- output values that will be passed to child nodes, if required
        """

        return dict()

    # VIRTUAL
    def predict(self, **kwargs):
        """Predict pipeline node.
        Each node computes its predict function in linear order.
        All args have to be specified in a parent node predict output or in the fit output of this node
        
        Returns:
            [dict] -- output values that will be passed to child nodes, if required
        """

        return dict()

    # VIRTUAL
    def set_pipeline(self, pipeline):
        self.pipeline = pipeline

    # VIRTUAL
    def get_pipeline_config_options(self):
        """Get available ConfigOption parameter.
        
        Returns:
            List[ConfigOption] -- list of available config options
        """

        return []

    # VIRTUAL
    def get_pipeline_config_conditions(self):
        """Get the conditions on the pipeline config (e.g. max_budget > min_budget)
        
        Returns:
            List[ConfigCondition] -- list of functions, that take a pipeline config and raise an Error, if faulty configuration is detected.
        """

        return []


    # VIRTUAL
    def get_hyperparameter_search_space(self, **pipeline_config):
        """Get hyperparameter that should be optimized.
        
        Returns:
            ConfigSpace -- config space
        """

        # if you override this function make sure to call _apply_user_updates
        return self._apply_user_updates(ConfigSpace.ConfigurationSpace())
    
    # VIRTUAL
    def insert_inter_node_hyperparameter_dependencies(self, config_space, **pipeline_config):
        """Insert Conditions and Forbiddens of hyperparameters of different nodes

        Returns:
            ConfigSpace -- config space
        """
        return config_space

    def _update_hyperparameter_range(self, name, new_value_range, log=False,
            check_validity=True, pipeline_config=None):
        """Allows the user to update a hyperparameter
        
        Arguments:
            name {string} -- name of hyperparameter
            new_value_range {List[?] -- value range can be either lower, upper or a list of possible conditionals
            log {bool} -- is hyperparameter logscale
        """

        if (len(new_value_range) == 0):
            raise ValueError("The new value range needs at least one value")

        if check_validity:
            configspace = self.get_hyperparameter_search_space(**pipeline_config)
            # this will throw an error if such a hyperparameter does not exist
            hyper = configspace.get_hyperparameter(name)
            if (isinstance(hyper, ConfigSpace.hyperparameters.NumericalHyperparameter)):
                if (len(new_value_range) != 2):
                    raise ValueError("If you modify a NumericalHyperparameter you have to specify a lower and upper bound")
                if (new_value_range[0] > new_value_range[1]):
                    raise ValueError("The lower bound has to be smaller than the upper bound")
            elif (isinstance(hyper, ConfigSpace.hyperparameters.CategoricalHyperparameter)):
                pass
            else:
                raise ValueError("Modifying " + str(type(hyper)) + " is not supported")

        self.user_hyperparameter_range_updates[name] = tuple([new_value_range, log])
    
    def _get_user_hyperparameter_range_updates(self, prefix=None):
        if prefix is None:
            return self.user_hyperparameter_range_updates
        result = dict()
        for key in self.user_hyperparameter_range_updates.keys():
            if key.startswith(prefix + ConfigWrapper.delimiter):
                result[key[len(prefix + ConfigWrapper.delimiter):]] = self.user_hyperparameter_range_updates[key][0]
        return result

    def _apply_user_updates(self, config_space):
        for name, update_params in self.user_hyperparameter_range_updates.items():
            if (config_space._hyperparameters.get(name) == None):
                # this can only happen if the config space got modified in the pipeline update/creation process (user)
                print("The modified hyperparameter " + name + " does not exist in the config space.")
                continue

            hyper = config_space.get_hyperparameter(name)
            if (isinstance(hyper, ConfigSpace.hyperparameters.NumericalHyperparameter)):
                hyper.__init__(name=hyper.name, lower=update_params[0][0], upper=update_params[0][1], log=update_params[1])
            elif (isinstance(hyper, ConfigSpace.hyperparameters.CategoricalHyperparameter)):
                hyper.__init__(name=hyper.name, choices=tuple(update_params[0]), log=update_params[1])

        return config_space


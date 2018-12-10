import torch
import numpy as np
import scipy.sparse

import ConfigSpace
import ConfigSpace.hyperparameters as CSH
        
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

from autoPyTorch.pipeline.base.pipeline_node import PipelineNode

from autoPyTorch.utils.config.config_option import ConfigOption
from autoPyTorch.utils.configspace_wrapper import ConfigWrapper

class Imputation(PipelineNode):

    strategies = ["mean", "median", "most_frequent"]

    def fit(self, hyperparameter_config, X, Y, train_indices, dataset_info):
        hyperparameter_config = ConfigWrapper(self.get_name(), hyperparameter_config)

        strategy = hyperparameter_config['strategy']
        fill_value = int(np.nanmax(X[train_indices])) + 1 if dataset_info.is_sparse else 0
        numerical_imputer = SimpleImputer(strategy=strategy, copy=False)
        categorical_imputer = SimpleImputer(strategy='constant', copy=False, fill_value=fill_value)
        transformer = ColumnTransformer(
            transformers=[('numerical_imputer', numerical_imputer, [i for i, c in enumerate(dataset_info.categorical_features) if not c]),
                          ('categorical_imputer', categorical_imputer,  [i for i, c in enumerate(dataset_info.categorical_features) if c])])
        transformer.fit(X[train_indices])
        X = transformer.transform(X)
        
        dataset_info.categorical_features = sorted(dataset_info.categorical_features)
        return { 'X': X, 'imputation_preprocessor': transformer, 'dataset_info': dataset_info }


    def predict(self, X, imputation_preprocessor):
        return { 'X': imputation_preprocessor.transform(X) }

    @staticmethod
    def get_hyperparameter_search_space(**pipeline_config):

        possible_strategies = set(Imputation.strategies).intersection(pipeline_config['imputation_strategies'])

        cs = ConfigSpace.ConfigurationSpace()
        cs.add_hyperparameter(CSH.CategoricalHyperparameter("strategy", possible_strategies))
        return cs

    def get_pipeline_config_options(self):
        options = [
            ConfigOption(name='imputation_strategies', default=Imputation.strategies, type=str, list=True, choices=Imputation.strategies)
        ]
        return options
        
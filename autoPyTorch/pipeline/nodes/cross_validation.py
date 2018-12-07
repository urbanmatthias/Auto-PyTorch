__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"

import torch
import logging
import scipy.sparse
import numpy as np
import pandas as pd
import signal
import time
import math
import inspect

from sklearn.model_selection import BaseCrossValidator
from autoPyTorch.pipeline.base.sub_pipeline_node import SubPipelineNode
from autoPyTorch.pipeline.base.pipeline import Pipeline

from autoPyTorch.utils.config.config_option import ConfigOption, to_bool, to_dict
from autoPyTorch.training.budget_types import BudgetTypeTime

import time

class CrossValidation(SubPipelineNode):
    def __init__(self, train_pipeline_nodes):
        """CrossValidation pipeline node.
        It will run the train_pipeline by providing different train and validation datasets given the cv_split value defined in the config.
        Cross validation can be disabled by setting cv_splits to <= 1 in the config
        This enables the validation_split config parameter which, if no validation data is provided, will split the train dataset according its value (percent of train dataset)

        Train:
        The train_pipeline will receive the following inputs:
        {hyperparameter_config, pipeline_config, X_train, Y_train, X_valid, Y_valid, budget, training_techniques, fit_start_time, categorical_features}

        Prediction:
        The train_pipeline will receive the following inputs:
        {pipeline_config, X}
        
        Arguments:
            train_pipeline {Pipeline} -- training pipeline that will be computed cv_split times
            train_result_node {PipelineNode} -- pipeline node that provides the results of the train_pipeline
        """

        super(CrossValidation, self).__init__(train_pipeline_nodes)

        self.cross_validators = {'none': None}
        self.cross_validators_adjust_y = dict()
        self.logger = logging.getLogger('autonet')


    def fit(self, hyperparameter_config, pipeline_config, X_train, Y_train, X_valid, Y_valid, budget, budget_type, optimize_start_time, refit):
        loss = 0
        infos = []
        num_cv_splits, cv_splits, loss_penalty, budget = self.initialize_cross_validation(
            pipeline_config=pipeline_config, budget=budget, X_train=X_train, Y_train=Y_train, X_valid=X_valid, Y_valid=Y_valid, refit=refit)

        # set up categorical features
        if 'categorical_features' in pipeline_config and pipeline_config['categorical_features']:
            categorical_features = pipeline_config['categorical_features']
        else:
            categorical_features = [False] * X_train.shape[1]
        
        # adjust budget in case of budget type time
        if budget_type == BudgetTypeTime:
            cv_start_time = time.time()
            budget = budget - (cv_start_time - optimize_start_time)

        # start cross validation
        self.logger.debug("Took " + str(time.time() - optimize_start_time) + " s to initialize optimization.")
        for i, split_indices in enumerate(cv_splits):
            if num_cv_splits > 1:
                self.logger.debug("[AutoNet] CV split " + str(i) + " of " + str(num_cv_splits))

            # adjust budget in case of budget type time
            if budget_type == BudgetTypeTime:
                remaining_budget = budget - (time.time() - cv_start_time)
                should_be_remaining_budget = (budget - i * budget / num_cv_splits)
                budget_type.compensate = max(10, should_be_remaining_budget - remaining_budget)
                cur_budget = remaining_budget / (num_cv_splits - i)
                self.logger.info("Reduced initial budget " + str(budget / num_cv_splits) + " to cv budget " + 
                                 str(cur_budget) + " compensate for " + str(should_be_remaining_budget - remaining_budget))
            else:
                cur_budget = budget / num_cv_splits

            # fit training pipeline
            result = self.sub_pipeline.fit_pipeline(
                hyperparameter_config=hyperparameter_config, pipeline_config=pipeline_config, 
                X_train=X_train, Y_train=Y_train, X_valid=X_valid, Y_valid=Y_valid, 
                budget=cur_budget, training_techniques=[budget_type()],
                fit_start_time=time.time(),
                categorical_features=categorical_features,
                split_indices=split_indices, cv_index=i)

            if result is not None:
                loss += result['loss']
                infos.append(result['info'])

        if (len(infos) == 0):
            raise Exception("Could not finish a single cv split due to memory or time limitation")

        df = pd.DataFrame(infos)
        info = dict(df.mean())
        info['num_cv_splits'] = num_cv_splits

        loss = loss / num_cv_splits + loss_penalty

        return {'loss': loss, 'info': info}

    def predict(self, pipeline_config, X):
       
        result = self.sub_pipeline.predict_pipeline(pipeline_config=pipeline_config, X=X)

        return {'Y': result['Y']}

    def get_pipeline_config_options(self):
        options = [
            ConfigOption("validation_split", default=0.0, type=float, choices=[0, 1],
                info='In range [0, 1). Part of train dataset used for validation. Ignored in fit if cross validator or valid data given.'),
            ConfigOption("cross_validator", default="none", type=str, choices=self.cross_validators.keys(),
                info='Class inheriting from sklearn.model_selection.BaseCrossValidator. Ignored if validation data is given.'),
            ConfigOption("cross_validator_args", default=dict(), type=to_dict,
                info="Args of cross validator. \n\t\tNote that random_state and shuffle are set by " +
                     "pipeline config options random_seed and shuffle, if not specified here."),
            ConfigOption("min_budget_for_cv", default=0, type=float,
                info='Specify minimum budget for cv. If budget is smaller use specified validation split.')
        ]
        return options

    def clean_fit_data(self):
        super(CrossValidation, self).clean_fit_data()
        self.sub_pipeline.root.clean_fit_data()
    
    def initialize_cross_validation(self, pipeline_config, budget, X_train, Y_train, X_valid, Y_valid, refit):
        budget_too_low_for_cv = budget < pipeline_config['min_budget_for_cv']
        val_split = max(0, min(1, pipeline_config['validation_split']))

        # validation set given. cv ignored.
        if X_valid is not None and Y_valid is not None:
            self.logger.info("[AutoNet] Validation set given. Continue with validation set (no cross validation).")
            return 1, [None], 0, budget
        
        # no cv, split train data
        if pipeline_config['cross_validator'] == "none" or budget_too_low_for_cv:
            self.logger.info("[AutoNet] No validation set given and either no cross validator given or budget to low for CV." + 
                             " Continue by splitting " + str(val_split) + " of training data.")
            return 1, [get_valid_split_indices(pipeline_config, X_train.shape[0], val_split)], (1000 if budget_too_low_for_cv else 0), budget

        # cross validation
        cross_validator_class = self.cross_validators[pipeline_config['cross_validator']]
        adjust_y = self.cross_validators_adjust_y[pipeline_config['cross_validator']]
        available_cross_validator_args = inspect.getfullargspec(cross_validator_class.__init__)[0]
        cross_validator_args = pipeline_config['cross_validator_args']

        if "shuffle" not in cross_validator_args and "shuffle" in available_cross_validator_args:
            cross_validator_args["shuffle"] = pipeline_config["shuffle"]
        if "random_state" not in cross_validator_args and "random_state" in available_cross_validator_args:
            cross_validator_args["random_state"] = pipeline_config["random_seed"]

        cross_validator = cross_validator_class(**cross_validator_args)
        num_cv_splits = cross_validator.get_n_splits(X_train, adjust_y(Y_train))
        cv_splits = cross_validator.split(X_train, adjust_y(Y_train))
        if not refit:
            self.logger.info("[Autonet] Continue with cross validation using " + str(pipeline_config['cross_validator']))
            return num_cv_splits, cv_splits, 0, budget
        
        self.logger.info("[Autonet] No cross validation when refitting! Continue by splitting " + str(val_split) + " of training data.")
        return 1, [get_valid_split_indices(pipeline_config, X_train.shape[0], val_split)], 0, budget / num_cv_splits

    def add_cross_validator(self, name, cross_validator, adjust_y=None):
        self.cross_validators[name] = cross_validator
        self.cross_validators_adjust_y[name] = adjust_y if adjust_y is not None else lambda x: x
    
    def remove_cross_validator(self, name):
        del self.cross_validators[name]
        del self.cross_validators_adjust_y[name]


def split_data(split_indices, X_train=None, Y_train=None, X_valid=None, Y_valid=None):
    if split_indices is None:
        return X_train, Y_train, X_valid, Y_valid
    train_indices = split_indices[0]
    valid_indices = split_indices[1]
    return (X_train[train_indices] if X_train is not None else None,
            Y_train[train_indices] if Y_train is not None else None,
            X_train[valid_indices] if X_train is not None and len(valid_indices) > 0 else None,
            Y_train[valid_indices] if Y_train is not None and len(valid_indices) > 0 else None)

def get_valid_split_indices(pipeline_config, num_datapoints, val_split):
    all_indices = np.arange(num_datapoints)
    rng = np.random.RandomState(pipeline_config["random_seed"])
    val_indices = np.random.choice(all_indices, int(all_indices.shape[0] * val_split), replace=False)
    train_indices = np.array([a for a in all_indices if a not in val_indices])
    if pipeline_config["shuffle"]:
        rng.shuffle(train_indices)
    return (train_indices, val_indices)


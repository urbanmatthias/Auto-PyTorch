
import logging
import os
from copy import copy
import traceback

import hpbandster.core.nameserver as hpns
from autoPyTorch.pipeline.base.pipeline_node import PipelineNode
from autoPyTorch.pipeline.nodes.optimization_algorithm import OptimizationAlgorithm
from autoPyTorch.utils.benchmarking.benchmark_pipeline.prepare_result_folder import \
    get_run_result_dir
from autoPyTorch.utils.config.config_option import ConfigOption, to_bool
from hpbandster.core.dispatcher import Job
from hpbandster.core.result import logged_results_to_HBS_result
from hpbandster.optimizers.config_generators.bohb import \
    BOHB as ConfigGeneratorBohb
from ConfigSpace.read_and_write.pcs_new import read as read_pcs
from ConfigSpace.read_and_write.json import read as read_json


class Collect(PipelineNode):

    def fit(self, pipeline_config, instance, initial_design_learner, warmstarted_model_builder, run_result_dir, data_manager, autonet):
        logger = logging.getLogger("metalearning")
        print("Collecting " + run_result_dir)
        # if os.path.exists(os.path.join(run_result_dir, "configspace.pcs")):
        #     with open(os.path.join(run_result_dir, "configspace.pcs"), "r") as f:
        #         config_space = read_pcs(f.readlines())
        if os.path.exists(os.path.join(run_result_dir, "configspace.json")):
            with open(os.path.join(run_result_dir, "configspace.json"), "r") as f:
                config_space = read_json("\n".join(f.readlines()))
        else:
            config_space = autonet.get_hyperparameter_search_space()

        exact_cost_model = AutoNetExactCostModel(autonet, data_manager, {
            "file_name": instance,
            "is_classification": (pipeline_config["problem_type"] in ['feature_classification', 'feature_multilabel']),
            "test_split": pipeline_config["test_split"]
        })

        initial_design_learner.add_result(run_result_dir, config_space, exact_cost_model)
        warmstarted_model_builder.add_result(run_result_dir, config_space)
        return dict()

class AutoNetExactCostModel():
    def __init__(self, autonet, dm, dm_kwargs):
        self.autonet = autonet
        self.dm = dm
        self.dm_kwargs = dm_kwargs
    
    def __enter__(self):
        self.dm.read_data(**self.dm_kwargs)
        return self
    
    def evaluate(self, config, budget):
        return self.autonet.refit(
            X_train=self.dm.X_train, Y_train=self.dm.Y_train, X_valid=self.dm.X_valid, Y_valid=self.dm.Y_valid,
            hyperparameter_config=config.get_dictionary(), budget=budget)
    
    def __exit__(self, error_type, error_value, error_traceback):
        del self.dm
        return error_type is not None
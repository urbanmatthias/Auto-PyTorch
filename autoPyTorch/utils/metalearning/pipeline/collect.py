
import logging
import os
from copy import copy
import traceback
import time

import hpbandster.core.nameserver as hpns
from autoPyTorch.pipeline.base.pipeline_node import PipelineNode
from autoPyTorch.pipeline.nodes.optimization_algorithm import OptimizationAlgorithm
from autoPyTorch.utils.benchmarking.benchmark_pipeline.prepare_result_folder import \
    get_run_result_dir
from autoPyTorch.utils.config.config_option import ConfigOption, to_bool
from autoPyTorch.training.budget_types import BudgetTypeTime
from hpbandster.core.dispatcher import Job
from hpbandster.core.result import logged_results_to_HBS_result
from hpbandster.optimizers.config_generators.bohb import \
    BOHB as ConfigGeneratorBohb
from ConfigSpace.read_and_write.pcs_new import read as read_pcs
from ConfigSpace.read_and_write.json import read as read_json


class Collect(PipelineNode):

    def fit(self, pipeline_config, instance, initial_design_learner, warmstarted_model_builder, run_result_dir, data_manager, autonet):
        logger = logging.getLogger("metalearning")

        if pipeline_config["only_finished_runs"] and not os.path.exists(os.path.join(run_result_dir, "summary.json")):
            logger.info('Skipping ' + run_result_dir + ' because the run is not finished yet')
            return dict()

        print("Collecting " + run_result_dir)

        if os.path.exists(os.path.join(run_result_dir, "configspace.json")):
            with open(os.path.join(run_result_dir, "configspace.json"), "r") as f:
                config_space = read_json("\n".join(f.readlines()))
        else:
            config_space = autonet.get_hyperparameter_search_space()

        exact_cost_model = None
        exact_cost_model = AutoNetExactCostModel(autonet, data_manager, {
            "file_name": instance,
            "is_classification": (pipeline_config["problem_type"] in ['feature_classification', 'feature_multilabel']),
            "test_split": pipeline_config["test_split"]
        }, pipeline_config["memory_limit_mb"], pipeline_config["time_limit_per_entry"])

        initial_design_learner[1].add_result(run_result_dir, config_space, instance, exact_cost_model)
        warmstarted_model_builder.add_result(run_result_dir, config_space, origin=instance)
        return dict()
    
    def get_pipeline_config_options(self):
        return [
            ConfigOption("memory_limit_mb", default=None, type=int),
            ConfigOption("time_limit_per_entry", default=None, type=int),
            ConfigOption("only_finished_runs", default=False, type=to_bool)
        ]

class AutoNetExactCostModel():
    def __init__(self, autonet, dm, dm_kwargs, memory_limit_mb=None, time_limit=None):
        self.autonet = autonet
        self.dm = dm
        self.dm_kwargs = dm_kwargs
        self.memory_limit_mb = memory_limit_mb
        self.time_limit = time_limit
    
    def __enter__(self):
        self.dm.read_data(**self.dm_kwargs)
        return self
    
    def evaluate(self, config, budget):
        if self.memory_limit_mb is None and self.time_limit is None:
            return self._evaluate(config, budget)

        print("enforce memory limit", self.memory_limit_mb, "and time limit", self.time_limit)
        import pynisher
        start_time = time.time()

        limit_train = pynisher.enforce_limits(mem_in_mb=self.memory_limit_mb, wall_time_in_s=self.time_limit)(self._evaluate)
        result = limit_train(config, budget)

        if (limit_train.exit_status == pynisher.TimeoutException):
            raise Exception("Time limit reached. Took " + str((time.time()-start_time)) + " seconds with budget " + str(budget))
        elif (limit_train.exit_status == pynisher.MemorylimitException):
            raise Exception("Memory limit reached. Took " + str((time.time()-start_time)) + " seconds with budget " + str(budget))
        elif (limit_train.exit_status != 0):
            raise Exception("Exception in train pipeline. Took " + str((time.time()-start_time)) + " seconds with budget " + str(budget))
        return result
    
    def _evaluate(self, config, budget):
        return self.autonet.refit(
            X_train=self.dm.X_train, Y_train=self.dm.Y_train, X_valid=self.dm.X_valid, Y_valid=self.dm.Y_valid,
            hyperparameter_config=config.get_dictionary(), budget=budget)
    
    def __exit__(self, error_type, error_value, error_traceback):
        del self.dm
        return error_type is not None
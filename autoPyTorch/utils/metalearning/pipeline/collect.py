
import logging
import os
from copy import copy

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

    def fit(self, pipeline_config, initial_design_learner, run_result_dir, autonet=None):
        logger = logging.getLogger("metalearning")
        print("Collecting " + run_result_dir)
        run_result = logged_results_to_HBS_result(run_result_dir)
        if os.path.exists(os.path.join(run_result_dir, "configspace.pcs")):
            with open(os.path.join(run_result_dir, "configspace.pcs"), "r") as f:
                config_space = read_pcs(f.readlines())
        elif os.path.exists(os.path.join(run_result_dir, "configspace.json")):
            with open(os.path.join(run_result_dir, "configspace.json"), "r") as f:
                config_space = read_json(f.readlines())
        else:
            config_space = autonet.get_hyperparameter_search_space()
        initial_design_learner.add_result(run_result, config_space)
        return dict()
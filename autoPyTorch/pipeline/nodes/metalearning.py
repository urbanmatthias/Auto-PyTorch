__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"

import os
from autoPyTorch.pipeline.base.pipeline_node import PipelineNode

from autoPyTorch.components.lr_scheduler.lr_schedulers import AutoNetLearningRateSchedulerBase

import ConfigSpace
import ConfigSpace.hyperparameters as CSH
from autoPyTorch.utils.configspace_wrapper import ConfigWrapper
from autoPyTorch.utils.config.config_option import ConfigOption
from autoPyTorch.training.lr_scheduling import LrScheduling
from hpbandster.metalearning.initial_design import InitialDesign
import pickle

class MetaLearning(PipelineNode):
    def fit(self, pipeline_config):
        initial_design = pipeline_config["initial_design"]
        warmstarted_model = pipeline_config["warmstarted_model"]

        if initial_design is not None:
            with open(initial_design, "rb") as f:
                initial_design = pickle.load(f)
        
        if warmstarted_model is not None:
            with open(warmstarted_model, "rb") as f:
                warmstarted_model = pickle.load(f)

        return {"warmstarted_model": warmstarted_model, "initial_design": initial_design}

    def get_pipeline_config_options(self):
        options = [
            ConfigOption(name="initial_design", default=None, type="directory"),
            ConfigOption(name="warmstarted_model", default=None, type="directory")
        ]
        return options

class MetaLearningSaveModelWeights(PipelineNode):
    def fit(self, pipeline_config, final_metric_score, optimized_hyperparameter_config, budget, warmstarted_model):
        if warmstarted_model is not None:    
            file_name = os.path.join(pipeline_config["result_logger_dir"], 'warmstarted_model_weights_history.txt')
            with open(file_name, "w") as f:
                warmstarted_model.print_weight_history(f)
        return {'final_metric_score': final_metric_score,
                'optimized_hyperparameter_config': optimized_hyperparameter_config,
                'budget': budget}
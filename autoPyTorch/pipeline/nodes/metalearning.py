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
    def fit(self, pipeline_config, result_loggers):
        initial_design = pipeline_config["initial_design"]
        warmstarted_model = pipeline_config["warmstarted_model"]

        if initial_design is not None:
            with open(initial_design, "rb") as f:
                initial_design = pickle.load(f)
        
            max_configs = pipeline_config["max_initial_design_configs"]
            if max_configs > 0 and max_configs < len(initial_design):
                initial_design = InitialDesign(initial_design.configs[:max_configs], initial_design.origins[:max_configs])
        
        if warmstarted_model is not None:
            with open(warmstarted_model, "rb") as f:
                warmstarted_model = pickle.load(f)

            result_loggers = [warmstarted_model_weights_logger(directory=pipeline_config["result_logger_dir"], warmstarted_model=warmstarted_model)] + result_loggers
        return {"warmstarted_model": warmstarted_model, "initial_design": initial_design, "result_loggers": result_loggers}

    def get_pipeline_config_options(self):
        options = [
            ConfigOption(name="initial_design", default=None, type="directory"),
            ConfigOption(name="warmstarted_model", default=None, type="directory"),
            ConfigOption(name="max_initial_design_configs", default=0, type=int)
        ]
        return options

class warmstarted_model_weights_logger(object):
    def __init__(self, directory, warmstarted_model):
        self.directory = directory
        self.warmstarted_model = warmstarted_model
        
        self.file_name = os.path.join(directory, 'warmstarted_model_weights_history.txt')


    def new_config(self, *args, **kwargs):
        pass

    def __call__(self, job):
        if job.result is None:
            return
        with open(self.file_name, "w") as f:
            self.warmstarted_model.print_weight_history(f)
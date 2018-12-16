__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"


from autoPyTorch.pipeline.base.pipeline_node import PipelineNode

from autoPyTorch.components.lr_scheduler.lr_schedulers import AutoNetLearningRateSchedulerBase

import ConfigSpace
import ConfigSpace.hyperparameters as CSH
from autoPyTorch.utils.configspace_wrapper import ConfigWrapper
from autoPyTorch.utils.config.config_option import ConfigOption
from autoPyTorch.training.lr_scheduling import LrScheduling
import pickle

class MetaLearning(PipelineNode):
    def __init__(self):
        super(MetaLearning, self).__init__()

        self.lr_scheduler = dict()
        self.lr_scheduler_settings = dict()

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
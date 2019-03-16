__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"

import pickle

from autoPyTorch.pipeline.base.pipeline_node import PipelineNode
from autoPyTorch.utils.config.config_option import ConfigOption


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
            ConfigOption(name="warmstarted_model", default=None, type="directory"),
            # ConfigOption(name="max_initial_design_configs", default=1, type=int)
        ]
        return options

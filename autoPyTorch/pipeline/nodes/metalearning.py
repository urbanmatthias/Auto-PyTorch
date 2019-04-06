__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"

import pickle

from autoPyTorch.pipeline.base.pipeline_node import PipelineNode
from autoPyTorch.utils.config.config_option import ConfigOption


class MetaLearning(PipelineNode):
    def fit(self, pipeline_config, dataset_info):
        initial_design = pipeline_config["initial_design"]
        warmstarted_model = pipeline_config["warmstarted_model"]

        if initial_design is not None and "<leave_out_suffix>" in initial_design:
            initial_design = initial_design.replace("<leave_out_suffix>",  "_leave_out_%s" % "_".join(dataset_info.name.split(":")))
        if warmstarted_model is not None and "<leave_out_suffix>" in warmstarted_model:
            warmstarted_model = warmstarted_model.replace("<leave_out_suffix>",  "_leave_out_%s" % "_".join(dataset_info.name.split(":")))

        if initial_design is not None:
            with open(initial_design, "rb") as f:
                initial_design = pickle.load(f)
        
        if warmstarted_model is not None:
            with open(warmstarted_model, "rb") as f:
                warmstarted_model = pickle.load(f)
                warmstarted_model.choose_sample_budget_strategy = pipeline_config["warmstarted_model_sample_budget"]
                warmstarted_model.choose_similarity_budget_strategy = pipeline_config["warmstarted_model_similarity_budget"]
                warmstarted_model.num_nonzero_weight = pipeline_config["warmstarted_model_num_nonzero_weight"]
                warmstarted_model.weight_type = pipeline_config["warmstarted_model_weight_type"]
                warmstarted_model.average_type = pipeline_config["warmstarted_model_average_type"]


        return {"warmstarted_model": warmstarted_model, "initial_design": initial_design}

    def get_pipeline_config_options(self):
        options = [
            ConfigOption(name="initial_design", default=None, type="directory"),
            ConfigOption(name="warmstarted_model", default=None, type="directory"),
            ConfigOption(name="warmstarted_model_similarity_budget", default="max_with_model", type=str, choices=["max_with_model", "current"]),
            ConfigOption(name="warmstarted_model_sample_budget", default="max_available", type=str, choices=["max_available", "current"]),
            ConfigOption(name="warmstarted_model_num_nonzero_weight", default=0, type=int),
            ConfigOption(name="warmstarted_model_weight_type", type=str, default="max_likelihood",
                         choices=["max_likelihood", "likelihood", "likelihood_sum", "log_likelihood"]),
            ConfigOption(name="warmstarted_model_average_type", type=str, default="weighted_arithmetic_mean",
                         choices=["weighted_arithmetic_mean", "weighted_geometric_mean"])
        ]
        return options

__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"

import pickle

from autoPyTorch.pipeline.base.pipeline_node import PipelineNode
from autoPyTorch.utils.config.config_option import ConfigOption, to_bool

from hpbandster.metalearning.initial_design import Hydra


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
                if pipeline_config["initial_design_force_num_sh_iter"]:
                    initial_design.num_configs_per_sh_iter = Hydra.get_num_configs_per_sh_iter(
                        num_min_budget=len(initial_design),
                        num_max_budget=1,
                        num_sh_iter=pipeline_config["initial_design_force_num_sh_iter"])
        
        if warmstarted_model is not None:
            with open(warmstarted_model, "rb") as f:
                warmstarted_model = pickle.load(f)
                warmstarted_model.choose_sample_budget_strategy = pipeline_config["warmstarted_model_sample_budget"]
                warmstarted_model.choose_similarity_budget_strategy = pipeline_config["warmstarted_model_similarity_budget"]
                warmstarted_model.weight_type = pipeline_config["warmstarted_model_weight_type"]


        return {"warmstarted_model": warmstarted_model, "initial_design": initial_design}

    def get_pipeline_config_options(self):
        options = [
            ConfigOption(name="initial_design", default=None, type="directory"),
            ConfigOption(name="warmstarted_model", default=None, type="directory"),
            ConfigOption(name="warmstarted_model_similarity_budget", default="current", type=str, choices=["max_with_model", "current"]),
            ConfigOption(name="warmstarted_model_sample_budget", default="max_available", type=str, choices=["max_available", "current"]),
            ConfigOption(name="warmstarted_model_weight_type", type=str, default="likelihood", choices=["max_likelihood", "likelihood"]),
            ConfigOption(name="initial_design_force_num_sh_iter", type=int, default=0)
        ]
        return options

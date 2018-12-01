import logging
import os
from copy import copy

from autoPyTorch.pipeline.base.pipeline_node import PipelineNode
from autoPyTorch.utils.benchmarking.benchmark_pipeline.prepare_result_folder import \
    get_run_result_dir
from autoPyTorch.utils.config.config_option import ConfigOption, to_bool
from autoPyTorch.utils.metalearning.meta_learning_model import MetaLearningModel
from hpbandster.core.result import logged_results_to_HBS_result


class InitializeMetaLearningModel(PipelineNode):

    def fit(self, pipeline_config):
        return {'meta_learning_model': MetaLearningModel()}

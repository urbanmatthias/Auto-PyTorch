import logging
import time

from autoPyTorch.pipeline.base.pipeline_node import PipelineNode
from autoPyTorch.utils.benchmarking.benchmark_pipeline.benchmark_settings import \
    BenchmarkSettings
from autoPyTorch.utils.config.config_file_parser import ConfigFileParser
from autoPyTorch.utils.config.config_option import ConfigOption, to_bool
from hpbandster.metalearning.initial_design import Hydra, LossMatrixComputation, normalize_strategies
from hpbandster.metalearning.model_warmstarting import WarmstartedModelBuilder


class MetaLearningSettings(BenchmarkSettings):
    def fit(self, pipeline_config):
        logger = logging.getLogger('metalearning')
        logger.setLevel(self.logger_settings[pipeline_config['log_level']])
        logger.info("Start building Meta Learning Models")
        print("Start building Meta Learning Models", time.time())

        # log level for autonet is set in SetAutoNetConfig

        return {'run_id_range': pipeline_config['run_id_range'],
                'initial_design_learner': (Hydra(normalize_loss=normalize_strategies[pipeline_config["initial_design_normalize_loss"]],
                                                 bigger_is_better=False), LossMatrixComputation(bigger_is_better=False)) ,
                'warmstarted_model_builder': WarmstartedModelBuilder()}

    def get_pipeline_config_options(self):
        options = [
            ConfigOption("run_id_range", type=str, default=None),
            ConfigOption("log_level", default="info", type=str, choices=list(self.logger_settings.keys())),
            ConfigOption('result_dir', default=None, type='directory', required=True),
            ConfigOption("benchmark_name", default=None, type=str, required=True),
            ConfigOption("initial_design_normalize_loss", default="normalized_distance_to_min",
                         type=str, choices=normalize_strategies.keys())
        ]
        return options

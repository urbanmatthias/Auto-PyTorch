import logging

from autoPyTorch.pipeline.base.pipeline_node import PipelineNode
from autoPyTorch.utils.benchmarking.benchmark_pipeline.benchmark_settings import \
    BenchmarkSettings
from autoPyTorch.utils.config.config_file_parser import ConfigFileParser
from autoPyTorch.utils.config.config_option import ConfigOption
from hpbandster.metalearning.initial_design import Hydra
from hpbandster.metalearning.model_warmstarting import WarmstartedModelBuilder


class MetaLearningSettings(BenchmarkSettings):
    def fit(self, pipeline_config):
        logger = logging.getLogger('metalearning')
        logger.setLevel(self.logger_settings[pipeline_config['log_level']])
        logger.info("Start building Meta Learning Models")

        # log level for autonet is set in SetAutoNetConfig

        return {'run_id_range': pipeline_config['run_id_range'],
                'initial_design_learner': Hydra(num_processes=pipeline_config["num_processes"]),
                'warmstarted_model_builder': WarmstartedModelBuilder()}

    def get_pipeline_config_options(self):
        options = [
            ConfigOption("run_id_range", type=str, default=None),
            ConfigOption("log_level", default="info", type=str, choices=list(self.logger_settings.keys())),
            ConfigOption('result_dir', default=None, type='directory', required=True),
            ConfigOption('num_processes', default=0, type=int)
        ]
        return options

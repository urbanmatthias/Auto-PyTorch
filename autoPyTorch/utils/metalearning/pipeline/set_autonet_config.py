import os

from autoPyTorch.pipeline.base.pipeline_node import PipelineNode
from autoPyTorch.utils.config.config_file_parser import ConfigFileParser
from autoPyTorch.utils.config.config_option import ConfigOption, to_bool
from autoPyTorch.utils.hyperparameter_search_space_update import \
    parse_hyperparameter_search_space_updates


class SetAutoNetConfig(PipelineNode):

    def fit(self, pipeline_config, autonet, data_manager, run_result_dir):
        parser = autonet.get_autonet_config_file_parser()
        config = parser.read(os.path.join(run_result_dir, "autonet.config"))
        parser.set_defaults(config)

        updates=None
        if os.path.exists(os.path.join(run_result_dir, "hyperparameter_search_space_updates")):
            updates = parse_hyperparameter_search_space_updates(
                os.path.join(run_result_dir, "hyperparameter_search_space_updates"))

        if (pipeline_config['use_dataset_metric'] and data_manager.metric is not None):
            config['train_metric'] = data_manager.metric
        if (pipeline_config['use_dataset_max_runtime'] and data_manager.max_runtime is not None):
            config['max_runtime'] = data_manager.max_runtime

        if "hyperparameter_search_space_updates" in config:
            del config["hyperparameter_search_space_updates"]

        config['log_level'] = pipeline_config['log_level']
        autonet.update_autonet_config(hyperparameter_search_space_updates=updates,
                                      **config)
        return dict()

    def get_pipeline_config_options(self):
        options = [
            ConfigOption("use_dataset_metric", default=False, type=to_bool),
            ConfigOption("use_dataset_max_runtime", default=False, type=to_bool),
        ]
        return options

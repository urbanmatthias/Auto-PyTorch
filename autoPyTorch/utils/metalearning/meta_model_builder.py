from autoPyTorch.pipeline.base.pipeline import Pipeline
from autoPyTorch.utils.benchmarking.benchmark_pipeline import CreateAutoNet
from autoPyTorch.utils.benchmarking.visualization_pipeline import ReadInstanceInfo
from autoPyTorch.utils.config.config_file_parser import ConfigFileParser
from autoPyTorch.utils.metalearning.pipeline import (Collect,
                                                     MetaLearningSettings,
                                                     MetaLearningFit,
                                                     ForInstance,
                                                     ForAutoNetConfig,
                                                     ForRun,
                                                     SetAutoNetConfig)


class MetaModelBuilder():
    def __init__(self):
        self.pipeline = self.get_pipeline()

    def run(self, **meta_learning_config):
        config = self.pipeline.get_pipeline_config(**meta_learning_config, throw_error_if_invalid=False)
        self.pipeline.fit_pipeline(pipeline_config=config)
    
    def get_benchmark_config_file_parser(self):
        return ConfigFileParser(self.get_pipeline().get_pipeline_config_options())

    def get_pipeline(self):
        return Pipeline([  # [pipeline_config]
            MetaLearningSettings(),  # pipeline_config --> run_id_range --> initial_design_learner
            ForInstance([  # [pipeline_config, run_id_range, instance, initial_design_learner]
                ForAutoNetConfig([  # [pipeline_config, run_id_range, config_file, initial_design_learner]
                    ForRun([  # [pipeline_config, run_number, run_id, autonet_config_file, run_result_dir, initial_design_learner]
                        ReadInstanceInfo(), # pipeline_config, run_result_dir -> data_manager
                        CreateAutoNet(), # data_manager --> autonet
                        SetAutoNetConfig(),  # pipeline_config, autonet, data_manager, run_result_dir
                        Collect()  # pipeline_config, autonet --> initial_design_learner
                    ])
                ])
            ]),
            MetaLearningFit()
        ])

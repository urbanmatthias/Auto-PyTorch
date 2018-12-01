from autoPyTorch.pipeline.base.pipeline import Pipeline
from autoPyTorch.utils.benchmarking.benchmark_pipeline import (CreateAutoNet,
                                                               ReadInstanceData)
from autoPyTorch.utils.benchmarking.visualization_pipeline import ReadInstanceInfo
from autoPyTorch.utils.config.config_file_parser import ConfigFileParser
from autoPyTorch.utils.metalearning.pipeline import (FitMetaLearningModel,
                                                     MetaLearningSettings,
                                                     ComputeMetaFeatures,
                                                     WriteMetaLearningModel,
                                                     ForInstance,
                                                     ForAutoNetConfig,
                                                     ForRun,
                                                     SetAutoNetConfig,
                                                     InitializeMetaLearningModel)


class MetaModelBuilder():
    def __init__(self):
        self.pipeline = self.get_pipeline()

    def get_config_file_parser(self):
        return ConfigFileParser(self.pipeline.get_pipeline_config_options())

    def run(self, **meta_learning_config):
        config = self.pipeline.get_pipeline_config(**meta_learning_config)
        self.pipeline.fit_pipeline(pipeline_config=config)

    def get_pipeline(self):
        return Pipeline([  # [pipeline_config]
            MetaLearningSettings(),  # pipeline_config --> run_id_range, 
            InitializeMetaLearningModel(),  # pipeline_config --> meta_learning_model
            ForInstance([  # [pipeline_config, run_id_range, instance, meta_learning_model]
                ReadInstanceData(),  # pipeline_config, instance --> data_manager
                ComputeMetaFeatures(),  # pipeline_config, data_manager --> meta_features
                CreateAutoNet(), # data_manager --> autonet
                ForAutoNetConfig([  # [pipeline_config, data_manager, run_id_range, config_file, meta_features, autonet, meta_learning_model]
                    ForRun([  # [pipeline_config, data_manager, run_number, run_id, autonet_config_file, meta_features, run_result_dir, autonet, meta_learning_model]
                        SetAutoNetConfig(),  # pipeline_config, autonet, data_manager, run_result_dir
                        FitMetaLearningModel()  # pipeline_config, autonet, meta_features --> meta_learning_model
                    ])
                ])
            ]),
            WriteMetaLearningModel()
        ])

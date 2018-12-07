import traceback

from autoPyTorch.pipeline.base.sub_pipeline_node import SubPipelineNode
from autoPyTorch.utils.benchmarking.benchmark_pipeline import \
    ForAutoNetConfig as BaseForAutonetConfig
from autoPyTorch.utils.config.config_option import ConfigOption


class ForAutoNetConfig(BaseForAutonetConfig):
    def fit(self, pipeline_config, instance, data_manager, run_id_range, meta_features, meta_learning_model, autonet):
        for config_file in self.get_config_files(pipeline_config):
            try:
                self.sub_pipeline.fit_pipeline(pipeline_config=pipeline_config,
                    instance=instance, data_manager=data_manager,
                    run_id_range=run_id_range, autonet_config_file=config_file,
                    meta_features=meta_features, meta_learning_model=meta_learning_model, autonet=autonet)
            except Exception as e:
                print(e)
                traceback.print_exc()
        return dict()

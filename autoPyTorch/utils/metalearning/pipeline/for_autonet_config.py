import traceback

from autoPyTorch.pipeline.base.sub_pipeline_node import SubPipelineNode
from autoPyTorch.utils.benchmarking.benchmark_pipeline import \
    ForAutoNetConfig as BaseForAutonetConfig
from autoPyTorch.utils.config.config_option import ConfigOption


class ForAutoNetConfig(BaseForAutonetConfig):
    def fit(self, pipeline_config, instance, run_id_range, initial_design_learner):
        for config_file in self.get_config_files(pipeline_config):
            try:
                self.sub_pipeline.fit_pipeline(pipeline_config=pipeline_config,
                    instance=instance,
                    run_id_range=run_id_range, autonet_config_file=config_file,
                    initial_design_learner=initial_design_learner)
            except Exception as e:
                print(e)
                traceback.print_exc()
        return dict()

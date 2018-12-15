import os
import logging
from autoPyTorch.utils.config.config_option import ConfigOption, to_bool
from autoPyTorch.utils.config.config_file_parser import ConfigFileParser
from autoPyTorch.pipeline.base.sub_pipeline_node import SubPipelineNode
from autoPyTorch.utils.benchmarking.benchmark_pipeline import ForInstance as BaseForInstance
import traceback

class ForInstance(BaseForInstance):
    def fit(self, pipeline_config, run_id_range, initial_design_learner, warmstarted_model_builder):
        instances = self.get_instances(pipeline_config, instance_slice=self.parse_slice(pipeline_config["instance_slice"]))
        instance_result_dirs = next(os.walk(pipeline_config["result_dir"]))[1]
        for i, instance in enumerate(instances):
            if i >= 10:
                break
            if "_".join(instance.split(":")) not in instance_result_dirs:
                continue
            print('Process instance ' +  str(i) + ' of ' + str(len(instances)))
            try:
                self.sub_pipeline.fit_pipeline(pipeline_config=pipeline_config, instance=instance,
                    run_id_range=run_id_range, initial_design_learner=initial_design_learner, warmstarted_model_builder=warmstarted_model_builder)
            except Exception as e:
                print(e)
                traceback.print_exc()
        return dict()
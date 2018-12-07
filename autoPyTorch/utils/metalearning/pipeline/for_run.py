import logging
import os
import traceback

from autoPyTorch.pipeline.base.sub_pipeline_node import SubPipelineNode
from autoPyTorch.utils.benchmarking.benchmark_pipeline import ForRun as BaseForRun
from autoPyTorch.utils.benchmarking.benchmark_pipeline.prepare_result_folder import \
    get_run_result_dir
from autoPyTorch.utils.config.config_option import ConfigOption


class ForRun(BaseForRun):
    def fit(self, pipeline_config, data_manager, instance, meta_features, meta_learning_model, autonet, autonet_config_file, run_id_range):
        for run_number in self.parse_range(pipeline_config['run_number_range'], pipeline_config['num_runs']):
            for run_id in run_id_range:
                try:
                    run_result_dir = get_run_result_dir(pipeline_config, instance, autonet_config_file, run_id, run_number)
                    if not os.path.exists(run_result_dir):
                        logging.getLogger('metalearning').debug("Skipping " + run_result_dir + "because it does not exist")
                        continue
                    logging.getLogger('metalearning').info("Fit run " + str(run_id) + "_" + str(run_number))
                    self.sub_pipeline.fit_pipeline(pipeline_config=pipeline_config,
                        data_manager=data_manager, instance=instance, 
                        run_number=run_number, run_id=run_id, autonet_config_file=autonet_config_file,
                        meta_features=meta_features, run_result_dir=run_result_dir, autonet=autonet,
                        meta_learning_model=meta_learning_model)
                    
                except Exception as e:
                    print(e)
                    traceback.print_exc()
        return dict()

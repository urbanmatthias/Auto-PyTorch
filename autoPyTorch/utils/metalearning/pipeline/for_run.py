import logging
import os
import traceback

from autoPyTorch.pipeline.base.sub_pipeline_node import SubPipelineNode
from autoPyTorch.utils.benchmarking.benchmark_pipeline import ForRun as BaseForRun
from autoPyTorch.utils.benchmarking.benchmark_pipeline.prepare_result_folder import \
    get_run_result_dir
from autoPyTorch.utils.benchmarking.visualization_pipeline.collect_trajectories import parse_run_folder_name
from autoPyTorch.utils.config.config_option import ConfigOption


class ForRun(BaseForRun):
    def fit(self, pipeline_config, instance, initial_design_learner, warmstarted_model_builder, autonet_config_file, run_id_range):
        run_number_range = self.parse_range(pipeline_config['run_number_range'], pipeline_config['num_runs'])
        instance_result_dir = os.path.abspath(os.path.join(get_run_result_dir(pipeline_config, instance, autonet_config_file, "0", "0"), ".."))
        if not os.path.exists(instance_result_dir):
            return dict()
        run_result_dirs = next(os.walk(instance_result_dir))[1]
        for run_result_dir in run_result_dirs:
            run_id, run_id_int, run_number = parse_run_folder_name(run_result_dir)
            run_result_dir = get_run_result_dir(pipeline_config, instance, autonet_config_file, run_id, run_number)
            if (run_id_range is not None and run_id_int not in run_id_range) or run_number not in run_number_range:
                continue
            logging.getLogger('metalearning').info("Fit run " + str(run_id) + "_" + str(run_number))
            try:
                self.sub_pipeline.fit_pipeline(pipeline_config=pipeline_config,
                    instance=instance, 
                    run_number=run_number, run_id=run_id, autonet_config_file=autonet_config_file,
                    run_result_dir=run_result_dir,
                    initial_design_learner=initial_design_learner,
                    warmstarted_model_builder=warmstarted_model_builder)
            except Exception as e:
                print(e)
                traceback.print_exc()
        return dict()
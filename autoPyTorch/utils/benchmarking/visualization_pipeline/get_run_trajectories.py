from hpbandster.core.result import logged_results_to_HBS_result
from autoPyTorch.pipeline.base.pipeline_node import PipelineNode
from autoPyTorch.utils.config.config_option import ConfigOption, to_bool
from autoPyTorch.utils.benchmarking.benchmark_pipeline.prepare_result_folder import get_run_result_dir
from autoPyTorch.pipeline.nodes import MetricSelector, LogFunctionsSelector
from copy import copy
import os
import logging

class GetRunTrajectories(PipelineNode):

    def fit(self, pipeline_config, autonet, run_result_dir):
        parser = autonet.get_autonet_config_file_parser()
        autonet_config = parser.read(os.path.join(run_result_dir, "autonet.config"))
        metrics = autonet.pipeline[MetricSelector.get_name()].metrics
        log_functions = autonet.pipeline[LogFunctionsSelector.get_name()].log_functions

        if pipeline_config["only_finished_runs"] and not os.path.exists(os.path.join(run_result_dir, "summary.json")):
            logging.getLogger('benchmark').info('Skipping ' + run_result_dir + ' because the run is not finished yet')
            return {"trajectories": dict(), "optimize_metric": None}

        trajectories = build_run_trajectories(run_result_dir, autonet_config, metrics, log_functions)
        if "test_result" in trajectories:
            trajectories["test_%s" % autonet_config["optimize_metric"]] = trajectories["test_result"]
        return {"trajectories": trajectories,
                "optimize_metric": autonet_config["optimize_metric"]}
    
    def get_pipeline_config_options(self):
        options = [
            ConfigOption('only_finished_runs', default=True, type=to_bool),
            ConfigOption('result_dir', default=None, type='directory', required=True),
        ]
        return options


def build_run_trajectories(results_folder, autonet_config, metrics, log_functions):
    # parse results
    try:
        res = logged_results_to_HBS_result(results_folder)
        incumbent_trajectory = res.get_incumbent_trajectory(bigger_is_better=False, non_decreasing_budget=False)
    except:
        print("No incumbent trajectory found")
        return dict()

    # prepare
    metric_name = autonet_config["optimize_metric"]
    all_metrics = autonet_config["additional_metrics"] + [metric_name]
    additional_metric_names = [("val_" + m, metrics[m]) for m in all_metrics]
    additional_metric_names += [("train_" + m, metrics[m]) for m in all_metrics]
    additional_metric_names += [(l, log_functions[l]) for l in autonet_config["additional_logs"]]

    # initialize incumbent trajectories
    incumbent_trajectories = dict()
    
    # save incumbent trajectories
    for name, obj in additional_metric_names:
        tj = copy(incumbent_trajectory)
        log_available = [name in run["info"] for config_id, budget in zip(tj["config_ids"], tj["budgets"])
                                             for run in res.get_runs_by_id(config_id)
                                             if run["budget"] == budget]
        tj["values"] = [run["info"][name] for config_id, budget in zip(tj["config_ids"], tj["budgets"])
                                          for run in res.get_runs_by_id(config_id)
                                          if run["budget"] == budget and name in run["info"]]
        tj["losses"] = [obj.loss_transform(x) for x in tj["values"]]

        # filter out timestamps etc, where no log value is given
        for key, value_list in tj.items():
            if key in ["losses", "values"]:
                continue
            tj[key] = [value for i, value in enumerate(value_list) if log_available[i]]

        # add to result
        if tj["losses"]:
            incumbent_trajectories[name] = tj
        
        # insert dummy values
        dummy_value = obj.dummy_value if obj.dummy_value is not None else metrics[autonet_config["optimize_metric"]].dummy_value
        tj["losses"] = [float("inf")] + tj["losses"]
        tj["values"] = [dummy_value] + tj["values"]
        tj["times_finished"] = [0] + tj["times_finished"]

    return incumbent_trajectories

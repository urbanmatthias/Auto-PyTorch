from hpbandster.core.result import logged_results_to_HBS_result
from autoPyTorch.pipeline.base.pipeline_node import PipelineNode
from autoPyTorch.utils.config.config_option import ConfigOption, to_bool, to_list
from autoPyTorch.utils.benchmarking.benchmark_pipeline.prepare_result_folder import get_run_result_dir
from autoPyTorch.pipeline.nodes.metric_selector import MetricSelector
from autoPyTorch.pipeline.nodes import OneHotEncoding
from autoPyTorch.pipeline.nodes.ensemble import build_ensemble, read_ensemble_prediction_file
from hpbandster.core.result import logged_results_to_HBS_result
from copy import copy
import os
import logging
import math
import numpy as np

class GetEnsembleTrajectories(PipelineNode):

    def fit(self, pipeline_config, autonet, run_result_dir, train_metric, trajectories):
        if not pipeline_config["enable_ensemble"] or train_metric is None:
            return {"trajectories": trajectories, "train_metric": train_metric}
        
        # prepare some variables
        parser = autonet.get_autonet_config_file_parser()
        autonet_config = parser.read(os.path.join(run_result_dir, "autonet.config"))
        metrics = autonet.pipeline[MetricSelector.get_name()].metrics
        y_transform = autonet.pipeline[OneHotEncoding.get_name()].complete_y_tranformation
        result = logged_results_to_HBS_result(run_result_dir)
        filename = os.path.join(run_result_dir, "predictions_for_ensemble.json")
        test_filename = os.path.join(run_result_dir, "test_predictions_for_ensemble.json")

        # compute which logs need to be evaluated with which ensemble size
        plot_logs = [l for l in pipeline_config['plot_logs'] if l.startswith("ensemble")]
        sizes = set(map(lambda l: int(l.split(":")[1]), plot_logs))
        ensemble_plot_logs = {size: [l for l in plot_logs if int(l.split(":")[1]) == size] for size in sizes}

        # read the predictions
        predictions, labels, model_identifiers, timestamps = read_ensemble_prediction_file(filename=filename, y_transform=y_transform)
        test_data_available = False
        try:
            test_predictions, test_labels, test_model_identifiers, test_timestamps = read_ensemble_prediction_file(filename=test_filename, y_transform=y_transform)
            assert test_model_identifiers == model_identifiers and test_timestamps == timestamps
            test_data_available = True
        except:
            pass

        # compute the prediction subset used to compute performance over time
        start_time = min(map(lambda t: t["submitted"], timestamps))
        end_time = max(map(lambda t: t["finished"], timestamps))
        step = math.log(end_time - start_time) / (pipeline_config["num_ensemble_evaluations"] - 1)
        steps = start_time + np.exp(np.arange(step, step * (pipeline_config["num_ensemble_evaluations"] + 1), step))
        subset_indices = [np.array([i for i, t in enumerate(timestamps) if t["finished"] < s]) for s in steps]

        # iterate over the subset to compute performance over time
        ensemble_trajectories = dict()
        for subset in subset_indices:
            if len(subset) == 0:
                continue
            times_finished = max(timestamps[s]["finished"] for s in subset)

            # fit an ensemble for all sizes
            for ensemble_size, log_names in ensemble_plot_logs.items():
                
                subset_predictions = [predictions[s] for s in subset]
                subset_model_identifiers = [model_identifiers[s] for s in subset]

                # build an ensemble with current subset and size
                ensemble, _ = build_ensemble(result=result,
                    train_metric=metrics[train_metric], y_transform=y_transform, minimize=autonet_config["minimize"], ensemble_size=ensemble_size,
                    all_predictions=subset_predictions, labels=labels, model_identifiers=subset_model_identifiers)

                # get the ensemble predictions
                ensemble_prediction = ensemble.predict(subset_predictions)
                if test_data_available:
                    subset_test_predictions = [test_predictions[s] for s in test_predictions]
                    test_ensemble_prediction = ensemble.predict(subset_test_predictions)

                # evaluate the metrics
                for log_name in log_names:
                    metric_name = log_name.split(":")[2]

                    if metric_name.startswith("test_"):
                        metric = metrics[metric_name[5:]]
                        performance = metric(test_ensemble_prediction, test_labels)
                    else:
                        metric = metrics[metric_name]
                        performance = metric(ensemble_prediction, labels)

                    # save in trajectory
                    if log_name not in ensemble_trajectories:
                        ensemble_trajectories[log_name] = {"times_finished": [], "losses": []}
                    ensemble_trajectories[log_name]["times_finished"].append(times_finished - start_time)
                    ensemble_trajectories[log_name]["losses"].append(performance)
                    ensemble_trajectories[log_name]["flipped"] = False
        return {"trajectories": dict(trajectories, **ensemble_trajectories), "train_metric": train_metric}
    
    def get_pipeline_config_options(self):
        options = [
            ConfigOption('enable_ensemble', default=False, type=to_bool),
            ConfigOption('num_ensemble_evaluations', default=100, type=int),
        ]
        return options
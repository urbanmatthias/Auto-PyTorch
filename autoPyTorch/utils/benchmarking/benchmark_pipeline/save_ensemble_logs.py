from hpbandster.core.result import logged_results_to_HBS_result
from autoPyTorch.pipeline.base.pipeline_node import PipelineNode
from autoPyTorch.utils.config.config_option import ConfigOption, to_bool, to_list
from autoPyTorch.pipeline.nodes.metric_selector import MetricSelector
from autoPyTorch.pipeline.nodes import OneHotEncoding
from autoPyTorch.pipeline.nodes.ensemble import build_ensemble, read_ensemble_prediction_file
from hpbandster.core.result import logged_results_to_HBS_result
from copy import copy
import os
import logging
import math
import numpy as np
import json

class SaveEnsembleLogs(PipelineNode):

    def fit(self, pipeline_config, autonet, result_dir):
        if not pipeline_config["enable_ensemble"]:
            return dict()
        
        # prepare some variables
        autonet_config = autonet.get_current_autonet_config()
        metrics = autonet.pipeline[MetricSelector.get_name()].metrics
        train_metric = metrics[autonet_config["train_metric"]]
        y_transform = autonet.pipeline[OneHotEncoding.get_name()].complete_y_tranformation
        result = logged_results_to_HBS_result(result_dir)
        filename = os.path.join(result_dir, "predictions_for_ensemble.npy")
        test_filename = os.path.join(result_dir, "test_predictions_for_ensemble.npy")
        ensemble_log_filename = os.path.join(result_dir, "ensemble_log.json")

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
        last_times_finished = 0
        for subset in subset_indices:
            if len(subset) == 0:
                continue
            
            times_finished = max(timestamps[s]["finished"] for s in subset)
            if times_finished == last_times_finished:
                continue
            last_times_finished = times_finished
            subset_predictions = [np.copy(predictions[s]) for s in subset]
            subset_model_identifiers = [model_identifiers[s] for s in subset]

            # build an ensemble with current subset and size
            ensemble, _ = build_ensemble(result=result,
                train_metric=train_metric, minimize=autonet_config["minimize"], ensemble_size=autonet_config["ensemble_size"],
                all_predictions=subset_predictions, labels=labels, model_identifiers=subset_model_identifiers,
                only_consider_n_best=autonet_config["ensemble_only_consider_n_best"],
                sorted_initialization_n_best=autonet_config["ensemble_sorted_initialization_n_best"])

            # get the ensemble predictions
            ensemble_prediction = ensemble.predict(subset_predictions)
            if test_data_available:
                subset_test_predictions = [np.copy(test_predictions[s]) for s in subset]
                test_ensemble_prediction = ensemble.predict(subset_test_predictions)

            # evaluate the metrics
            metric_performances = dict()
            for metric_name, metric in metrics.items():
                if metric_name != autonet_config["train_metric"] and metric_name not in autonet_config["additional_metrics"]:
                    continue
                metric_performances[metric_name] = metric(ensemble_prediction, labels)
                if test_data_available:
                    metric_performances["test_%s" % metric_name] = metric(test_ensemble_prediction, test_labels)

            # write to log
            with open(ensemble_log_filename, "a") as f:
                print(json.dumps([
                    {"started": start_time, "finished": times_finished},
                    metric_performances,
                    sorted([(identifier, weight) for identifier, weight in zip(ensemble.identifiers_, ensemble.weights_) if weight > 0],
                           key=lambda x: -x[1]),
                    [ensemble.identifiers_[i] for i in ensemble.indices_],
                    {
                        "ensemble_size": ensemble.ensemble_size,
                        "metric": autonet_config["train_metric"],
                        "minimize": ensemble.minimize,
                        "sorted_initialization": ensemble.sorted_initialization,
                        "bagging": ensemble.bagging,
                        "mode": ensemble.mode,
                        "num_input_models": ensemble.num_input_models_,
                        "trajectory": ensemble.trajectory_,
                        "train_score": ensemble.train_score_
                    }
                ]), file=f)
        return dict()
 
    def get_pipeline_config_options(self):
        options = [
            ConfigOption('num_ensemble_evaluations', default=100, type=int)
        ]
        return options
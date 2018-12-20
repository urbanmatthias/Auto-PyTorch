__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"

import os
import numpy as np
import json
import glob
import time

from autoPyTorch.pipeline.base.pipeline_node import PipelineNode
from autoPyTorch.utils.config.config_option import ConfigOption
from autoPyTorch.pipeline.nodes.metric_selector import MetricSelector
from autoPyTorch.components.ensembles.ensemble_selection import EnsembleSelection

from hpbandster.core.result import logged_results_to_HBS_result

def predictions_for_ensemble(y_pred, y_true):
    return y_pred

def combine_predictions(data, pipeline_kwargs, X, Y):
    all_indices = None
    all_predictions = None
    for split, d in data.items():
        predictions = d["predictions"]
        indices = pipeline_kwargs[split]["valid_indices"] if d["from_valid"] else pipeline_kwargs[split]["train_indices"]
        all_indices = indices if all_indices is None else np.append(all_indices, indices)
        all_predictions = predictions if all_predictions is None else np.vstack((all_predictions, predictions))
    sorted_indices = np.argsort(all_indices)
    sorted_predictions = all_predictions[sorted_indices]
    return sorted_predictions.tolist(), Y[sorted_indices].tolist()

class EnableComputePredictionsForEnsemble(PipelineNode):
    """Put this Node in the training pipeline after the metric selector node"""
    def fit(self, pipeline_config, additional_metrics, refit):
        if refit:
            return dict()
        return {'additional_metrics': additional_metrics + [predictions_for_ensemble]}


class SavePredictionsForEnsemble(PipelineNode):
    """Put this Node in the training pipeline after the training node"""
    def fit(self, pipeline_config, loss, info, refit):
        if refit:
            return {"loss": loss, "info": info}

        if "val_predictions_for_ensemble" in info:
            predictions = info["val_predictions_for_ensemble"]
            from_valid = True
            del info["val_predictions_for_ensemble"]
        else:
            predictions = info["train_predictions_for_ensemble"]
            from_valid = False
        del info["train_predictions_for_ensemble"]

        result = {
            "combinator": combine_predictions,
            "data": {"predictions": predictions, "from_valid": from_valid}
        }
        return {"loss": loss, "info": info, "predictions_for_ensemble": result}

    def predict(self, Y):
        return {"Y": Y}


class BuildEnsemble(PipelineNode):
    """Put this node after the optimization algorithm node"""
    def fit(self, pipeline_config, final_metric_score, optimized_hyperparamater_config, budget):
        result = logged_results_to_HBS_result(pipeline_config["result_logger_dir"])
        filename = os.path.join(pipeline_config["result_logger_dir"], 'predictions_for_ensemble.json')
        train_metric = self.pipeline[MetricSelector.get_name()].metrics[pipeline_config["train_metric"]]
        ensemble_selection = EnsembleSelection(10, train_metric)
        
        all_predictions = list()
        labels = None
        model_identifiers = list()
        with open(filename, "r") as f:
            for line in f:
                job_id, budget, timestamps, predictions = json.loads(line)
                model_identifiers.append(job_id + [budget])
                all_predictions.append(np.array(predictions[0]))
                if labels is not None:
                    assert np.all(np.array(predictions[1]) == labels)
                else:
                    labels = np.array(predictions[1])
        ensemble_selection.fit(all_predictions, labels, model_identifiers)
        return {"final_metric_score": final_metric_score, "optimized_hyperparamater_config": optimized_hyperparamater_config, "budget": budget,
            "ensemble": ensemble_selection}

class AddEnsembleLogger(PipelineNode):
    """Put this node in fromt of the optimization algorithm node"""
    def fit(self, pipeline_config, result_loggers, refit=False):
        if refit:
            return dict()
        result_loggers = [ensemble_logger(directory=pipeline_config["result_logger_dir"], overwrite=True)] + result_loggers
        return {"result_loggers": result_loggers}

class ensemble_logger(object):
    def __init__(self, directory, overwrite):
        self.start_time = time.time()
        self.directory = directory
        self.overwrite = overwrite
        
        self.file_name = os.path.join(directory, 'predictions_for_ensemble.json')

        try:
            with open(self.file_name, 'x') as fh: pass
        except FileExistsError:
            if overwrite:
                with open(self.file_name, 'w') as fh: pass
            else:
                raise FileExistsError('The file %s already exists.'%self.file_name)
        except:
            raise

    def new_config(self, *args, **kwargs):
        pass

    def __call__(self, job):
        if job.result is None:
            return
        with open(self.file_name, "a") as f:
            print(json.dumps([job.id, job.kwargs['budget'], job.timestamps, job.result["predictions_for_ensemble"]]), file=f)
        del job.result["predictions_for_ensemble"]
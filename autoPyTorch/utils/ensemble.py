import os
import time
import numpy as np
import json
import math
from autoPyTorch.components.ensembles.ensemble_selection import EnsembleSelection
from autoPyTorch.core.api import AutoNet

def build_ensemble(result, train_metric, y_transform, minimize, ensemble_size, all_predictions, labels, model_identifiers):
    id2config = result.get_id2config_mapping()
    ensemble_selection = EnsembleSelection(ensemble_size, train_metric, minimize)

    # fit ensemble
    ensemble_selection.fit(np.array(all_predictions), labels, model_identifiers)
    ensemble_configs = dict()
    for identifier in ensemble_selection.get_selected_model_identifiers():
        ensemble_configs[tuple(identifier[:3])] = id2config[tuple(identifier[:3])]["config"]
    return ensemble_selection, ensemble_configs


def read_ensemble_prediction_file(filename, y_transform):
    all_predictions = list()
    all_timestamps = list()
    labels = None
    model_identifiers = list()
    with open(filename, "r") as f:
        i = 0
        for line in f:
            if i == 0:
                labels = np.array(json.loads(line))
                labels, _ = y_transform(labels)
                i += 1
                continue
            job_id, budget, timestamps, predictions = json.loads(line)
            model_identifiers.append(tuple(job_id + [budget]))
            predictions = np.array(predictions)
            all_predictions.append(predictions)
            all_timestamps.append(timestamps)

            # the following assert statement only works with accuracy when using cv
            # performance = train_metric(predictions, labels)
            # run_performance = next(filter(lambda run: run.budget == budget, result.get_runs_by_id(tuple(job_id)))).loss
            # assert math.isclose(run_performance * minimize, performance, rel_tol=0.002), str(run_performance * minimize) + "!=" + str(performance)
    return all_predictions, labels, model_identifiers, all_timestamps


def predictions_for_ensemble(y_pred, y_true):
    return y_pred


class test_predictions_for_ensemble():
    def __init__(self, autonet, X_test, Y_test):
        self.autonet = autonet
        self.X_test = X_test
        self.Y_test = Y_test
    
    def __call__(self, model, epochs):
        if self.Y_test is None or self.X_test is None:
            return float("nan")
        
        return AutoNet.predict(self.autonet, self.X_test, return_probabilities=True)[1], self.Y_test

def combine_predictions(data, pipeline_kwargs, X, Y):
    all_indices = None
    all_predictions = None
    for split, predictions in data.items():
        indices = pipeline_kwargs[split]["valid_indices"]
        assert len(predictions) == len(indices), "Different number of predictions and indices:" + str(len(predictions)) + "!=" + str(len(indices))
        all_indices = indices if all_indices is None else np.append(all_indices, indices)
        all_predictions = predictions if all_predictions is None else np.vstack((all_predictions, predictions))
    argsort = np.argsort(all_indices)
    sorted_predictions = all_predictions[argsort]
    sorted_indices = all_indices[argsort]
    return sorted_predictions.tolist(), Y[sorted_indices].tolist()

def combine_test_predictions(data, pipeline_kwargs, X, Y):
    predictions = [d[0] for d in data.values() if d == d]
    labels = [d[1] for d in data.values() if d == d]
    assert all(np.all(labels[0] == l) for l in labels[1:])
    assert len(predictions) == len(labels)
    if len(predictions) == 0:
        return None
    return np.mean(np.stack(predictions), axis=0).tolist(), labels[0].tolist()


class ensemble_logger(object):
    def __init__(self, directory, overwrite):
        self.start_time = time.time()
        self.directory = directory
        self.overwrite = overwrite
        self.labels = None
        self.test_labels = None
        
        self.file_name = os.path.join(directory, 'predictions_for_ensemble.json')
        self.test_file_name = os.path.join(directory, 'test_predictions_for_ensemble.json')

        try:
            with open(self.file_name, 'x') as fh: pass
        except FileExistsError:
            if overwrite:
                with open(self.file_name, 'w') as fh: pass
            else:
                raise FileExistsError('The file %s already exists.'%self.file_name)
        except:
            raise
        
        if os.path.exists(self.test_file_name) and not overwrite:
            raise FileExistsError('The file %s already exists.'%self.file_name)

    def new_config(self, *args, **kwargs):
        pass

    def __call__(self, job):
        if job.result is None:
            return
        if "predictions_for_ensemble" in job.result:
            predictions, labels = job.result["predictions_for_ensemble"]
            with open(self.file_name, "a") as f:
                if self.labels is None:
                    self.labels = labels
                    print(json.dumps(labels), file=f)
                else:
                    assert self.labels == labels
                print(json.dumps([job.id, job.kwargs['budget'], job.timestamps, predictions]), file=f)
            del job.result["predictions_for_ensemble"]

            if "test_predictions_for_ensemble" in job.result:
                if job.result["test_predictions_for_ensemble"] is not None:
                    test_predictions, test_labels = job.result["test_predictions_for_ensemble"]
                    with open(self.test_file_name, "a") as f:
                        if self.test_labels is None:
                            self.test_labels = test_labels
                            print(json.dumps(test_labels), file=f)
                        else:
                            assert self.test_labels == test_labels
                        print(json.dumps([job.id, job.kwargs['budget'], job.timestamps, test_predictions]), file=f)
                del job.result["test_predictions_for_ensemble"]
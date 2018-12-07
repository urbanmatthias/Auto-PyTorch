__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"


from autoPyTorch.pipeline.base.pipeline_node import PipelineNode
from autoPyTorch.utils.config.config_option import ConfigOption
from autoPyTorch.pipeline.nodes.metric_selector import MetricSelector
from autoPyTorch.components.ensembles.ensemble_selection import EnsembleSelection

def predictions_for_ensemble(y_true, y_pred):
    return y_pred.max(1)[1].cpu().numpy()

class PrepareEnsemble(PipelineNode):
    """Put this Node in the training pipeline after the metric selector node"""
    def fit(self, pipeline_config, additional_metrics):
        return {'additional_metrics': additional_metrics + [predictions_for_ensemble]}


def BuildEnsemble():
    """Put this node after the optimization algorithm node"""
    def fit(self, pipeline_config):
        result = self.parse_results(pipeline_config["result_logger_dir"])
        train_metric = self.pipeline[MetricSelector.get_name()].metrics[pipeline_config["train_metric"]]

        predictions = dict()
        for r in result.get_all_runs:
            prediction = r.info["val_predictions"] if "val_predictions" in r.info else r.info["train_predictions"]
            predictions[(r.config_id, r.budget)] = prediction
        

        return dict()
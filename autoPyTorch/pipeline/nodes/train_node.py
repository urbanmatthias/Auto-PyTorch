__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"

import torch
import time
import logging

import scipy.sparse
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

from autoPyTorch.pipeline.base.pipeline_node import PipelineNode

import ConfigSpace
import ConfigSpace.hyperparameters as CSH
from autoPyTorch.utils.configspace_wrapper import ConfigWrapper
from autoPyTorch.utils.config.config_option import ConfigOption, to_bool
from autoPyTorch.training.base_training import BaseTrainingTechnique, BaseBatchLossComputationTechnique
from autoPyTorch.training.trainer import Trainer
from autoPyTorch.training.lr_scheduling import LrScheduling
from autoPyTorch.training.budget_types import BudgetTypeTime, BudgetTypeEpochs

import signal

class TrainNode(PipelineNode):
    def __init__(self):
        super(TrainNode, self).__init__()
        self.default_minimize_value = True
        self.logger = logging.getLogger('autonet')
        self.training_techniques = dict()
        self.batch_loss_computation_techniques = dict()
        self.add_batch_loss_computation_technique("standard", BaseBatchLossComputationTechnique)

    def fit(self, hyperparameter_config, pipeline_config,
            train_loader, valid_loader,
            network, optimizer,
            train_metric, additional_metrics,
            log_functions,
            budget,
            loss_function,
            training_techniques,
            fit_start_time):
        hyperparameter_config = ConfigWrapper(self.get_name(), hyperparameter_config) 
        self.logger.debug("Start train. Budget: " + str(budget))

        trainer = Trainer(
            model=network,
            loss_computation=self.batch_loss_computation_techniques[hyperparameter_config["batch_loss_computation_technique"]](),
            metrics=[train_metric] + additional_metrics,
            criterion=loss_function,
            budget=budget,
            optimizer=optimizer,
            training_techniques=[t for t in training_techniques if not isinstance(t, BudgetTypeTime) and not isinstance(t, BudgetTypeEpochs)],
            budget_type=[t for t in training_techniques if isinstance(t, BudgetTypeTime) or isinstance(t, BudgetTypeEpochs)][0],
            device=Trainer.get_device(pipeline_config),
            logger=self.logger)
        trainer.prepare(pipeline_config, hyperparameter_config, fit_start_time)

        logs = network.logs
        epoch = network.epochs_trained
        training_start_time = time.time()
        while True:
            trainer.before_train_batches()
            # prepare epoch
            log = dict()
            
            train_metric_results, train_loss, stop_training = trainer.train(epoch + 1, train_loader)
            if valid_loader is not None:
                valid_metric_results = trainer.evaluate(valid_loader)

            log['loss'] = train_loss
            for i, metric in enumerate(trainer.metrics):
                log['train_' + metric.__name__] = train_metric_results[i]

                if valid_loader is not None:
                    log['val_' + metric.__name__] = valid_metric_results[i]

            # handle logs
            logs.append(log)
            # update_logs(t, budget, log, 5, epoch + 1, verbose, True)
            self.logger.debug("Epoch: " + str(epoch) + " : " + str(log))
            # print("Epoch: " + str(epoch) + " : " + str(log))
            
            if 'use_tensorboard_logger' in pipeline_config and pipeline_config['use_tensorboard_logger']:
                import tensorboard_logger as tl
                worker_path = 'Train/'
                tl.log_value(worker_path + 'budget', float(budget), int(time.time()))
                tl.log_value(worker_path + 'epoch', float(epoch+1), int(time.time()))
                for name, value in log.items():
                    tl.log_value(worker_path + name, float(value), int(time.time()))

            if trainer.after_train_batches(log, epoch):
                break

            if stop_training:
                break
            
            epoch += 1
            torch.cuda.empty_cache()

        # wrap up
        wrap_up_start_time = time.time()
        network.epochs_trained = epoch
        network.logs = logs
        opt_metric_name = 'train_' + train_metric.__name__
        if valid_loader is not None:
            opt_metric_name = 'val_' + train_metric.__name__

        if pipeline_config["minimize"]:
            final_log = min(logs, key=lambda x:x[opt_metric_name])
        else:
            final_log = max(logs, key=lambda x:x[opt_metric_name])

        loss = final_log[opt_metric_name] * (1 if pipeline_config["minimize"] else -1)

        self.logger.info("Finished train with budget " + str(budget) +
                         ": Preprocessing took " + str(int(training_start_time - fit_start_time)) +
                         "s, Training took " + str(int(wrap_up_start_time - training_start_time)) + 
                         "s, Wrap up took " + str(int(time.time() - wrap_up_start_time)) +
                         "s. Total time consumption in s: " + str(int(time.time() - fit_start_time)))
    
        return {'loss': loss, 'info': final_log}


    def predict(self, pipeline_config, network, predict_loader):
        device = torch.device('cuda:0' if pipeline_config['cuda'] else 'cpu')
        
        Y = predict(network, predict_loader, None, device)
        return {'Y': Y.detach().cpu().numpy()}
    
    def add_training_technique(self, name, training_technique):
        if (not issubclass(training_technique, BaseTrainingTechnique)):
            raise ValueError("training_technique type has to inherit from BaseTrainingTechnique")
        self.training_techniques[name] = training_technique
    
    def remove_training_technique(self, name):
        del self.training_techniques[name]
    
    def add_batch_loss_computation_technique(self, name, batch_loss_computation_technique):
        if (not issubclass(batch_loss_computation_technique, BaseBatchLossComputationTechnique)):
            raise ValueError("batch_loss_computation_technique type has to inherit from BaseBatchLossComputationTechnique, got " + str(batch_loss_computation_technique))
        self.batch_loss_computation_techniques[name] = batch_loss_computation_technique
    
    def remove_batch_loss_computation_technique(self, name, batch_loss_computation_technique):
        del self.batch_loss_computation_techniques[name]

    def get_hyperparameter_search_space(self, **pipeline_config):
        pipeline_config = self.pipeline.get_pipeline_config(**pipeline_config)
        cs = ConfigSpace.ConfigurationSpace()

        hp_batch_loss_computation = CSH.CategoricalHyperparameter("batch_loss_computation_technique",
            pipeline_config['batch_loss_computation_techniques'], default_value=pipeline_config['batch_loss_computation_techniques'][0])
        cs.add_hyperparameter(hp_batch_loss_computation)

        for name in pipeline_config['batch_loss_computation_techniques']:
            technique = self.batch_loss_computation_techniques[name]
            cs.add_configuration_space(prefix=name, configuration_space=technique.get_hyperparameter_search_space(**pipeline_config),
                delimiter=ConfigWrapper.delimiter, parent_hyperparameter={'parent': hp_batch_loss_computation, 'value': name})

        return self._apply_user_updates(cs)

    def get_pipeline_config_options(self):
        options = [
            ConfigOption(name="batch_loss_computation_techniques", default=list(self.batch_loss_computation_techniques.keys()),
                type=str, list=True, choices=list(self.batch_loss_computation_techniques.keys())),
            ConfigOption("minimize", default=self.default_minimize_value, type=to_bool, choices=[True, False]),
            ConfigOption("cuda", default=True, type=to_bool, choices=[True, False])
        ]
        for name, technique in self.training_techniques.items():
            options += technique.get_pipeline_config_options()
        for name, technique in self.batch_loss_computation_techniques.items():
            options += technique.get_pipeline_config_options()
        return options


def predict(network, test_loader, metrics, device, move_network=True):
    """ predict batchwise """
    # Build DataLoader
    if move_network:
        network = network.to(device)

    # Batch prediction
    network.eval()
    if metrics is not None:
        metric_results = [0] * len(metrics)
    
    N = 0.0
    for i, (X_batch, Y_batch) in enumerate(test_loader):
        # Predict on batch
        X_batch = Variable(X_batch).to(device)
        batch_size = X_batch.size(0)

        Y_batch_pred = network(X_batch).detach().cpu()

        if metrics is None:
            # Infer prediction shape
            if i == 0:
                Y_pred = Y_batch_pred
            else:
                # Add to prediction tensor
                Y_pred = torch.cat((Y_pred, Y_batch_pred), 0)
        else:
            for i, metric in enumerate(metrics):
                metric_results[i] += metric(Y_batch, Y_batch_pred) * batch_size

        N += batch_size
    
    if metrics is None:
        return Y_pred
    else:
        return [res / N for res in metric_results]


import time
import os
import torch

from torch.autograd import Variable
from autoPyTorch.utils.configspace_wrapper import ConfigWrapper

# from util.transforms import mixup_data, mixup_criterion
# from checkpoints import save_checkpoint

class Trainer(object):
    def __init__(self, metrics, loss_computation, model, criterion, budget, optimizer, training_techniques, logger, device):
        
        self.criterion = criterion
        self.optimizer = optimizer
        self.metrics = metrics
        self.model = model
        self.device = device

        for t in training_techniques:
            for key, value in t.training_components.items():
                setattr(self, key, value)
        self.training_techniques = training_techniques

        self.budget = budget
        self.loss_computation = loss_computation

        self.cumulative_time = 0
        self.logger = logger
        self.fit_start_time = None

        self.to(device)
    
    def prepare(self, pipeline_config, hyperparameter_config, fit_start_time):
        self.fit_start_time = fit_start_time
        self.loss_computation.set_up(
            pipeline_config=pipeline_config,
            hyperparameter_config=ConfigWrapper(hyperparameter_config["batch_loss_computation_technique"], hyperparameter_config),
            logger=self.logger)
        for t in self.training_techniques:
            t.set_up(trainer=self, pipeline_config=pipeline_config)
    
    def to(self, device):
        self.device = device
        self.model = self.model.to(device)
        self.criterion = self.criterion.to(device)
    
    @staticmethod
    def get_device(pipeline_config):
        if not torch.cuda.is_available():
            pipeline_config["cuda"] = False
        return torch.device('cuda:0' if pipeline_config['cuda'] else 'cpu')
    
    def on_epoch_start(self, log, epoch):
        for t in self.training_techniques:
            t.on_epoch_start(trainer=self, log=log, epoch=epoch)
    
    def on_epoch_end(self, log, epoch):
        return any([t.on_epoch_end(trainer=self, log=log, epoch=epoch) for t in self.training_techniques])
    
    def needs_eval_on_snapshot(self):
        return any([t.needs_eval_on_snapshot() for t in self.training_techniques])
    
    def final_eval(self, logs, valid_loader):
        final_log = None
        for t in self.training_techniques:
            log = t.select_log(trainer=self, logs=logs)
            if log:
                final_log = log
        final_log = final_log or logs[-1]

        if any([t.needs_eval_on_snapshot() for t in self.training_techniques]):
            if valid_loader is not None:
                    valid_metric_results = self.evaluate(valid_loader)
            for i, metric in enumerate(self.metrics):
                if valid_loader is not None:
                    final_log['val_' + metric.__name__] = valid_metric_results[i]
        return final_log

    def train(self, epoch, train_loader):
        '''
            Trains the model for a single epoch
        '''

        loss_sum = 0.0
        N = 0
        self.model.train()

        metric_results = [0] * len(self.metrics)
        for step, (data, targets) in enumerate(train_loader):
   
            # prepare
            data = data.to(self.device)
            targets = targets.to(self.device)

            data, criterion_kwargs = self.loss_computation.prepare_data(data, targets)
            data = Variable(data)
            batch_size = data.size(0)

            for t in self.training_techniques:
                t.on_batch_start(trainer=self, epoch=epoch, step=step, num_steps=len(train_loader), cumulative_time=self.cumulative_time)

            # used for SGDR with seconds as budget
            start_time_batch = time.time()
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss_func = self.loss_computation.criterion(**criterion_kwargs)
            loss = loss_func(self.criterion, outputs)
            loss.backward()
            self.optimizer.step()

            # used for SGDR with seconds as budget
            delta_time = time.time() - start_time_batch
            self.cumulative_time += delta_time

            # evaluate metrics
            for i, metric in enumerate(self.metrics):
                metric_results[i] += self.loss_computation.evaluate(metric, outputs, **criterion_kwargs) * batch_size

            loss_sum += loss.item() * batch_size
            N += batch_size

            if any([t.on_batch_end(batch_loss=loss.item(), trainer=self, epoch=epoch, step=step, num_steps=len(train_loader),
                                           cumulative_time=self.cumulative_time) for t in self.training_techniques]):
                return [res / N for res in metric_results], loss_sum / N, True

        return [res / N for res in metric_results], loss_sum / N, False


    def evaluate(self, test_loader):

        N = 0
        metric_results = [0] * len(self.metrics)
        
        self.model.eval()

        with torch.no_grad():
            for _, (data, targets) in enumerate(test_loader):
    
                data = data.to(self.device)
                targets = targets.to(self.device)

                data = Variable(data)
                targets = Variable(targets)
                
                batch_size = data.size(0)

                outputs = self.model(data)
                
                for i, metric in enumerate(self.metrics):
                    metric_results[i] += metric(outputs.data, targets.data) * batch_size

                N += batch_size

        self.model.train()
            
        return [res / N for res in metric_results]
    
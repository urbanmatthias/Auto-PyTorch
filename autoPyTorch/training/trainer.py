import time
import os
import torch

from torch.autograd import Variable

# from util.transforms import mixup_data, mixup_criterion
# from checkpoints import save_checkpoint

class Trainer(object):
    def __init__(self, metrics, loss_computation, model, criterion, budget, optimizer, scheduler, budget_type, device):
        
        self.criterion = criterion
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.metrics = metrics
        self.device = device

        self.budget = budget
        self.loss_computation = loss_computation

        self.budget_type = budget_type
        self.cumulative_time = 0
        self.model = model.to(self.device)

    def train(self, epoch, train_loader):
        '''
            Trains the model for a single epoch
        '''

        # train_size = int(0.9 * len(train_loader.dataset.train_data) / self.config.batch_size)
        loss_sum = 0.0
        N = 0

        # print('\33[1m==> Training epoch # {}\033[0m'.format(str(epoch)))
        
        self.model.train()

        metric_results = [0] * len(self.metrics)
        # start_time_step = time.time()
        for step, (data, targets) in enumerate(train_loader):
            # data_timer = time.time()
   
            data = data.to(self.device)
            targets = targets.to(self.device)

            data, criterion_kwargs = self.loss_computation.prepare_data(data, targets)

            data = Variable(data)
            
            batch_size = data.size(0)

            if epoch != 1 or step != 0:
                self.scheduler.step(epoch=self.cumulative_time)
            else:
                self.scheduler.step(epoch=epoch)

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
            self.scheduler.last_step = self.cumulative_time - delta_time - 1e-10

            # Each time before the learning rate restarts we save a checkpoint in order to create snapshot ensembles after the training finishes
            # if (epoch != 1 or step != 0) and (self.cumulative_time > self.config.T_e + delta_time + 5) and (self.scheduler.last_step < 0):
            #     save_checkpoint(int(round(self.cumulative_time)), self.model, self.optimizer, self.scheduler, self.config)

            for i, metric in enumerate(self.metrics):
                metric_results[i] += self.loss_computation.evaluate(metric, outputs, **criterion_kwargs) * batch_size

            loss_sum += loss.item() * batch_size
            N += batch_size

            # start_time_step = time.time()
            # print('Update', (metric_results[0] / N), 'loss', (loss_sum / N))

            if self.budget_type.during_train_batches(None, None):
                # print(' * Stopping at Epoch: [%d][%d/%d] for a budget of %.3f s' % (epoch, step + 1, train_size, self.config.budget))
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
    
import logging
import os
from copy import copy

import hpbandster.core.nameserver as hpns
from autoPyTorch.pipeline.base.pipeline_node import PipelineNode
from autoPyTorch.pipeline.nodes.optimization_algorithm import OptimizationAlgorithm
from autoPyTorch.utils.benchmarking.benchmark_pipeline.prepare_result_folder import \
    get_run_result_dir
from autoPyTorch.utils.config.config_option import ConfigOption, to_bool
from hpbandster.core.dispatcher import Job
from hpbandster.core.result import logged_results_to_HBS_result
from hpbandster.optimizers.config_generators.bohb import \
    BOHB as ConfigGeneratorBohb


class FitMetaLearningModel(PipelineNode):

    def fit(self, pipeline_config, instance, autonet, meta_features, meta_learning_model, run_result_dir):
        run_result = logged_results_to_HBS_result(run_result_dir)
        cg = ConfigGeneratorBohb(autonet.get_hyperparameter_search_space())

        results_for_budget = dict()
        build_model_jobs = dict()
        id2conf = run_result.get_id2config_mapping()
        for id in id2conf:
            for r in run_result.get_runs_by_id(id):
                j = Job(id, config=id2conf[id]['config'], budget=r.budget)
                if r.loss is None:
                    r.loss = float('inf')
                if r.info is None:
                    r.info = dict()
                j.result = {'loss': r.loss, 'info': r.info}
                j.error_logs = r.error_logs

                if r.budget not in results_for_budget:
                    results_for_budget[r.budget] = list()
                results_for_budget[r.budget].append(j)

                if r.loss is not None and r.budget not in build_model_jobs:
                    build_model_jobs[r.budget] = j
                    continue
                cg.new_result(j, update_model=False)
        for j in build_model_jobs.values():
            cg.new_result(j, update_model=True)
        
        logger = logging.getLogger('metalearning')
        logger.info('Number of finished jobs for each budget: '+ str({k: len(v) for k, v in results_for_budget.items()}))

        if pipeline_config['save_metafeatures']:
            meta_learning_model.add_meta_features(instance, meta_features)
            logger.info('Saved meta-features for instance ' + str(instance))

        if len(cg.kde_models) > 0 and pipeline_config['save_kde_models']:
            budgets_with_model = sorted(cg.kde_models.keys())
            logger.info('Budgets with model: ' + str(budgets_with_model))
            budget = budgets_with_model[-1]
            
            meta_learning_model.add_kde_model(instance, cg.kde_models[budget], cg.configspace)
            logger.info('Saved KDE model for budget: ' + str(budget))
        
        if pipeline_config['save_largest_budget_results']:
            meta_learning_model.add_largest_budget_jobs(instance,
                                                        results_for_budget[max(results_for_budget.keys())],
                                                        cg.configspace)
            logger.info('Saved ' + str(len(results_for_budget[max(results_for_budget.keys())])) + ' jobs for the largest budget: ' + str(max(results_for_budget.keys())))
        return dict()
    
    def get_pipeline_config_options(self):
        options = [
            ConfigOption('save_metafeatures', default=True, type=to_bool),
            ConfigOption('save_kde_models', default=True, type=to_bool),
            ConfigOption('save_largest_budget_results', default=True, type=to_bool)
        ]
        return options

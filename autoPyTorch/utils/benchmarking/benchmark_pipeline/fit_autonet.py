import time
import logging
from autoPyTorch.pipeline.base.pipeline_node import PipelineNode

class FitAutoNet(PipelineNode):

    def fit(self, autonet, data_manager, **kwargs):
        start_time = time.time()

        logging.getLogger('benchmark').debug("Fit autonet")

        result = autonet.fit(
            data_manager.X_train, data_manager.Y_train, 
            data_manager.X_valid, data_manager.Y_valid, 
            refit=False)
        config, score = result[0], result[1]

        return { 'fit_duration': int(time.time() - start_time), 
                 'incumbent_config': config,
                 'final_score': score }
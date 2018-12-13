from autoPyTorch.pipeline.base.pipeline_node import PipelineNode
from autoPyTorch.utils.config.config_option import ConfigOption
import pickle
import logging

class MetaLearningFit(PipelineNode):    
    def fit(self, pipeline_config, initial_design_learner):
        initial_design_learner.learn()
        logger = logging.getLogger('metalearning')
        with open(pipeline_config["save_filename"], "wb") as f:
            pickle.dump(initial_design_learner, f)
            logger.info('Success!')
        return dict()
    
    def get_pipeline_config_options(self):
        options = [
            ConfigOption("save_filename", default="./initial_design.pkl", type="directory")
        ]
        return options
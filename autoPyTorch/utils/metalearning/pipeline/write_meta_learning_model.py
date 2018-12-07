from autoPyTorch.pipeline.base.pipeline_node import PipelineNode
from autoPyTorch.utils.config.config_option import ConfigOption
import pickle
import logging

class WriteMetaLearningModel(PipelineNode):    
    def fit(self, pipeline_config, meta_learning_model):
        logger = logging.getLogger('metalearning')
        logger.info('Write meta learning model to disk at: ' + pipeline_config["save_filename"])
        try:
            with open(pipeline_config["save_filename"], "wb") as f:
                pickle.dump(meta_learning_model, f)
                logger.info('Success!')
        except:
            logger.warn('Error writing to disk. Try again!')
            fix_statsmodels_pickle()
            with open(pipeline_config["save_filename"], "wb") as f:
                pickle.dump(meta_learning_model, f)
                logger.info('Success!')
        return dict()
    
    def get_pipeline_config_options(self):
        options = [
            ConfigOption("save_filename", default="./metamodel.pkl", type="directory")
        ]
        return options

def fix_statsmodels_pickle():
    try:
        import copyreg
    except ImportError:
        import copy_reg as copyreg
    import types

    if hasattr(copyreg, 'dispatch_table') \
    and isinstance(copyreg.dispatch_table, dict) \
    and types.MethodType in copyreg.dispatch_table:
        method_pickler = copyreg.dispatch_table.get(types.MethodType)
        if hasattr(method_pickler, '__module__') \
        and 'statsmodels.graphics.functional' in method_pickler.__module__ \
        and hasattr(method_pickler, '__name__') \
        and '_pickle_method' in method_pickler.__name__:
            del copyreg.dispatch_table[types.MethodType]

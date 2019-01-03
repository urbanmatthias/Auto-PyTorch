from autoPyTorch.pipeline.base.pipeline_node import PipelineNode
from autoPyTorch.utils.config.config_option import ConfigOption
import pickle
import logging
import os

class MetaLearningFit(PipelineNode):    
    def fit(self, pipeline_config, initial_design_learner, warmstarted_model_builder):
        initial_design = initial_design_learner.learn()
        logger = logging.getLogger('metalearning')
        save_path = os.path.join(pipeline_config["save_path"], "initial_design.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(initial_design, f)
            logger.info('Success!')
        del initial_design_learner
        del initial_design
        
        warmstarted_model = warmstarted_model_builder.build()
        save_path = os.path.join(pipeline_config["save_path"], "warmstarted_model.pkl")
        try:
            with open(save_path, "wb") as f:
                pickle.dump(warmstarted_model, f)
                logger.info('Success!')
        except:
            logger.warn('Error writing to disk. Try again!')
            fix_statsmodels_pickle()
            with open(save_path, "wb") as f:
                pickle.dump(warmstarted_model, f)
                print('Success!')
        del warmstarted_model
        del warmstarted_model_builder
        return dict()
    
    def get_pipeline_config_options(self):
        options = [
            ConfigOption("save_path", default=".", type="directory")
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
from autoPyTorch.pipeline.base.pipeline_node import PipelineNode
from autoPyTorch.utils.config.config_option import ConfigOption, to_bool
import pickle
import logging
import os

class MetaLearningFit(PipelineNode):    
    def fit(self, pipeline_config, initial_design_learner, warmstarted_model_builder):
        if pipeline_config["calculate_loss_matrix_entry"] >= 0:
            assert not pipeline_config["learn_initial_design"] and not pipeline_config["learn_warmstarted_model"]
            initial_design_learner[1].write_loss(pipeline_config["loss_matrix_dir"], pipeline_config["calculate_loss_matrix_entry"],
                num_files=pipeline_config["loss_matrix_num_files"])

        if pipeline_config["learn_initial_design"]:
            assert pipeline_config["initial_design_max_total_budget"] is not None, "initial_design_max_total_budget needs to be specified"
            assert pipeline_config["initial_design_convergence_threshold"] is not None, "initial_design_convergence_threshold needs to be specified"

            losses, incumbent_dict = initial_design_learner[1].read_loss(pipeline_config["loss_matrix_dir"])
            initial_design_learner[0].set_incumbent_losses(losses, incumbent_dict)
            initial_design, cost = initial_design_learner[0].learn(
                max_total_budget = pipeline_config["initial_design_max_total_budget"],
                convergence_threshold = pipeline_config["initial_design_convergence_threshold"]
            )
            if initial_design is not None:
                logger = logging.getLogger('metalearning')
                save_path = os.path.join(pipeline_config["save_path"], "initial_design.pkl")
                with open(save_path, "wb") as f:
                    pickle.dump(initial_design, f)
                    logger.info('Success!')
                del initial_design_learner
                del initial_design
        
        if pipeline_config["learn_warmstarted_model"]:
            warmstarted_model = warmstarted_model_builder.build()
            if warmstarted_model is not None:
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
            ConfigOption("save_path", default=".", type="directory"),
            ConfigOption("learn_warmstarted_model", default=False, type=to_bool),
            ConfigOption("learn_initial_design", default=False, type=to_bool),
            ConfigOption("calculate_loss_matrix_entry", default=-1, type=int),
            ConfigOption("loss_matrix_dir", default="./loss_matrix.txt", type=to_bool),
            ConfigOption("loss_matrix_num_files", default=1, type=int),
            ConfigOption("initial_design_max_total_budget", default=None, type=float),
            ConfigOption("initial_design_convergence_threshold", default=None, type=float)
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
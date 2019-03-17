from autoPyTorch.pipeline.base.pipeline_node import PipelineNode
from autoPyTorch.utils.config.config_option import ConfigOption, to_bool
import pickle
import logging
import os

class MetaLearningFit(PipelineNode):    
    def fit(self, pipeline_config, initial_design_learner, warmstarted_model_builder, leave_out_instance_name):
        if pipeline_config["calculate_loss_matrix_entry"] >= 0:
            assert not pipeline_config["learn_initial_design"] and not pipeline_config["learn_warmstarted_model"]
            initial_design_learner[1].write_loss(collection_name=pipeline_config["benchmark_name"],
                                                 db_config=pipeline_config["loss_matrix_db_config"],
                                                 entry=pipeline_config["calculate_loss_matrix_entry"])
        
        if pipeline_config["print_missing_loss_matrix_entries"]:
            print("\n".join(map(str, initial_design_learner[1].missing_loss_matrix_entries(
                collection_name=pipeline_config["benchmark_name"],
                db_config=pipeline_config["loss_matrix_db_config"]))))

        # fit and save metemodels
        if not os.path.exists(pipeline_config["save_path"]):
            os.mkdir(pipeline_config["save_path"])
        leave_out_name_suffix = "_leave_out_%s" % leave_out_instance_name if leave_out_instance_name is not None else ""
        logger = logging.getLogger('metalearning')

        # fit and save initial design
        if pipeline_config["learn_initial_design"]:
            assert pipeline_config["initial_design_max_total_budget"] is not None, "initial_design_max_total_budget needs to be specified"
            assert pipeline_config["initial_design_convergence_threshold"] is not None, "initial_design_convergence_threshold needs to be specified"

            losses, incumbent_dict = initial_design_learner[1].read_loss(collection_name=pipeline_config["benchmark_name"],
                                                                         db_config=pipeline_config["loss_matrix_db_config"])
            initial_design_learner[0].set_incumbent_losses(losses, incumbent_dict)
            initial_design, cost = initial_design_learner[0].learn(
                max_total_budget = pipeline_config["initial_design_max_total_budget"],
                convergence_threshold = pipeline_config["initial_design_convergence_threshold"]
            )
            if initial_design is not None:
                save_path = os.path.join(pipeline_config["save_path"], "initial_design%s.pkl" % leave_out_name_suffix)
                with open(save_path, "wb") as f:
                    pickle.dump(initial_design, f)
                    logger.info('Success!')
                del initial_design_learner
                del initial_design
        
        # fit and save warmstarted model
        if pipeline_config["learn_warmstarted_model"]:
            warmstarted_model = warmstarted_model_builder.build()
            if warmstarted_model is not None:
                save_path = os.path.join(pipeline_config["save_path"], "warmstarted_model%s.pkl" % leave_out_name_suffix)
                try:  # this fails sometimes for some reason, then fix_statsmodels_pickle() needs to be called
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
            ConfigOption("print_missing_loss_matrix_entries", default=False, type=to_bool),
            ConfigOption("calculate_loss_matrix_entry", default=-1, type=int),
            ConfigOption("initial_design_max_total_budget", default=None, type=float),
            ConfigOption("initial_design_convergence_threshold", default=None, type=float),
            ConfigOption("loss_matrix_db_config", default=dict(), type=dict)
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
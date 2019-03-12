from autoPyTorch.pipeline.base.pipeline_node import PipelineNode
from autoPyTorch.utils.config.config_option import ConfigOption, to_bool

class MetalearningEval(PipelineNode):    
    def fit(self, pipeline_config, evaluator):
        if pipeline_config["metalearning_evaluate"]:
            score = evaluator.evaluate()
            print("Warmstarted model score on this benchmark", score)
        return dict()

    
    def get_pipeline_config_options(self):
        options = [
            ConfigOption("metalearning_evaluate", default=False, type=to_bool)
        ]
        return options
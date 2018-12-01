import autoPyTorch.utils.metalearning.metafeatures as mf
import numpy as np
from autoPyTorch.pipeline.base.pipeline_node import PipelineNode

metafeature_set = mf.subsets["all"]

class ComputeMetaFeatures(PipelineNode):    
    def fit(self, pipeline_config, data_manager, instance):
        features = mf.calculate_all_metafeatures(data_manager.X_train, data_manager.Y_train, data_manager.categorical_features,
            dataset_name=instance, calculate=metafeature_set)
        result = []

        for key in sorted(features.keys()):
            if features[key].type_ == "METAFEATURE":
                result.append(features[key].value)
        return {"meta_features": np.array(result)}

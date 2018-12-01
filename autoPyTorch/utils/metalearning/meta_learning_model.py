class MetaLearningModel():
    def __init__(self):
        self.meta_features = dict()
        self.kde_models = dict()
        self.largest_budget_jobs = dict()

    def add_meta_features(self, instance, meta_features):
        self.meta_features[instance] = meta_features
    
    def add_kde_model(self, instance, kde_model, config_space):
        if instance not in self.kde_models:
            self.kde_models[instance] = list()
        self.kde_models[instance].append({
            "kde_model": kde_model,
            "config_space": config_space
        })
    
    def add_largest_budget_jobs(self, instance, jobs, config_space):
        if instance not in self.largest_budget_jobs:
            self.largest_budget_jobs[instance] = list()
        self.largest_budget_jobs[instance].append({
            "jobs": jobs,
            "config_space": config_space
        })
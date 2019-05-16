import os
import logging
from autoPyTorch.utils.config.config_option import ConfigOption, to_bool
from autoPyTorch.utils.config.config_file_parser import ConfigFileParser
from autoPyTorch.pipeline.base.sub_pipeline_node import SubPipelineNode
import traceback

class ForInstance(SubPipelineNode):
    def fit(self, pipeline_config, task_id, run_id):
        instances = self.get_instances(pipeline_config, instance_slice=self.parse_slice(pipeline_config["instance_slice"]))
        for instance in instances:
            try:
                self.sub_pipeline.fit_pipeline(pipeline_config=pipeline_config, instance=instance, run_id=run_id, task_id=task_id)
            except Exception as e:
                print(e)
                traceback.print_exc()
        return dict()

    def get_pipeline_config_options(self):
        options = [
            ConfigOption("instances", default=None, type='directory', required=True),
            ConfigOption("instance_slice", default=None, type=str),
            ConfigOption("dataset_root", default=ConfigFileParser.get_autonet_home(), type='directory'),
        ]
        return options

    @staticmethod
    def get_instances(benchmark_config, instances_must_exist=True, instance_slice=None):
        # get list of instances
        instances = []
        if os.path.isfile(benchmark_config["instances"]):
            with open(benchmark_config["instances"], "r") as instances_file:
                for line in instances_file:
                    if line.strip().startswith("openml"):
                        instances.append(line.strip())
                        continue
                        
                    if line.strip().startswith("["):
                        instances.append([make_path(path, benchmark_config["dataset_root"]) for path in line.strip(' []').split(',')])
                        continue

                    instance = os.path.abspath(os.path.join(benchmark_config["dataset_root"], line.strip()))
                    if os.path.isfile(instance) or os.path.isdir(instance):
                        instances.append(instance)
                    else:
                        if not instances_must_exist:
                            instances.append(instance)
                        logging.getLogger('benchmark').warning(str(instance) + " does not exist")
        elif os.path.isdir(benchmark_config["instances"]):
            for root, directories, filenames in os.walk(benchmark_config["instances"]):
                for filename in filenames: 
                    instances.append(os.path.join(root,filename))
        if isinstance(instance_slice, slice):
            return instances[instance_slice]
        if isinstance(instance_slice, list):
            return [instances[i] for i in instance_slice]
        return instances

    @staticmethod
    def parse_slice(slice_string):
        if (slice_string is None):
            return None
        if "," in slice_string:
            return [int(x) for x in slice_string.split(",")]

        split = slice_string.split(":")
        if len(split) == 1:
            start = int(split[0]) if split[0] != "" else 0
            stop = (int(split[0]) + 1) if split[0] != "" else None
            step = 1
        elif len(split) == 2:
            start = int(split[0]) if split[0] != "" else 0
            stop = int(split[1]) if split[1] != "" else None
            step = 1
        elif len(split) == 3:
            start = int(split[0]) if split[0] != "" else 0
            stop = int(split[1]) if split[1] != "" else None
            step = int(split[2]) if split[2] != "" else 1
        return slice(start, stop, step)


def make_path(path, root):
    path = path.strip()
    if not os.path.isabs(path):
        path = os.path.join(root, path)
    if os.path.exists(path):
        return os.path.abspath(path)
    return None

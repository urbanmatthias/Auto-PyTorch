import os
from autoPyTorch.utils.config.config_option import ConfigOption, to_bool
from autoPyTorch.pipeline.base.pipeline_node import PipelineNode
import numpy as np
import scipy.stats
import logging
import json
import heapq

class ComputeSpeedup(PipelineNode):

    def fit(self, pipeline_config, trajectories, optimize_metrics, instance):
        if not pipeline_config["show_speedup_plot"]:
            return {"speedup_trajectories": None}

        speedup_trajectories = speedup_sampling(pipeline_config, trajectories, optimize_metrics, instance)

        return {"speedup_trajectories": speedup_trajectories}

    def get_pipeline_config_options(self):
        options = [
            ConfigOption('show_speedup_plot', default=None, type=str)
        ]
        return options


def speedup_sampling(pipeline_config, trajectories, optimize_metrics, instance, num_samples=1000):
    reference = pipeline_config["show_speedup_plot"]

    plot_logs = pipeline_config['plot_logs'] or optimize_metrics
    speedup_trajectories = dict()
    for metric_name in plot_logs:
        for i in range(num_samples):

            # sample trajectories
            sampled_trajectories = dict()
            for prefix in pipeline_config["prefixes"]:
                trajectory_name = ("%s_%s" % (prefix, metric_name)) if prefix else metric_name
                if trajectory_name not in trajectories:
                    continue
                for config_name, run_trajectories in trajectories[trajectory_name].items():
                    sampled = np.random.choice(run_trajectories)
                    sampled_trajectories[("%s: %s" % (prefix, config_name)) if prefix else config_name] = sampled
            
            #insert
            all_speedup_data, all_speedup_times = compute_speedup(sampled_trajectories, reference, pipeline_config["xscale"] != "linear")
            for label in all_speedup_data.keys():
                prefix = label.split(": ")[0] if label.split(": ")[0] in pipeline_config["prefixes"] else None
                config_name = label.split(": ")[1] if prefix else label
                trajectory_name = ("%s_%s" % (prefix, metric_name)) if prefix else metric_name

                if trajectory_name not in speedup_trajectories:
                    speedup_trajectories[trajectory_name] = dict()
                if config_name not in speedup_trajectories[trajectory_name]:
                    speedup_trajectories[trajectory_name][config_name] = list()
                speedup_trajectories[trajectory_name][config_name].append({
                    "losses": all_speedup_data[label],
                    "values": all_speedup_data[label],
                    "times_finished": all_speedup_times[label]
                })
    return speedup_trajectories
            

def compute_speedup(sampled_trajectories, reference_label, x_log_scale):
    reference_data = sampled_trajectories[reference_label]["losses"]
    reference_times = sampled_trajectories[reference_label]["times_finished"]

    all_speedup_data = dict()
    all_speedup_times = dict()

    for compare_label, d in sampled_trajectories.items():
        compare_data = d["losses"]
        compare_times = d["times_finished"]

        compare_pointer = 0
        reference_pointer = 0

        speedup_data = []
        speedup_times = []
        while True:
            if reference_data[reference_pointer] > compare_data[compare_pointer]:
                if reference_times[reference_pointer] > 0:
                    speedup_data.append(reference_times[reference_pointer] / compare_times[compare_pointer])
                    speedup_times.append(reference_times[reference_pointer])

                while reference_pointer < len(reference_times) and reference_data[reference_pointer] > compare_data[compare_pointer]:
                    reference_pointer += 1
                if reference_pointer >= len(reference_times):
                    speedup_data.append(reference_times[reference_pointer - 1] / compare_times[compare_pointer])
                    speedup_times.append(reference_times[reference_pointer - 1] - 1)
                    break
                
                if reference_times[reference_pointer] > 0:
                    speedup_data.append(reference_times[reference_pointer] / compare_times[compare_pointer])
                    speedup_times.append(reference_times[reference_pointer] - 1)

            elif reference_data[reference_pointer] < compare_data[compare_pointer]:

                while compare_pointer < len(compare_times) and reference_data[reference_pointer] < compare_data[compare_pointer]:
                    compare_pointer += 1
                if compare_pointer >= len(compare_times):
                    speedup_data.append(reference_times[reference_pointer] / compare_times[compare_pointer - 1])
                    speedup_times.append(reference_times[reference_pointer] - 1)
                    break

            else:
                if reference_times[reference_pointer] > 0:
                    speedup_data.append(reference_times[reference_pointer] / compare_times[compare_pointer])
                    speedup_times.append(reference_times[reference_pointer])

                reference_pointer += 1
                compare_pointer += 1

                if compare_pointer >= len(compare_times) or reference_pointer >= len(reference_times):
                    break
                if compare_times[compare_pointer] > 0:
                    speedup_data.append(reference_times[reference_pointer] / compare_times[compare_pointer])
                    speedup_times.append(reference_times[reference_pointer] - 1)
        speedup_data, speedup_times = interpolate(speedup_data, speedup_times, x_log_scale)
        all_speedup_data[compare_label] = speedup_data
        all_speedup_times[compare_label] = speedup_times
    return all_speedup_data, all_speedup_times


def interpolate(speedup_data, speedup_times, x_log_scale, num=20):
    min_time, max_time = speedup_times[0], speedup_times[-1]
    if x_log_scale:
        min_time, max_time = np.log(min_time), np.log(max_time)
    linspace = np.linspace(min_time, max_time, num=num)
    linspace = np.exp(linspace) if x_log_scale else linspace

    pointer = 0

    final_speedup_data = []
    final_speedup_times = []

    for x in linspace:
        while pointer < len(speedup_times) and speedup_times[pointer] < x:
            final_speedup_data.append(speedup_data[pointer])
            final_speedup_times.append(speedup_times[pointer])
            pointer += 1

        if pointer >= len(speedup_times):
            break
        
        if pointer > 0:
            prog = x - speedup_times[pointer - 1]
            tot = speedup_times[pointer] - speedup_times[pointer - 1]
            inc = (prog / tot) *  (speedup_data[pointer] - speedup_data[pointer - 1])
            
            final_speedup_data.append(speedup_data[pointer - 1] + inc)
            final_speedup_times.append(x)

    return final_speedup_data, final_speedup_times



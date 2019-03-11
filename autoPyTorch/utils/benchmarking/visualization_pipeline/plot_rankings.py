from autoPyTorch.pipeline.base.pipeline_node import PipelineNode
from autoPyTorch.utils.benchmarking.visualization_pipeline.plot_trajectories import plot
import os
import logging
import numpy as np

class PlotRankings(PipelineNode):
    def fit(self, pipeline_config, trajectories, train_metrics):
        plot(pipeline_config, trajectories, train_metrics, "ranking", plot_ranking)
        return dict()

def plot_ranking(instance_name, metric_name, prefixes, trajectories, agglomeration, scale_uncertainty, font_size, plt):
    assert instance_name == "ranking"
    cmap = plt.get_cmap('jet')
    trajectory_names_to_prefix = {(("%s_%s" % (prefix, metric_name)) if prefix else metric_name): prefix
        for prefix in prefixes}
    trajectory_names = [t for t in trajectory_names_to_prefix.keys() if t in trajectories]

    # save pointers for each trajectory to iterate over them simultaneously
    trajectory_pointers = {(config, name): {instance: ([0] * len(run_trajectories))
        for instance, run_trajectories in instance_trajectories.items()}
        for name in trajectory_names
        for config, instance_trajectories in trajectories[name].items()}
    trajectory_values = {(config, name): {instance: ([None] * len(run_trajectories))
        for instance, run_trajectories in instance_trajectories.items()}
        for name in trajectory_names
        for config, instance_trajectories in trajectories[name].items()}

    # data to plot
    center = {(config, name): [] for name in trajectory_names for config in trajectories[name].keys()}
    upper = {(config, name): [] for name in trajectory_names for config in trajectories[name].keys()}
    lower = {(config, name): [] for name in trajectory_names for config in trajectories[name].keys()}
    finishing_times = []
    plot_empty = True

    # iterate simultaneously over all trajectories with increasing finishing times
    while any(trajectory_pointers[(config, name)][instance][j] < len(instance_trajectories[instance][j]["losses"])
            for name in trajectory_names
            for config, instance_trajectories in trajectories[name].items()
            for instance, run_trajectories in instance_trajectories.items()
            for j in range(len(run_trajectories))):

        # get trajectory with lowest finishing time
        times_finished, current_config, current_name, current_instance, trajectory_id = min(
            (run_trajectories[j]["times_finished"][trajectory_pointers[(config, name)][instance][j]], config, name, instance, j)
            for name in trajectory_names
            for config, instance_trajectories in trajectories[name].items()
            for instance, run_trajectories in instance_trajectories.items()
            for j in range(len(run_trajectories))
            if trajectory_pointers[(config, name)][instance][j] < len(instance_trajectories[instance][j]["losses"]))

        # update trajectory values and pointers
        current_trajectory = trajectories[current_name][current_config][current_instance][trajectory_id]
        current_pointer = trajectory_pointers[(current_config, current_name)][current_instance][trajectory_id]
        current_value = current_trajectory["losses"][current_pointer]

        trajectory_values[(current_config, current_name)][current_instance][trajectory_id] = current_value * (-1 if current_trajectory["flipped"] else 1)
        trajectory_pointers[(current_config, current_name)][current_instance][trajectory_id] += 1

        # calculate ranks
        values = to_dict([(instance, (config, name), value)
            for (config, name), instance_values in trajectory_values.items()
            for instance, values in instance_values.items()
            for value in values if value is not None])
        sorted_values = {instance: sorted(map(lambda x: x[1], v), reverse=True) for instance, v in values.items()}  # configs sorted by value
        ranks = {instance: {k: [sorted_values[instance].index(value) for config_name, value in v if config_name == k] for k in center.keys()} for instance, v in values.items()}
        ranks = to_dict([(k, r) for rank_dict in ranks.values() for k, r in rank_dict.items()])

        # populate plotting data
        for key in center.keys():
            r = [i for j in  ranks[key] for i in j]
            if not r:
                center[key].append(float("nan"))
                lower[key].append(float("nan"))
                upper[key].append(float("nan"))
            elif agglomeration == "median":
                center[key].append(np.median(r))
                lower[key].append(np.percentile(r, int(50 - scale_uncertainty * 25)))
                upper[key].append(np.percentile(r, int(50 + scale_uncertainty * 25)))
            elif agglomeration == "mean":
                center[key].append(np.mean(r))
                lower[key].append(-1 * scale_uncertainty * np.std(r) + center[key][-1])
                upper[key].append(scale_uncertainty * np.std(r) + center[key][-1])
        finishing_times.append(times_finished)
        plot_empty = False
    
    if plot_empty:
        return False
    
    # do the plotting
    for i, (config, name) in enumerate(center.keys()):
        prefix = trajectory_names_to_prefix[name]
        label = ("%s: %s" % (prefix, config)) if prefix else config
        color = cmap(i / len(center))
        plt.plot(finishing_times, center[(config, name)], color=color, label=label)
        color = (color[0], color[1], color[2], 0.5)
        plt.fill_between(finishing_times, lower[(config, name)], upper[(config, name)], step=None, color=[color])

    # setup labels, legend etc.
    plt.xlabel('wall clock time [s]', fontsize=font_size)
    plt.ylabel('ranking ' + metric_name, fontsize=font_size)
    plt.legend(loc='best', prop={'size': font_size})
    plt.title("Ranking", fontsize=font_size)
    plt.xscale("log")
    plt.xlim((50, None))
    return True

def to_dict(tuple_list):
    result = dict()
    for v in tuple_list:
        a = v[0]
        b = v[1:]
        if len(b) == 1:
            b = b[0]
        if a not in result:
            result[a] = list()
        result[a].append(b)
    return result

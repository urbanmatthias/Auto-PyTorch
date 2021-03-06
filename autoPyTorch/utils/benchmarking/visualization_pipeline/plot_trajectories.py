import os
from autoPyTorch.utils.config.config_option import ConfigOption, to_bool
from autoPyTorch.pipeline.base.pipeline_node import PipelineNode
import numpy as np
import scipy.stats
import logging
import json
import heapq

class PlotTrajectories(PipelineNode):

    def fit(self, pipeline_config, trajectories, speedup_trajectories, optimize_metrics, instance):
        global LABEL_RENAME
        if isinstance(pipeline_config["label_rename"], dict):
            LABEL_RENAME = pipeline_config["label_rename"]
            LABEL_RENAME["LABEL_RENAME_SET_BY_JSON"] = True
        if not pipeline_config["skip_dataset_plots"]:
            plot(pipeline_config, trajectories, optimize_metrics, instance, process_trajectory, plot_trajectory, "plot")
        
            if speedup_trajectories:
                plot(dict(pipeline_config, agglomeration="gmean", step=False, yscale="log"),
                     speedup_trajectories, optimize_metrics, instance, process_trajectory, plot_trajectory, "speedup")

        return {"trajectories": trajectories, "optimize_metrics": optimize_metrics}
    

    def get_pipeline_config_options(self):
        options = [
            ConfigOption('plot_logs', default=None, type='str', list=True),
            ConfigOption('output_folder', default=None, type='directory'),
            ConfigOption('agglomeration', default='mean', choices=['mean', 'median', 'mean+sem']),
            ConfigOption('scale_uncertainty', default=1, type=float),
            ConfigOption('font_size', default=12, type=int),
            ConfigOption('prefixes', default=["val"], list=True, choices=["", "train", "val", "test", "ensemble", "ensemble_test"]),
            ConfigOption('label_rename', default=False, type=to_bool),
            ConfigOption('skip_dataset_plots', default=False, type=to_bool),
            ConfigOption('plot_markers', default=False, type=to_bool),
            ConfigOption('plot_individual', default=False, type=to_bool),
            ConfigOption('plot_type', default="values", type=str, choices=["values", "losses"]),
            ConfigOption('xscale', default='log', type=str),
            ConfigOption('yscale', default='linear', type=str),
            ConfigOption('xmin', default=None, type=float),
            ConfigOption('xmax', default=None, type=float),
            ConfigOption('ymin', default=None, type=float),
            ConfigOption('ymax', default=None, type=float),
            ConfigOption('value_multiplier', default=1, type=float),
            ConfigOption('hide_legend', default=False, type=to_bool),
            ConfigOption('step', default=True, type=to_bool)
        ]
        return options


def plot(pipeline_config, trajectories, optimize_metrics, instance, process_fnc, plot_fnc, filename_suffix):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    extension = "pdf"

    plot_logs = pipeline_config['plot_logs'] or optimize_metrics
    output_folder = pipeline_config['output_folder']
    instance_name = os.path.basename(instance).split(".")[0]

    if output_folder and not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # iterate over all incumbent trajectories for each metric
    for i, metric_name in enumerate(plot_logs):
        
        # prepare pdf
        if output_folder is not None:
            pdf_destination = os.path.join(output_folder, instance_name + '_' + metric_name + '_' + filename_suffix + '.' + extension)
            pp = PdfPages(pdf_destination)

        # create figure
        figure = plt.figure(i)
        plot_empty, plot_data = process_fnc(instance_name=instance_name,
                                            metric_name=metric_name,
                                            prefixes=pipeline_config["prefixes"],
                                            trajectories=trajectories,
                                            plot_type=pipeline_config["plot_type"],
                                            agglomeration=pipeline_config["agglomeration"],
                                            scale_uncertainty=pipeline_config['scale_uncertainty'],
                                            value_multiplier=pipeline_config['value_multiplier'],
                                            cmap=plt.get_cmap('jet'),
                                            significance_reference=pipeline_config["show_significance_plot"])
        if plot_empty:
            logging.getLogger('benchmark').warn('Not showing empty plot for ' + instance)
            plt.close(figure)
            continue

        plot_fnc(plot_data=plot_data,
                instance_name=instance_name,
                metric_name=metric_name,
                font_size=pipeline_config["font_size"],
                do_label_rename=pipeline_config['label_rename'],
                plt=plt,
                plot_individual=pipeline_config["plot_individual"],
                plot_markers=pipeline_config["plot_markers"],
                agglomeration=pipeline_config["agglomeration"],
                hide_legend=pipeline_config["hide_legend"],
                step=pipeline_config["step"])
        
        plt.xscale(pipeline_config["xscale"])
        plt.yscale(pipeline_config["yscale"])
        plt.xlim((pipeline_config["xmin"], pipeline_config["xmax"]))
        plt.ylim((pipeline_config["ymin"], pipeline_config["ymax"]))

        # show or save
        if output_folder is None:
            logging.getLogger('benchmark').info('Showing plot for ' + instance)
            plt.show()
        else:
            logging.getLogger('benchmark').info('Saving plot for ' + instance + ' at ' + pdf_destination)
            pp.savefig(figure)
            pp.close()
            plt.close(figure)


def process_trajectory(instance_name, metric_name, prefixes, trajectories, plot_type, agglomeration,
                       scale_uncertainty, value_multiplier, cmap, significance_reference):
    # iterate over the incumbent trajectories of the different runs
    linestyles = ['-', '--', '-.', ':']
    plot_empty = True
    plot_data = dict()
    for p, prefix in enumerate(prefixes):
        trajectory_name = ("%s_%s" % (prefix, metric_name)) if prefix else metric_name
        linestyle = linestyles[p % len(linestyles)]
        if trajectory_name not in trajectories:
            continue

        config_trajectories = trajectories[trajectory_name]
        for i, (config_name, trajectory) in enumerate(config_trajectories.items()):
            color = cmap((i *len(prefixes) + p) / (len(config_trajectories) * len(prefixes)))

            trajectory_pointers = [0] * len(trajectory)  # points to current entry of each trajectory
            trajectory_values = [None] * len(trajectory)  # list of current values of each trajectory
            individual_trajectories = [[] for _ in range(len(trajectory))]
            individual_times_finished = [[] for _ in range(len(trajectory))]
            heap = [(trajectory[j]["times_finished"][0], j) for j in range(len(trajectory))]
            heapq.heapify(heap)
            # progress = 0
            # total = sum(len(trajectory[j]["times_finished"]) for j in range(len(trajectory)))

            # data to plot
            center = []
            lower = []
            upper = []
            finishing_times = []
            # print("Calculate plot data for instance %s and trajectory %s and config %s" % (instance_name, trajectory_name, config_name))

            # iterate simultaneously over all trajectories with increasing finishing times
            while heap:

                # get trajectory with lowest finishing times
                times_finished, trajectory_id = heapq.heappop(heap)
                current_trajectory = trajectory[trajectory_id]

                # update trajectory values and pointers
                trajectory_values[trajectory_id] = current_trajectory[plot_type][trajectory_pointers[trajectory_id]]
                individual_trajectories[trajectory_id].append(trajectory_values[trajectory_id])
                individual_times_finished[trajectory_id].append(times_finished)
                trajectory_pointers[trajectory_id] += 1
                if trajectory_pointers[trajectory_id] < len(current_trajectory[plot_type]):
                    heapq.heappush(heap,
                        (trajectory[trajectory_id]["times_finished"][trajectory_pointers[trajectory_id]], trajectory_id)
                    )

                # progress += 1
                # print("Progress:", (progress / total) * 100, " " * 20, end="\r" if progress != total else "\n")

                # populate plotting data
                if any(v is None for v in trajectory_values):
                    continue
                if finishing_times and np.isclose(times_finished, finishing_times[-1]):
                    [x.pop() for x in [center, upper, lower, finishing_times]]
                values = [v * value_multiplier for v in trajectory_values if v is not None]
                if agglomeration == "median":
                    center.append(np.median(values))
                    lower.append(np.percentile(values, int(50 - scale_uncertainty * 25)))
                    upper.append(np.percentile(values, int(50 + scale_uncertainty * 25)))
                elif agglomeration == "mean":
                    center.append(np.mean(values))
                    lower.append(-1 * scale_uncertainty * scipy.stats.sem(values) + center[-1])
                    upper.append(scale_uncertainty * scipy.stats.sem(values) + center[-1])
                elif agglomeration == "mean+sem":
                    center.append(np.mean(values))
                    lower.append(-1 * scale_uncertainty * scipy.stats.sem(values) + center[-1])
                    upper.append(scale_uncertainty * scipy.stats.sem(values) + center[-1])
                elif agglomeration == "gmean":
                    center.append(scipy.stats.gmean(values))
                    lower.append(np.exp(-1 * scale_uncertainty * np.std(np.log(values)) +  np.log(center[-1])))
                    upper.append(np.exp(scale_uncertainty * np.std(np.log(values)) + np.log(center[-1])))
                finishing_times.append(times_finished)
                plot_empty = False
            label = ("%s: %s" % (prefix, config_name)) if prefix else config_name

            plot_data[label] = {
                "individual_trajectories": individual_trajectories,
                "individual_times_finished": individual_times_finished,
                "color": color,
                "linestyle": linestyle,
                "center": center,
                "lower": lower,
                "upper": upper,
                "finishing_times": finishing_times
            }
    return plot_empty, plot_data

def plot_trajectory(plot_data, instance_name, metric_name, font_size, do_label_rename, plt, plot_individual, plot_markers, agglomeration, hide_legend, step):
    plot_p_values = any(("p_values" in d and d["p_values"]) for d in plot_data.values())
    gridshape = (5, 1) if plot_p_values else (4,1)
    if plot_p_values:
        ax2 = plt.subplot2grid(gridshape, (4, 0))
        ax2.set_xscale("log")
        ax2.set_ylim([-0.01,0.2])
    ax1 = plt.subplot2grid(gridshape, (0, 0), rowspan=4)
    if plot_p_values:
        ax1.get_xaxis().set_visible(False)
    
    for label, d in plot_data.items():

        if do_label_rename:
            label = label_rename(label)
        
        if plot_individual and d["individual_trajectories"] and d["individual_times_finished"]:
            for individual_trajectory, individual_times_finished in zip(d["individual_trajectories"], d["individual_times_finished"]):
                plt.step(individual_times_finished, individual_trajectory, color=d["color"], where='post', linestyle=":", marker="x" if plot_markers else None)
        
        if step:
            plt.step(d["finishing_times"], d["center"], color=d["color"], label=label, where='post', linestyle=d["linestyle"], marker="o" if plot_markers else None)
            plt.fill_between(d["finishing_times"], d["lower"], d["upper"], step="post", color=[(d["color"][0], d["color"][1], d["color"][2], 0.5)])
        else:
            plt.plot(d["finishing_times"], d["center"], color=d["color"], label=label, linestyle=d["linestyle"], marker="o" if plot_markers else None)
            plt.fill_between(d["finishing_times"], d["lower"], d["upper"], color=[(d["color"][0], d["color"][1], d["color"][2], 0.5)])
        
        if "p_values" in d and d["p_values"]:
            ax2.plot(d["finishing_times"], d["p_values"], color=d["color"], linestyle=d["linestyle"])
            ax2.plot(d["finishing_times"], [0.05] * len(d["finishing_times"]), color="black")
    xlabel = 'wall clock time [s]'
    ylabel = agglomeration + " " + metric_name

    (ax2 if plot_p_values else ax1).set_xlabel(xlabel if not do_label_rename else label_rename(xlabel), fontsize=font_size)
    plt.ylabel(ylabel if not do_label_rename else label_rename(ylabel), fontsize=font_size)
    if not hide_legend:
        plt.legend(loc='best', prop={'size': font_size})
    plt.title(instance_name if not do_label_rename else label_rename(instance_name), fontsize=font_size)


LABEL_RENAME = {"LABEL_RENAME_SET_BY_JSON": False}
def label_rename(label):
    if label not in LABEL_RENAME and LABEL_RENAME["LABEL_RENAME_SET_BY_JSON"]:
        LABEL_RENAME[label] = label
    elif label not in LABEL_RENAME:
        rename = input("Rename label %s to? (Leave empty for no rename) " % label)
        LABEL_RENAME[label] = rename if rename else label
    return LABEL_RENAME[label]

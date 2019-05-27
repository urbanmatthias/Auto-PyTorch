from hpbandster.core.result import logged_results_to_HBS_result
import argparse
import heapq
import os
import sys

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(
    __file__, '..', '..', 'submodules/HpBandSter')))


sum_of_weights_history = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run benchmarks for autonet.')
    parser.add_argument("weight_history_files", type=str,
                        nargs="+", help="The files to plot")

    args = parser.parse_args()

    weight_deviation_history = list()
    weight_deviation_timestamps = list()

    all_plot_data = list()
    all_additional_info = list()

    additional_info_names = set()

    #iterate over all files and lot the weight history
    for i, weight_history_file in enumerate(args.weight_history_files):
        print(i)
        plot_data = dict()
        additional_info = dict()
        current_read = False
        with open(weight_history_file, "r") as f:
            for line in f:

                #read the data
                line = line.split("\t")
                if len(line) == 1 or not line[-1].strip():
                    continue
                data = line[-1]
                data = list(map(float, map(str.strip, data.split(","))))
                title = "\t".join(line[:-1]).strip()

                # and save it later for plotting
                if not current_read:
                    plot_data[title] = data
                else:
                    additional_info[title] = data
                    additional_info_names |= set([title])

                current_read = current_read or title == "current"

        # only show labels for top datasets
        sorted_keys = sorted(plot_data.keys(
        ), key=lambda x, d=plot_data: -d[x][-1] if x != "current" else -float("inf"))

        # parse results to get the timestamps for the weights
        x_axis = []
        try:
            r = logged_results_to_HBS_result(
                os.path.dirname(weight_history_file))
            sampled_configs = set()
            budget_consumed = 0
            for run in sorted(r.get_all_runs(), key=(lambda run: run.time_stamps["submitted"])):
                print(run.config_id, run.budget)
                if run.config_id not in sampled_configs:
                    x_axis.append(budget_consumed)
                sampled_configs |= set([run.config_id])
                budget_consumed += run.budget
        except Exception as e:
            print(e)
            continue

        for title, data in sorted(plot_data.items()):
                plot_data[title] = x_axis, data
        for title, data in sorted(additional_info.items()):
            additional_info[title] = x_axis, data

        all_plot_data.append(plot_data)
        all_additional_info.append(additional_info)

    # do the plotting
    num_subplots = (1 + len(additional_info_names)) * 100
    gridshape = (max(1, len(additional_info_names) * 4), 1)
    ax1 = plt.subplot2grid(
        gridshape, (0, 0), rowspan=max(1, len(additional_info) * 3))

    for plot_data in all_plot_data:
        for title, (x_axis, data) in plot_data.items():
            plt.plot(x_axis, data[:len(x_axis)], lw=0.1 if title != "current" else 0.5,
                     color="red" if title == "current" else "blue")

    plt.title("weight history")
    plt.xscale("log")

    for i, title in enumerate(sorted(additional_info_names)):
        ax = plt.subplot2grid(
            gridshape, (int(gridshape[0] * 3/4) + i, 0), sharex=ax1)
        for j, additional_info in enumerate(all_additional_info):
            x_axis, data = additional_info[title]
            plt.scatter(x_axis, data[:len(
                x_axis)], label=title if j == 0 else None, marker="x", color="orange")
        plt.legend(loc='upper left')

    plt.show()

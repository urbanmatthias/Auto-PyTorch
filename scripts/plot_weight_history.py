import matplotlib.pyplot as plt
import argparse
import numpy as np
import os, sys

hpbandster = os.path.abspath(os.path.join(__file__, '..', '..', 'submodules', 'HpBandSter'))
sys.path.append(hpbandster)

from hpbandster.core.result import logged_results_to_HBS_result

sum_of_weights_history = None


if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Run benchmarks for autonet.')
    parser.add_argument("--only_summary", action="store_true", help="The maximum number of configs in the legend")
    parser.add_argument("--max_legend_size", default=5, type=int, help="The maximum number of datasets in the legend")
    parser.add_argument("--num_consider_in_summary", default=15, type=int, help="The number of datasets considered in the summary")
    parser.add_argument("weight_history_files", type=str, nargs="+", help="The files to plot")

    args = parser.parse_args()
    
    weight_deviation_history = list()
    weight_deviation_timestamps = list()
    for i, weight_history_file in enumerate(args.weight_history_files):
        print(i)
        plot_data = dict()
        with open(weight_history_file, "r") as f:
            for line in f:
                line = line.split("\t")
                if len(line) == 1 or not line[-1].strip():
                    continue
                data = line[-1]
                data = list(map(float, map(str.strip, data.split(","))))
                title = "\t".join(line[:-1]).strip()
                plot_data[title] = data
        sorted_keys = sorted(plot_data.keys(), key=lambda x, d=plot_data: -d[x][-1] if x != "current" else -float("inf"))
        show_labels = set(sorted_keys[:args.max_legend_size])
        consider_in_summary = set(sorted_keys[:args.num_consider_in_summary])

        x_axis = []
        x_scale = "linear"
        try:
            r = logged_results_to_HBS_result(os.path.dirname(weight_history_file))
            sampled_configs = set()
            for run in r.get_all_runs():
                if run.config_id not in sampled_configs:
                    x_axis.append(run.time_stamps["submitted"])
                sampled_configs |= set([run.config_id])
            x_scale = "log"
        except Exception as e:
            print(e)

        if not args.only_summary:
            for title, data in sorted(plot_data.items()):
                    plt.plot(x_axis or list(range(len(data))), data,
                        label=title if title in show_labels else None,
                        linestyle="-." if title == "current" else ("-" if title in show_labels else ":"),
                        marker="x")

            plt.legend(loc='best')
            plt.title(weight_history_file)
            plt.xscale(x_scale)
            plt.show()
        
        for title, data in plot_data.items():
            if title in consider_in_summary:
                weight_deviation_history.append([abs(d - data[-1]) for d in data])

                if not x_axis:
                    weight_deviation_timestamps = None

                if weight_deviation_timestamps is not None:
                    weight_deviation_timestamps.append(x_axis)
    
    weight_deviation_history = np.array(weight_deviation_history)

    if weight_deviation_timestamps is None:
        plt.plot(range(weight_deviation_history.shape[1]), np.mean(weight_deviation_history, axis=0))
        plt.title("weight deviation over time")
        plt.show()
        exit(0)
    
    weight_deviation_timestamps = np.array(weight_deviation_timestamps)
    history_pointers = [0] * weight_deviation_timestamps.shape[0]
    current_values = [None] * weight_deviation_timestamps.shape[0]

    times = []
    values = []
    while any(p < weight_deviation_timestamps.shape[1] for p in history_pointers):
        time, v, i, p = min((weight_deviation_timestamps[i, p], weight_deviation_history[i,p], i, p)
            for i, p in enumerate(history_pointers) if p < weight_deviation_timestamps.shape[1])
        current_values[i] = v
        values.append(np.mean([v for v in current_values if v is not None]))
        times.append(time)
        history_pointers[i] += 1

    plt.plot(times, values, marker="x")
    plt.title("weight deviation over time")
    plt.show()
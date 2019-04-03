import matplotlib.pyplot as plt
import argparse
import numpy as np
import os, sys

sum_of_weights_history = None

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Run benchmarks for autonet.')
    parser.add_argument("--only_summary", action="store_true", help="The maximum number of configs in the legend")
    parser.add_argument("--max_legend_size", default=5, type=int, help="The maximum number of datasets in the legend")
    parser.add_argument("--num_consider_in_summary", default=15, type=int, help="The number of datasets considered in the summary")
    parser.add_argument("weight_history_files", type=str, nargs="+", help="The files to plot")

    args = parser.parse_args()
    
    weight_deviation_over_time = list()
    for weight_history_file in args.weight_history_files:
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

        if not args.only_summary:
            for title, data in sorted(plot_data.items()):
                    plt.plot(list(range(len(data))), data,
                        label=title if title in show_labels else None,
                        linestyle="-." if title == "current" else ("-" if title in show_labels else ":"))


            plt.legend(loc='best')
            plt.title(weight_history_file)
            plt.show()
        
        for title, data in plot_data.items():
            if title in consider_in_summary:
                weight_deviation_over_time.append([abs(d - data[-1]) for d in data])
    
    weight_deviation_over_time = np.array(weight_deviation_over_time)
    print(weight_deviation_over_time.shape)
    plt.plot(range(weight_deviation_over_time.shape[1]), np.mean(weight_deviation_over_time, axis=0))
    plt.title("weight deviation over time")
    plt.show()